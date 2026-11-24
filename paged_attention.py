from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from model import (
    CustomLlamaForCausalLM,
    ModelConfig,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUMLP,
    apply_rotary_pos_emb,
    repeat_kv,
)


# ==============================================================================
# 1. Custom Triton Kernels for PagedAttention (Turing sm_75 Optimized)
# ==============================================================================


@triton.jit
def _reshape_and_cache_kernel(
    k_ptr,  # [num_tokens, num_kv_heads, head_dim]
    v_ptr,  # [num_tokens, num_kv_heads, head_dim]
    k_cache_ptr,  # [num_blocks, num_kv_heads, block_size, head_dim]
    v_cache_ptr,  # [num_blocks, num_kv_heads, block_size, head_dim]
    slot_mapping_ptr,  # [num_tokens] (int32: physical_block_id * block_size + block_offset)
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_cb,
    stride_ch,
    stride_cs,
    stride_cd,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused Triton kernel that scatters newly computed K and V vectors into
    non-contiguous physical blocks in the Paged KV Cache pool.
    Grid: (num_tokens, num_kv_heads)
    """
    token_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    # Negative slot_idx indicates a padded dummy token (e.g. during CUDA Graph replay)
    if slot_idx < 0:
        return

    physical_block_id = slot_idx // block_size
    block_offset = slot_idx % block_size

    offs_d = tl.arange(0, BLOCK_D)
    k_offs = token_idx * stride_kt + kv_head_idx * stride_kh + offs_d * stride_kd
    v_offs = token_idx * stride_vt + kv_head_idx * stride_vh + offs_d * stride_vd

    k_val = tl.load(k_ptr + k_offs)
    v_val = tl.load(v_ptr + v_offs)

    cache_offs = (
        physical_block_id * stride_cb
        + kv_head_idx * stride_ch
        + block_offset * stride_cs
        + offs_d * stride_cd
    )
    tl.store(k_cache_ptr + cache_offs, k_val)
    tl.store(v_cache_ptr + cache_offs, v_val)


@triton.jit
def _paged_attention_decode_kernel(
    out_ptr,  # [batch_size, num_heads, head_dim]
    q_ptr,  # [batch_size, num_heads, head_dim]
    k_cache_ptr,  # [num_blocks, num_kv_heads, block_size, head_dim]
    v_cache_ptr,  # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables_ptr,  # [batch_size, max_num_blocks_per_seq] (int32)
    context_lens_ptr,  # [batch_size] (int32)
    sm_scale,  # float32: 1.0 / sqrt(head_dim)
    stride_qb,
    stride_qh,
    stride_qd,
    stride_ob,
    stride_oh,
    stride_od,
    stride_cb,
    stride_ch,
    stride_cs,
    stride_cd,
    stride_bt_b,
    stride_bt_s,
    num_kv_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Phase 4 PagedAttention Decode Kernel tuned for GTX 1650 (Turing sm_75).

    - Uses Online Softmax in fp32 for numerical accuracy while keeping K/V tiles in fp16.
    - Tile size is [BLOCK_SIZE=16, BLOCK_D=64] (only 2 KB per K/V block), respecting
      sm_75's 64 KB shared memory budget and avoiding Ampere-only (sm_80+) instructions.
    - Supports Grouped-Query Attention (GQA) via `kv_head_idx = head_idx // num_kv_groups`.
    - Reads `context_lens` and `block_tables` directly from GPU pointers so the kernel
      can be captured inside a static PyTorch CUDA Graph (Phase 5).

    Grid: (batch_size, num_heads)
    """
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // num_kv_groups

    seq_len = tl.load(context_lens_ptr + seq_idx)
    if seq_len <= 0:
        return

    offs_d = tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, BLOCK_SIZE)

    # Load single decode query vector [BLOCK_D] into registers
    q_offs = seq_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptr + q_offs).to(tl.float32)

    # Online softmax running statistics in float32
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    num_logical_blocks = tl.cdiv(seq_len, BLOCK_SIZE)

    for logical_block_idx in range(0, num_logical_blocks):
        # Look up physical block index from the sequence's page table row
        physical_block_id = tl.load(
            block_tables_ptr + seq_idx * stride_bt_b + logical_block_idx * stride_bt_s
        )

        # Load K block of shape [BLOCK_SIZE, BLOCK_D] from physical block
        kv_block_offs = (
            physical_block_id * stride_cb
            + kv_head_idx * stride_ch
            + offs_s[:, None] * stride_cs
            + offs_d[None, :] * stride_cd
        )
        token_positions = logical_block_idx * BLOCK_SIZE + offs_s
        valid_mask = token_positions < seq_len

        k_block = tl.load(
            k_cache_ptr + kv_block_offs,
            mask=valid_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # Compute scaled dot-product scores qk: [BLOCK_SIZE]
        qk = tl.sum(q[None, :] * k_block, axis=1) * sm_scale
        qk = tl.where(valid_mask, qk, -float("inf"))

        # Online softmax update
        m_ij = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        # Load V block of shape [BLOCK_SIZE, BLOCK_D] and accumulate weighted values
        v_block = tl.load(
            v_cache_ptr + kv_block_offs,
            mask=valid_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
        m_i = m_new

    out = (acc / l_i).to(tl.float16)
    out_offs = seq_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptr + out_offs, out)


def reshape_and_cache_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache_layer: torch.Tensor,
    v_cache_layer: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Launches `_reshape_and_cache_kernel` to store K and V into the layer's physical block pool.
      k, v: [num_tokens, num_kv_heads, head_dim] (float16)
      k_cache_layer, v_cache_layer: [num_blocks, num_kv_heads, block_size, head_dim] (float16)
      slot_mapping: [num_tokens] (int32)
    """
    num_tokens, num_kv_heads, head_dim = k.shape
    block_size = k_cache_layer.shape[2]
    assert head_dim in (32, 64, 128), f"Unsupported head_dim={head_dim}"

    grid = (num_tokens, num_kv_heads)
    _reshape_and_cache_kernel[grid](
        k,
        v,
        k_cache_layer,
        v_cache_layer,
        slot_mapping,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_cache_layer.stride(0),
        k_cache_layer.stride(1),
        k_cache_layer.stride(2),
        k_cache_layer.stride(3),
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        BLOCK_D=head_dim,
        num_warps=2,
    )


def paged_attention_decode_triton(
    q: torch.Tensor,
    k_cache_layer: torch.Tensor,
    v_cache_layer: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Launches `_paged_attention_decode_kernel` for single-step token decoding across a batch.
      q: [batch_size, num_heads, head_dim] (float16)
      k_cache_layer, v_cache_layer: [num_blocks, num_kv_heads, block_size, head_dim] (float16)
      block_tables: [batch_size, max_num_blocks_per_seq] (int32)
      context_lens: [batch_size] (int32)
    Returns:
      out: [batch_size, num_heads, head_dim] (float16)
    """
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_cache_layer.shape[1]
    block_size = k_cache_layer.shape[2]
    num_kv_groups = num_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    out = torch.empty_like(q)
    grid = (batch_size, num_heads)

    _paged_attention_decode_kernel[grid](
        out,
        q,
        k_cache_layer,
        v_cache_layer,
        block_tables,
        context_lens,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        k_cache_layer.stride(0),
        k_cache_layer.stride(1),
        k_cache_layer.stride(2),
        k_cache_layer.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        num_kv_groups=num_kv_groups,
        BLOCK_SIZE=block_size,
        BLOCK_D=head_dim,
        num_warps=2,
    )
    return out


# ==============================================================================
# 2. Physical Paged KV Cache Pool & PagedLlamaForCausalLM
# ==============================================================================


class PagedKVCachePool:
    """
    Physical GPU VRAM Block Pool for PagedAttention.
    Allocates fixed-size physical blocks upfront:
      k_cache, v_cache: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
    Indexed via `BlockSpaceManager`'s virtual-to-physical page tables.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.k_cache = torch.zeros(
            (num_layers, num_blocks, num_kv_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (num_layers, num_blocks, num_kv_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )

    def gather_sequence_kv(
        self,
        layer_idx: int,
        physical_blocks: List[int],
        total_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gathers K and V for a sequence of `total_tokens` across its assigned `physical_blocks`
        (including any prefix-cached physical blocks!).
        Returns (k_seq, v_seq) of shape [1, num_kv_heads, total_tokens, head_dim].
        """
        # Select physical blocks: [num_seq_blocks, num_kv_heads, block_size, head_dim]
        k_blocks = self.k_cache[layer_idx, physical_blocks]
        v_blocks = self.v_cache[layer_idx, physical_blocks]

        # Permute to [num_kv_heads, num_seq_blocks, block_size, head_dim] and flatten tokens
        k_seq = (
            k_blocks.permute(1, 0, 2, 3)
            .reshape(1, self.num_kv_heads, -1, self.head_dim)[:, :, :total_tokens, :]
        )
        v_seq = (
            v_blocks.permute(1, 0, 2, 3)
            .reshape(1, self.num_kv_heads, -1, self.head_dim)[:, :, :total_tokens, :]
        )
        return k_seq, v_seq


class PagedLlamaAttention(nn.Module):
    """
    Phase 4 & 5 Attention Layer backed by Triton PagedAttention kernels.
    Weights are shared directly with Phase 1's `LlamaAttention`.
    """

    def __init__(self, base_attn: nn.Module, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5

        # Share Linear projection modules with base model (zero extra VRAM)
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_pool: PagedKVCachePool,
        slot_mapping: torch.Tensor,
        physical_blocks: List[int],
        total_seq_len: int,
    ) -> torch.Tensor:
        """
        Prefill (or partial Suffix Prefill when Prefix Caching hits):
          1. Computes Q, K, V only for the uncached input tokens (`q_len`).
          2. Scatters K, V into `kv_pool` using Triton `reshape_and_cache_triton`.
          3. Gathers full sequence K, V (cached prefix + new tokens = `total_seq_len`)
             from `physical_blocks` and computes causal attention.
        """
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "Prefill operates per-sequence (bsz=1)"

        q = self.q_proj(hidden_states).view(1, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scatter new K and V tokens directly into physical blocks via Triton kernel
        k_flat = k.transpose(1, 2).contiguous().view(q_len, self.num_kv_heads, self.head_dim)
        v_flat = v.transpose(1, 2).contiguous().view(q_len, self.num_kv_heads, self.head_dim)
        reshape_and_cache_triton(
            k_flat,
            v_flat,
            kv_pool.k_cache[self.layer_idx],
            kv_pool.v_cache[self.layer_idx],
            slot_mapping,
        )

        # Read full K, V (including any cached prefix blocks) from physical blocks
        k_full, v_full = kv_pool.gather_sequence_kv(
            self.layer_idx, physical_blocks, total_seq_len
        )
        k_expanded = repeat_kv(k_full, self.num_kv_groups)
        v_expanded = repeat_kv(v_full, self.num_kv_groups)

        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * self.scaling
        if q_len > 1:
            causal_mask = torch.full(
                (q_len, total_seq_len),
                torch.finfo(attn_weights.dtype).min,
                device=hidden_states.device,
                dtype=attn_weights.dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1 + (total_seq_len - q_len))
            attn_weights = attn_weights + causal_mask[None, None, :, :]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_expanded)
        attn_output = attn_output.transpose(1, 2).contiguous().view(1, q_len, self.hidden_size)
        return self.o_proj(attn_output)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_pool: PagedKVCachePool,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched Decode step using Triton `_reshape_and_cache_kernel` and
        `_paged_attention_decode_kernel` (CUDA Graph compatible).
        """
        bsz, q_len, _ = hidden_states.size()
        assert q_len == 1, "Decode step expects q_len == 1 per sequence"

        q = self.q_proj(hidden_states).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q_2d = q.squeeze(2).contiguous()  # [bsz, num_heads, head_dim]
        k_2d = k.squeeze(2).contiguous()  # [bsz, num_kv_heads, head_dim]
        v_2d = v.squeeze(2).contiguous()  # [bsz, num_kv_heads, head_dim]

        # 1. Store newly generated token's K, V into its physical block slot
        reshape_and_cache_triton(
            k_2d,
            v_2d,
            kv_pool.k_cache[self.layer_idx],
            kv_pool.v_cache[self.layer_idx],
            slot_mapping,
        )

        # 2. Compute attention directly over non-contiguous physical blocks in Triton
        attn_out = paged_attention_decode_triton(
            q_2d,
            kv_pool.k_cache[self.layer_idx],
            kv_pool.v_cache[self.layer_idx],
            block_tables,
            context_lens,
            sm_scale=self.scaling,
        )

        attn_out = attn_out.view(bsz, 1, self.hidden_size)
        return self.o_proj(attn_out)


class PagedTransformerBlock(nn.Module):
    def __init__(self, base_block: nn.Module, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = PagedLlamaAttention(base_block.self_attn, config, layer_idx)
        self.mlp = base_block.mlp
        self.input_layernorm = base_block.input_layernorm
        self.post_attention_layernorm = base_block.post_attention_layernorm

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_pool: PagedKVCachePool,
        slot_mapping: torch.Tensor,
        physical_blocks: List[int],
        total_seq_len: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_prefill(
            hidden_states,
            cos,
            sin,
            kv_pool=kv_pool,
            slot_mapping=slot_mapping,
            physical_blocks=physical_blocks,
            total_seq_len=total_seq_len,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_pool: PagedKVCachePool,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_decode(
            hidden_states,
            cos,
            sin,
            kv_pool=kv_pool,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class PagedLlamaForCausalLM(nn.Module):
    """
    Phase 4 & 5 PagedAttention Model wrapping `CustomLlamaForCausalLM`.
    Shares all weights with `CustomLlamaForCausalLM` so zero duplicate weight VRAM is used,
    while leaving `CustomLlamaForCausalLM` completely untouched for Phase 1-3 tests.
    """

    def __init__(self, base_model: CustomLlamaForCausalLM) -> None:
        super().__init__()
        self.config = base_model.config
        self.embed_tokens = base_model.embed_tokens
        self.layers = nn.ModuleList(
            [
                PagedTransformerBlock(base_model.layers[i], self.config, layer_idx=i)
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb
        self.lm_head = base_model.lm_head

    @classmethod
    def from_safetensors(
        cls,
        model_id_or_path: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> "PagedLlamaForCausalLM":
        base_model = CustomLlamaForCausalLM.from_safetensors(
            model_id_or_path, dtype=dtype, device=device
        )
        return cls(base_model)

    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_pool: PagedKVCachePool,
        slot_mapping: torch.Tensor,
        physical_blocks: List[int],
        total_seq_len: int,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer.forward_prefill(
                hidden_states,
                cos,
                sin,
                kv_pool=kv_pool,
                slot_mapping=slot_mapping,
                physical_blocks=physical_blocks,
                total_seq_len=total_seq_len,
            )

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_pool: PagedKVCachePool,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer.forward_decode(
                hidden_states,
                cos,
                sin,
                kv_pool=kv_pool,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                context_lens=context_lens,
            )

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
