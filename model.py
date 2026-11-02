from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import glob
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

###################################################################################
# Below class defins the model configuration to run pretrained llama implementation
# from huggingface, we define all the required model paramters (e.g. embedding size,
# number of layers etc.) for the model.
# hidden_size -> dimension of the input embedding
# intermediate_size -> dimension of the intermediate layer in the feed-forward network
# num_hidden_layers -> number of layers in the model
# num_attention_heads -> number of attention heads
# num_key_value_heads -> number of key-value heads
# vocab_size -> size of the vocabulary
# rms_norm_eps -> epsilon for RMS norm
# rope_theta -> theta for RoPE
# max_position_embeddings -> maximum position embeddings
# tie_word_embeddings -> whether to tie word embeddings
# qkv_bias -> whether to use bias in the query-key-value projection
# model_type -> type of the model
###################################################################################
@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = False
    qkv_bias: bool = False
    model_type: str = "llama"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained(cls, model_path: str) -> Tuple["ModelConfig", str]:
        path = Path(model_path)
        if not path.exists():
            local_dir = snapshot_download(
                repo_id=model_path,
                allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
            )
            path = Path(local_dir)

        with open(path / "config.json", "r", encoding="utf-8") as f:
            raw = json.load(f)

        model_type = raw.get("model_type", "llama")
        # Qwen2 uses bias on q_proj, k_proj, v_proj by default; LLaMA uses attention_bias (default False)
        if model_type == "qwen2":
            qkv_bias = True
        else:
            qkv_bias = bool(raw.get("attention_bias", False))

        config = cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            max_position_embeddings=raw.get("max_position_embeddings", 2048),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            qkv_bias=qkv_bias,
            model_type=model_type,
        )
        return config, str(path)


class RMSNorm(nn.Module):
    """RMSNorm computing variance in float32 for numerical stability on sm_75 (float16)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed.to(input_dtype)

#####################################################################
# RoPE implementation:
# 1. Initialize a vector of theta angles for alternative positions from the input vector, this empty vector will be of
#    size head_size/2.
# 2. In the forward pass with no gradient calculation, we expand the inv_freq vector to size [batch_size, head_size//2 ,1] 
#    and position_ids to the shape [batch_size, 1, seq_len].
# 3. We then compute the actual frequencies by matrix multiplication of the inv_freq(theta) and postions(m) to get a vector
#    of shape [batch_size, head_size//2, seq_len]. And this is concatnated to itself to make the frequency matrix size match the
#    the required shape which is [batch_size, head_size, seq_len].
# 4. We then use this matrix of m*theta calculation to calculate the cos and sin matrices, as the 2d rotation matrix is
#    [cos(theta), -sin(theta)]
#    [sin(theta), cos(theta)]
#    we can perform element wise multiplication of the query and key vectors with the cos and sin matrices to apply the RoPE.
# 5. This matrix rotation can be simplified as:
#    x_1' = x_1*cos(m*theta) - x_2*sin(m*theta)
#    x_2' = x_1*sin(m*theta) + x_2*cos(m*theta)
#    which in matrix notation can be then simplified into
#    [x_1'] = [x_1] . cos(m*theta) + [-x_2] . sin(m*theta)
#    [x_2']   [x_2]                  [ x_1]
# 6. So we create a roation function as has been done here rotate_half(x) for simpilfication
# 7. Use unsqueeze to broadcast the sin adn cos matrices over the number of heads, and then calculate the q and k matrix values
#   using the method mentioned in step 5.
#####################################################################
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) matching LLaMA / Qwen2 rotate_half layout."""

    def __init__(self, head_dim: int, max_position_embeddings: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        # creates a vector of theta angles for RoPE (1/(base^(2i/dim)) for i in [0, dim/2-1]), shape [head_dim/2]
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch_size, seq_len]
        # here we expand inv_freq to the shape [batch_size, num_heads, 1] and position_ids to the shape [batch_size, 1, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        #matrix multiplication to compute the frequency of each position
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin: [batch_size, seq_len, head_dim] -> unsqueeze(1) for broadcast over heads
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

########################################## CORE MODEL IMPLEMENTATION ###################################################################
# 1. KV Cache Implementation
#  1.1. Pre-allocates 2 5D matrices in the VRAM of dimensions [num_layers, batch_size, num_kv_heads, max_seq_len, head_dim]
#  1.2. Tensors are of the max input prompt length, so we need a pointer to fill from the start(which is the 0th position)
#       to the length of the input prompt.
#  1.3. As the seq_len is a increasing parameter in itself, we can keep generating new tokens that add to the seq_len
#       on each forward pass, taking up more of the allocated VRAM. This is why we need to use contiguous memory 
#       allocation and indexing to manage the cache
# 2. SWIGLU Layer
#  2.1. Straightformward inplementation, create layers for up, down and gate projections and them combine them using the SiLU 
#       activation function as per the formula, refer below code. 
# 3. Attention layer with GQA.
#  3.1. Multi Head Attention with Grouped-Query Attention (GQA). 
#  3.2. Create Q, K and V projections using the loaded weights and apply RoPE embeddings to Q and K.
#  3.3. Q generally has n*num_heads_kv, while K and V have heads, 1*num_heads_kv.
#  3.4. Each K and V head is shared by 'n' query heads.
#  3.5. A repeater function is setup below so that the K and V heads are duplicated 'n' times to match the number of query heads.
#  3.6. Q and expanded K are matrix multplied and then scaled.
#  3.7. Causal Mask is applied to this result to prevent the model from attending to future tokens during prefill.
#  3.8. Softmax is applied to the result to get the attention weights.
#  3.9. Attention weights are matrix multiplied with the V matrix expanded(via the repeater block) to get the output.
#  3.10. We also have an output projection which is a linear layer that projects the output of the attention layer to the
#        desired output dimensions. which is mostly the transpose of the Q,K,V projection layers.
#  3.11. The output is generally of the dimension [batch_size, seq_len, hidden_dim]
# 4. Transformer Block Implementation
#  4.1. Each trnsformer block contains atteition, swiflu, rmsnorm and residual attention and is triggered sequentially using the output from
#       the previous block.
#  4.2 Each block has a layer_idx as its identifier, it takes the input from the previous block and performs the operations
#       sequentially
# 5. Pytorch Model Implemention combining the above components
#  5.1. from_safetensor method to load weights with fp16 precision in safetensor format
#   5.1.1 This method iterates through the .safetensors file and loads the weights in fp16 precision.
#   5.1.2 Creates the model using multiple transformer layers, rmsnorm, residuals and a linear layer, and loads the 
#         safetensor weights to the corresponding layers using torch.load.
#  5.2. naive_generation method that runs inference on the model, a while loop based implmentation that runs till
#       EOS or till the output tokens reach max_new_tokens.
#############################################################################################################
class ContiguousKVCache:
    """
    Phase 1 Naive Contiguous KV Cache.
    Pre-allocates [num_layers, 2, batch_size, num_kv_heads, max_seq_len, head_dim] upfront in VRAM.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        self.max_seq_len = max_seq_len
        self.seq_len = 0
        self.k_cache = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )

    def update(
        self, layer_idx: int, k_val: torch.Tensor, v_val: torch.Tensor, start_pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = k_val.shape[2]
        end_pos = start_pos + seq_len
        self.k_cache[layer_idx, :, :, start_pos:end_pos, :] = k_val
        self.v_cache[layer_idx, :, :, start_pos:end_pos, :] = v_val
        return (
            self.k_cache[layer_idx, :, :, :end_pos, :],
            self.v_cache[layer_idx, :, :, :end_pos, :],
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats KV heads to match query heads for Grouped-Query Attention (GQA)."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network used in LLaMA / Qwen2."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Grouped-Query Attention with RoPE and Naive Contiguous KV Cache."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[ContiguousKVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v, start_pos)

        k_expanded = repeat_kv(k, self.num_kv_groups)
        v_expanded = repeat_kv(v, self.num_kv_groups)

        # Standard scaled dot-product attention
        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * self.scaling

        if q_len > 1:
            total_len = k_expanded.size(-2)
            causal_mask = torch.full(
                (q_len, total_len), torch.finfo(attn_weights.dtype).min, device=hidden_states.device, dtype=attn_weights.dtype
            )
            causal_mask = torch.triu(causal_mask, diagonal=1 + (total_len - q_len))
            attn_weights = attn_weights + causal_mask[None, None, :, :]

        # Upcast softmax to fp32 matching HF reference implementation
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_expanded)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx=layer_idx)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[ContiguousKVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, kv_cache=kv_cache, start_pos=start_pos)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CustomLlamaForCausalLM(nn.Module):
    """
    Custom pure-PyTorch LLaMA/Qwen2 Causal LM loaded directly from safetensors.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[ContiguousKVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(
                start_pos, start_pos + seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(bsz, -1)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cos, sin, kv_cache=kv_cache, start_pos=start_pos
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def from_safetensors(
        cls,
        model_id_or_path: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> "CustomLlamaForCausalLM":
        config, local_dir = ModelConfig.from_pretrained(model_id_or_path)

        # Instantiate model directly in target dtype to save RAM/VRAM
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = cls(config)
        finally:
            torch.set_default_dtype(orig_dtype)

        safetensor_files = sorted(glob.glob(f"{local_dir}/*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {local_dir}")

        state_dict = {}
        for sf_path in safetensor_files:
            raw_sd = load_file(sf_path, device="cpu")
            for k, v in raw_sd.items():
                # Strip 'model.' prefix used by HF CausalLM wrappers
                clean_key = k[len("model.") :] if k.startswith("model.") else k
                # Ignore precomputed rotary inv_freq buffers if present in checkpoint
                if "rotary_emb.inv_freq" in clean_key:
                    continue
                state_dict[clean_key] = v.to(dtype=dtype)

        if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch! Missing: {missing}, Unexpected: {unexpected}")

        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    @torch.inference_mode()
    def generate_naive(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        max_seq_len: int = 512,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Naive manual while-loop generation with a contiguous KV cache allocated upfront.
        Returns (generated_ids, prefill_last_token_logits).
        """
        bsz, prompt_len = input_ids.shape
        assert prompt_len + max_new_tokens <= max_seq_len, (
            f"Sequence length {prompt_len + max_new_tokens} exceeds pre-allocated max_seq_len {max_seq_len}"
        )

        # Allocate contiguous KV cache for max_seq_len upfront
        kv_cache = ContiguousKVCache(
            num_layers=self.config.num_hidden_layers,
            batch_size=bsz,
            num_kv_heads=self.config.num_key_value_heads,
            max_seq_len=max_seq_len,
            head_dim=self.config.head_dim,
            dtype=next(self.parameters()).dtype,
            device=str(input_ids.device),
        )

        generated = input_ids.clone()
        curr_pos = 0
        curr_input = input_ids
        step = 0
        prefill_logits = None

        while step < max_new_tokens:
            logits = self.forward(
                curr_input,
                kv_cache=kv_cache,
                start_pos=curr_pos,
            )
            next_token_logits = logits[:, -1, :]
            if step == 0:
                prefill_logits = next_token_logits.clone()

            # Greedy argmax decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            curr_pos += curr_input.shape[1]
            curr_input = next_token
            step += 1

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return generated, prefill_logits
