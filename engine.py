from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import time

import torch

from block_manager import BlockSpaceManager
from paged_attention import PagedKVCachePool, PagedLlamaForCausalLM


class CUDAGraphDecodeRunner:
    """
    Phase 5 (Task 2): CUDA Graph Capture & Replay Runner for the Decode Step.

    Eliminates Python and CPU kernel-launch overhead across the 24 Transformer layers
    by pre-allocating static GPU tensor buffers (`input_ids`, `position_ids`,
    `slot_mapping`, `block_tables`, `context_lens`) and capturing the entire Triton
    PagedAttention decode forward pass into a single `torch.cuda.CUDAGraph`.
    """

    def __init__(
        self,
        model: PagedLlamaForCausalLM,
        kv_pool: PagedKVCachePool,
        max_batch_size: int = 4,
        max_blocks_per_seq: int = 64,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.kv_pool = kv_pool
        self.max_batch_size = max_batch_size
        self.max_blocks_per_seq = max_blocks_per_seq
        self.device = device

        # Pre-allocate static GPU input buffers whose memory addresses never change
        self.static_input_ids = torch.zeros(
            (max_batch_size, 1), dtype=torch.long, device=device
        )
        self.static_position_ids = torch.zeros(
            (max_batch_size, 1), dtype=torch.long, device=device
        )
        self.static_slot_mapping = torch.full(
            (max_batch_size,), -1, dtype=torch.int32, device=device
        )
        self.static_context_lens = torch.zeros(
            (max_batch_size,), dtype=torch.int32, device=device
        )
        self.static_block_tables = torch.zeros(
            (max_batch_size, max_blocks_per_seq), dtype=torch.int32, device=device
        )

        self.static_logits: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.is_captured: bool = False

    @torch.inference_mode()
    def capture(self) -> None:
        """Warms up Triton JIT compilation and records the decode forward pass into a CUDAGraph."""
        # Warmup pass (compiles Triton PTX kernels before graph capture starts)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _ = self.model.forward_decode(
                    self.static_input_ids,
                    self.static_position_ids,
                    kv_pool=self.kv_pool,
                    slot_mapping=self.static_slot_mapping,
                    block_tables=self.static_block_tables,
                    context_lens=self.static_context_lens,
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        # Capture the static decode graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self.model.forward_decode(
                self.static_input_ids,
                self.static_position_ids,
                kv_pool=self.kv_pool,
                slot_mapping=self.static_slot_mapping,
                block_tables=self.static_block_tables,
                context_lens=self.static_context_lens,
            )
        torch.cuda.synchronize()
        self.is_captured = True

    @torch.inference_mode()
    def replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Copies dynamic step metadata into static buffers and replays the captured CUDA Graph.
        Supports any active batch_size <= max_batch_size (inactive rows are masked with
        slot_mapping = -1 and context_lens = 0).
        """
        if not self.is_captured or self.graph is None or self.static_logits is None:
            raise RuntimeError("CUDAGraphDecodeRunner.capture() must be called before replay()!")

        bsz = input_ids.shape[0]
        num_blocks_cols = block_tables.shape[1]
        assert bsz <= self.max_batch_size, f"Batch size {bsz} exceeds max_batch_size {self.max_batch_size}"
        assert num_blocks_cols <= self.max_blocks_per_seq

        # Reset inactive rows to safe no-op values (slot=-1, context_len=0)
        self.static_slot_mapping.fill_(-1)
        self.static_context_lens.zero_()

        # Copy current step's values in-place into the static GPU addresses
        self.static_input_ids[:bsz].copy_(input_ids)
        self.static_position_ids[:bsz].copy_(position_ids)
        self.static_slot_mapping[:bsz].copy_(slot_mapping)
        self.static_context_lens[:bsz].copy_(context_lens)
        self.static_block_tables[:bsz, :num_blocks_cols].copy_(block_tables)

        self.graph.replay()
        return self.static_logits[:bsz]


@dataclass
class GenerationStats:
    seq_id: int
    generated_ids: List[int]
    prompt_tokens: int
    cached_prefix_tokens: int
    prefill_tokens_computed: int
    assigned_blocks: List[int]
    prefill_time_ms: float
    decode_time_ms: float
    used_cuda_graph: bool


class OptimizedPagedEngine:
    """
    Phase 4 & 5 Complete Engine integrating:
      - `PagedLlamaForCausalLM` & Triton PagedAttention kernels (`sm_75`)
      - `BlockSpaceManager` with `xxhash` Prefix Caching
      - `PagedKVCachePool` physical block tensors in GPU VRAM
      - `CUDAGraphDecodeRunner` for zero-overhead decode steps
    """

    def __init__(
        self,
        model: PagedLlamaForCausalLM,
        num_blocks: int = 128,
        block_size: int = 16,
        enable_prefix_caching: bool = True,
        enable_cuda_graph: bool = True,
        max_batch_size: int = 1,
    ) -> None:
        self.model = model
        self.config = model.config
        self.device = str(next(model.parameters()).device)
        self.dtype = next(model.parameters()).dtype
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_cuda_graph = enable_cuda_graph

        self.block_manager = BlockSpaceManager(
            num_blocks=num_blocks,
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.kv_pool = PagedKVCachePool(
            num_layers=self.config.num_hidden_layers,
            num_blocks=num_blocks,
            num_kv_heads=self.config.num_key_value_heads,
            block_size=block_size,
            head_dim=self.config.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        self.cuda_graph_runner: Optional[CUDAGraphDecodeRunner] = None
        if enable_cuda_graph:
            self.cuda_graph_runner = CUDAGraphDecodeRunner(
                model=self.model,
                kv_pool=self.kv_pool,
                max_batch_size=max_batch_size,
                max_blocks_per_seq=num_blocks,
                device=self.device,
            )
            self.cuda_graph_runner.capture()

    @torch.inference_mode()
    def generate(
        self,
        seq_id: int,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        free_on_finish: bool = True,
        verbose: bool = True,
    ) -> GenerationStats:
        """
        Generates up to `max_new_tokens` for `seq_id` using:
          1. Prefix-cache-aware block allocation (`xxhash`).
          2. Partial prefill on only the uncached suffix tokens.
          3. Triton PagedAttention decode (via CUDA Graph replay if enabled).
        """
        prompt_ids = list(prompt_token_ids)
        total_prompt_len = len(prompt_ids)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1. Allocate physical blocks (reusing cached xxhash prefix blocks if enabled)
        if self.enable_prefix_caching:
            assigned_blocks, cached_tokens = self.block_manager.allocate_with_prefix_cache(
                seq_id, prompt_ids
            )
        else:
            assigned_blocks = self.block_manager.allocate(seq_id, prompt_ids)
            cached_tokens = 0

        uncached_token_ids = prompt_ids[cached_tokens:]
        prefill_tokens_computed = len(uncached_token_ids)

        if verbose:
            print(
                f"[Engine | Seq {seq_id} PREFILL] Prompt={total_prompt_len} tokens | "
                f"Prefix Cache Hit={cached_tokens} tokens ({cached_tokens // self.block_size} blocks) | "
                f"Computing Prefill for only {prefill_tokens_computed} tokens | "
                f"Page Table={assigned_blocks}"
            )

        # Build physical slot mapping only for the uncached suffix tokens
        slot_list = []
        for logical_idx in range(cached_tokens, total_prompt_len):
            _, _, phys_slot = self.block_manager.resolve_logical_token(seq_id, logical_idx)
            slot_list.append(phys_slot)

        slot_mapping = torch.tensor(slot_list, dtype=torch.int32, device=self.device)
        input_ids = torch.tensor([uncached_token_ids], dtype=torch.long, device=self.device)
        position_ids = torch.arange(
            cached_tokens, total_prompt_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # 2. Run Prefill forward pass on uncached suffix tokens
        logits = self.model.forward_prefill(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_pool=self.kv_pool,
            slot_mapping=slot_mapping,
            physical_blocks=assigned_blocks,
            total_seq_len=total_prompt_len,
        )
        next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated: List[int] = [next_token]

        torch.cuda.synchronize()
        t_prefill_done = time.perf_counter()
        prefill_ms = (t_prefill_done - t0) * 1000.0

        # 3. Run Decode steps using Triton PagedAttention (+ CUDA Graph if enabled)
        step = 1
        while step < max_new_tokens:
            if eos_token_id is not None and next_token == eos_token_id:
                break

            curr_pos = self.block_manager.seq_num_tokens[seq_id]
            _, _, phys_slot = self.block_manager.append_slot(seq_id, next_token)
            curr_total_len = self.block_manager.seq_num_tokens[seq_id]

            # If appending a token completed a full 16-token block and prefix caching is on,
            # register its chained xxhash so even generated blocks or full prompt tail blocks can be cached
            dec_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            dec_pos = torch.tensor([[curr_pos]], dtype=torch.long, device=self.device)
            dec_slot = torch.tensor([phys_slot], dtype=torch.int32, device=self.device)
            dec_ctx_len = torch.tensor([curr_total_len], dtype=torch.int32, device=self.device)
            dec_block_table = self.block_manager.get_block_table_tensor([seq_id], device=self.device)

            if self.enable_cuda_graph and self.cuda_graph_runner is not None:
                dec_logits = self.cuda_graph_runner.replay(
                    input_ids=dec_input,
                    position_ids=dec_pos,
                    slot_mapping=dec_slot,
                    block_tables=dec_block_table,
                    context_lens=dec_ctx_len,
                )
            else:
                dec_logits = self.model.forward_decode(
                    input_ids=dec_input,
                    position_ids=dec_pos,
                    kv_pool=self.kv_pool,
                    slot_mapping=dec_slot,
                    block_tables=dec_block_table,
                    context_lens=dec_ctx_len,
                )

            next_token = int(torch.argmax(dec_logits[:, -1, :], dim=-1).item())
            generated.append(next_token)
            step += 1

        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_prefill_done) * 1000.0
        final_blocks = list(self.block_manager.page_tables[seq_id])

        if free_on_finish:
            self.block_manager.free(seq_id)

        return GenerationStats(
            seq_id=seq_id,
            generated_ids=generated,
            prompt_tokens=total_prompt_len,
            cached_prefix_tokens=cached_tokens,
            prefill_tokens_computed=prefill_tokens_computed,
            assigned_blocks=final_blocks,
            prefill_time_ms=prefill_ms,
            decode_time_ms=decode_ms,
            used_cuda_graph=self.enable_cuda_graph,
        )
