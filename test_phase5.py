import argparse
import torch
from transformers import AutoTokenizer

from engine import OptimizedPagedEngine
from paged_attention import PagedLlamaForCausalLM


MASSIVE_SYSTEM_PROMPT = (
    "You are an expert systems architect specializing in GPU kernel programming, "
    "operating system virtual memory, and high-throughput Large Language Model inference engines "
    "such as vLLM, SGLang, and TensorRT-LLM. Always obey these strict technical guidelines:\n"
    "1. Memory Hierarchy Awareness: Distinguish clearly between GPU High-Bandwidth / GDDR6 global memory, "
    "L2 cache, on-chip shared memory (SRAM), and warp register files.\n"
    "2. PagedAttention Mechanics: Explain how logical token indices map to non-contiguous physical "
    "16-token KV blocks via per-sequence page tables to eliminate internal and external fragmentation.\n"
    "3. Turing sm_75 Constraints: Always account for GTX 1650 hardware characteristics—64 KB max shared "
    "memory per SM, 4 GB total VRAM, and native FP16 Tensor Cores without hardware BFloat16 support.\n"
    "4. Prefix Caching & Determinism: Provide concise, mathematically accurate explanations.\n\n"
)


def test_phase5(model_id: str = "Qwen/Qwen2.5-0.5B", max_new_tokens: int = 30) -> None:
    assert torch.cuda.is_available(), "CUDA is required for Phase 5 tests."
    torch.set_default_dtype(torch.float16)
    device = "cuda"

    print("=== Phase 5 Test: Prefix Caching (xxhash) & CUDA Graph Decode Capture ===")
    print(f"Model: {model_id}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    paged_model = PagedLlamaForCausalLM.from_safetensors(
        model_id, dtype=torch.float16, device=device
    )

    # Initialize Phase 5 Engine with BOTH xxhash Prefix Caching and CUDA Graph Capture enabled
    print("[Step 1] Initializing OptimizedPagedEngine (Capturing Decode CUDA Graph)...")
    engine = OptimizedPagedEngine(
        model=paged_model,
        num_blocks=128,
        block_size=16,
        enable_prefix_caching=True,
        enable_cuda_graph=True,
        max_batch_size=4,
    )
    assert engine.cuda_graph_runner is not None and engine.cuda_graph_runner.is_captured
    print("  -> CUDA Graph successfully captured for 24-layer Triton PagedAttention decode!\n")

    prompt_1 = MASSIVE_SYSTEM_PROMPT + "User Question: What is the primary role of a page table?\nAnswer:"
    prompt_2 = MASSIVE_SYSTEM_PROMPT + "User Question: Why do we use float16 instead of bfloat16 on GTX 1650?\nAnswer:"

    sys_tokens = tokenizer.encode(MASSIVE_SYSTEM_PROMPT)
    p1_tokens = tokenizer.encode(prompt_1)
    p2_tokens = tokenizer.encode(prompt_2)
    expected_shared_blocks = len(sys_tokens) // 16
    expected_shared_tokens = expected_shared_blocks * 16

    print(
        f"[Step 2] Massive System Prompt Length: {len(sys_tokens)} tokens "
        f"({expected_shared_blocks} full 16-token blocks = {expected_shared_tokens} cacheable prefix tokens)"
    )
    print(f"  Prompt 1 Total Length: {len(p1_tokens)} tokens")
    print(f"  Prompt 2 Total Length: {len(p2_tokens)} tokens\n")

    # Run Prompt 1 (Cold Cache - computes full prefill and populates xxhash block cache)
    # Keep free_on_finish=False first so both sequences are simultaneously active and share physical blocks!
    print("[Step 3] Running Request 1 (Cold Prefix Cache - populates xxhash physical blocks):")
    stats_1 = engine.generate(
        seq_id=1,
        prompt_token_ids=p1_tokens,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        free_on_finish=False,
        verbose=True,
    )

    # Run Prompt 2 (Warm Cache - shares the massive system prompt!)
    print("\n[Step 4] Running Request 2 (Identical Massive System Prompt - Warm xxhash Prefix Cache):")
    stats_2 = engine.generate(
        seq_id=2,
        prompt_token_ids=p2_tokens,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        free_on_finish=False,
        verbose=True,
    )

    # Verify physical block sharing in the page tables
    shared_phys_blocks_req1 = stats_1.assigned_blocks[:expected_shared_blocks]
    shared_phys_blocks_req2 = stats_2.assigned_blocks[:expected_shared_blocks]
    ref_counts = [engine.block_manager.blocks[b].ref_count for b in shared_phys_blocks_req1]
    block_hashes = [hex(engine.block_manager.blocks[b].block_hash) for b in shared_phys_blocks_req1[:3]]

    print("\n--- Prefix Caching & Page Table Verification ---")
    print(f"Req 1 System Prompt Physical Blocks: {shared_phys_blocks_req1}")
    print(f"Req 2 System Prompt Physical Blocks: {shared_phys_blocks_req2}")
    print(f"Shared Physical Block Ref Counts:    {ref_counts} (ref_count=2 proves zero-copy sharing!)")
    print(f"Sample Chained xxhash64 Digests:     {block_hashes} ...")
    print(
        f"Req 1 Prefill Tokens Computed:       {stats_1.prefill_tokens_computed}/{stats_1.prompt_tokens} "
        f"({stats_1.prefill_time_ms:.2f} ms)"
    )
    print(
        f"Req 2 Prefill Tokens Computed:       {stats_2.prefill_tokens_computed}/{stats_2.prompt_tokens} "
        f"(Skipped {stats_2.cached_prefix_tokens} cached tokens! {stats_2.prefill_time_ms:.2f} ms)"
    )

    # Verify exact numerical equivalence between prefix-cached Request 2 and a fresh uncached run
    print("\n[Step 5] Verifying numerical equivalence of Prefix-Cached + CUDA-Graph output vs uncached baseline...")
    uncached_engine = OptimizedPagedEngine(
        model=paged_model,
        num_blocks=64,
        block_size=16,
        enable_prefix_caching=False,
        enable_cuda_graph=False,
    )
    baseline_stats_2 = uncached_engine.generate(
        seq_id=99,
        prompt_token_ids=p2_tokens,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        verbose=False,
    )

    text_1 = tokenizer.decode(stats_1.generated_ids, skip_special_tokens=True)
    text_2 = tokenizer.decode(stats_2.generated_ids, skip_special_tokens=True)
    print(f"\nReq 1 Generated Answer: {text_1!r}")
    print(f"Req 2 Generated Answer: {text_2!r}")

    assert shared_phys_blocks_req1 == shared_phys_blocks_req2, (
        "Request 2 page table did not point to Request 1's cached physical blocks!"
    )
    assert all(rc == 2 for rc in ref_counts), (
        f"Expected ref_count == 2 on all shared blocks, got {ref_counts}"
    )
    assert stats_2.cached_prefix_tokens >= expected_shared_tokens, (
        f"Expected at least {expected_shared_tokens} cached prefix tokens, got {stats_2.cached_prefix_tokens}"
    )
    assert stats_2.prefill_tokens_computed == len(p2_tokens) - stats_2.cached_prefix_tokens
    assert stats_2.generated_ids == baseline_stats_2.generated_ids, (
        f"Prefix-cached + CUDA-Graph tokens diverged from uncached baseline!\n"
        f"Cached+Graph: {stats_2.generated_ids}\nBaseline:     {baseline_stats_2.generated_ids}"
    )

    # Clean up both sequences and confirm ref_counts decrement properly
    engine.block_manager.free(seq_id=1)
    assert all(engine.block_manager.blocks[b].ref_count == 1 for b in shared_phys_blocks_req1)
    engine.block_manager.free(seq_id=2)
    assert all(engine.block_manager.blocks[b].ref_count == 0 for b in shared_phys_blocks_req1)
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks

    print("\n[SUCCESS] Phase 5 (xxhash Prefix Caching & CUDA Graph Decode) tests passed 100%!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()
    test_phase5(model_id=args.model, max_new_tokens=args.max_new_tokens)
