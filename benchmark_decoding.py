import argparse
import time
import torch
from transformers import AutoTokenizer

from engine import OptimizedPagedEngine
from model import CustomLlamaForCausalLM
from paged_attention import PagedLlamaForCausalLM


def benchmark_decoding(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    max_new_tokens: int = 128,
    num_runs: int = 3,
) -> None:
    assert torch.cuda.is_available(), "CUDA is required for benchmarking."
    torch.set_default_dtype(torch.float16)
    device = "cuda"

    print("=== Phase 4 & 5 Decoding Benchmark (GTX 1650 / sm_75) ===")
    print(f"Model: {model_id} | Decode Length: {max_new_tokens} tokens | Runs: {num_runs}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = CustomLlamaForCausalLM.from_safetensors(
        model_id, dtype=torch.float16, device=device
    )
    paged_model = PagedLlamaForCausalLM(base_model)

    prompt = (
        "Explain the architecture of modern GPU memory hierarchies, including global GDDR VRAM, "
        "L2 cache, shared memory (SRAM), and registers:"
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_list = input_ids[0].tolist()

    # Warmup Phase 1 Naive loop
    _ = base_model.generate_naive(input_ids, max_new_tokens=8, max_seq_len=512)
    torch.cuda.synchronize()

    # 1. Benchmark Phase 1 Naive PyTorch Loop (Contiguous KV Cache allocated to max_seq_len=2048)
    naive_times = []
    naive_tokens = None
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, _ = base_model.generate_naive(
            input_ids, max_new_tokens=max_new_tokens, max_seq_len=2048
        )
        torch.cuda.synchronize()
        naive_times.append(time.perf_counter() - t0)
        naive_tokens = out[0, input_ids.shape[1] :].tolist()

    avg_naive_s = sum(naive_times) / len(naive_times)
    naive_tps = max_new_tokens / avg_naive_s

    # 2. Benchmark Phase 4 Triton PagedAttention (Eager Decode, No CUDA Graph)
    paged_eager_engine = OptimizedPagedEngine(
        model=paged_model,
        num_blocks=128,
        block_size=16,
        enable_prefix_caching=False,
        enable_cuda_graph=False,
    )
    # Warmup Triton JIT compilation
    _ = paged_eager_engine.generate(
        seq_id=999, prompt_token_ids=prompt_list, max_new_tokens=8, verbose=False
    )
    torch.cuda.synchronize()

    paged_times = []
    paged_tokens = None
    for r in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        stats = paged_eager_engine.generate(
            seq_id=100 + r,
            prompt_token_ids=prompt_list,
            max_new_tokens=max_new_tokens,
            verbose=False,
        )
        torch.cuda.synchronize()
        paged_times.append(time.perf_counter() - t0)
        paged_tokens = stats.generated_ids

    avg_paged_s = sum(paged_times) / len(paged_times)
    paged_tps = max_new_tokens / avg_paged_s

    del paged_eager_engine
    torch.cuda.empty_cache()

    # 3. Benchmark Phase 5 Triton PagedAttention + CUDA Graph Decode
    paged_graph_engine = OptimizedPagedEngine(
        model=paged_model,
        num_blocks=128,
        block_size=16,
        enable_prefix_caching=True,
        enable_cuda_graph=True,
        max_batch_size=1,
    )
    # Warmup graph replay
    _ = paged_graph_engine.generate(
        seq_id=998, prompt_token_ids=prompt_list, max_new_tokens=8, verbose=False
    )
    torch.cuda.synchronize()
    graph_times = []
    graph_tokens = None
    for r in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        stats = paged_graph_engine.generate(
            seq_id=200 + r,
            prompt_token_ids=prompt_list,
            max_new_tokens=max_new_tokens,
            verbose=False,
        )
        torch.cuda.synchronize()
        graph_times.append(time.perf_counter() - t0)
        graph_tokens = stats.generated_ids

    avg_graph_s = sum(graph_times) / len(graph_times)
    graph_tps = max_new_tokens / avg_graph_s

    print(f"{'Engine Mode':<48} | {'Throughput':>14} | {'Latency/Tok':>13} | {'Exact Match'}")
    print("-" * 92)
    print(
        f"{'Phase 1: Naive PyTorch (Contiguous KV)':<48} | "
        f"{naive_tps:>10.2f} tok/s | {1000.0 / naive_tps:>9.2f} ms | Baseline"
    )
    print(
        f"{'Phase 4: Triton PagedAttention (Eager)':<48} | "
        f"{paged_tps:>10.2f} tok/s | {1000.0 / paged_tps:>9.2f} ms | {paged_tokens == naive_tokens}"
    )
    print(
        f"{'Phase 5: Triton PagedAttention + CUDA Graph + APC':<48} | "
        f"{graph_tps:>10.2f} tok/s | {1000.0 / graph_tps:>9.2f} ms | {graph_tokens == naive_tokens}"
    )

    assert paged_tokens == naive_tokens, "Phase 4 tokens diverged from Phase 1 baseline!"
    assert graph_tokens == naive_tokens, "Phase 5 tokens diverged from Phase 1 baseline!"
    print("\n[SUCCESS] Benchmark completed with 100% token accuracy across all engines!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    benchmark_decoding(
        model_id=args.model, max_new_tokens=args.max_new_tokens, num_runs=args.runs
    )
