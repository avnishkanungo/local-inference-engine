import argparse
import torch
from transformers import AutoTokenizer

from block_manager import BlockSpaceManager
from engine import OptimizedPagedEngine
from model import CustomLlamaForCausalLM, repeat_kv
from paged_attention import (
    PagedLlamaForCausalLM,
    paged_attention_decode_triton,
    reshape_and_cache_triton,
)


def test_triton_paged_attention_kernel_unit() -> None:
    """
    Unit-tests `_reshape_and_cache_kernel` and `_paged_attention_decode_kernel`
    directly against a pure PyTorch GQA reference across non-contiguous physical blocks.
    """
    print("=== Phase 4 Test 1: Triton PagedAttention Kernel Unit Test (sm_75) ===")
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    num_blocks = 16
    block_size = 16
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    num_kv_groups = num_heads // num_kv_heads
    sm_scale = head_dim ** -0.5

    k_cache = torch.zeros((num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype, device=device)
    v_cache = torch.zeros((num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype, device=device)

    # Simulate 2 sequences with non-contiguous page tables:
    #   Seq 0: 37 tokens across physical blocks [5, 2, 9]
    #   Seq 1: 51 tokens across physical blocks [11, 0, 7, 3]
    seq_blocks = [[5, 2, 9], [11, 0, 7, 3]]
    seq_lens = [37, 51]

    ref_k_list = []
    ref_v_list = []
    for b_idx, (blocks, slen) in enumerate(zip(seq_blocks, seq_lens)):
        k_seq = torch.randn((slen, num_kv_heads, head_dim), dtype=dtype, device=device)
        v_seq = torch.randn((slen, num_kv_heads, head_dim), dtype=dtype, device=device)
        ref_k_list.append(k_seq)
        ref_v_list.append(v_seq)

        slots = []
        for tok_idx in range(slen):
            phys_block = blocks[tok_idx // block_size]
            offset = tok_idx % block_size
            slots.append(phys_block * block_size + offset)
        slot_tensor = torch.tensor(slots, dtype=torch.int32, device=device)

        reshape_and_cache_triton(k_seq, v_seq, k_cache, v_cache, slot_tensor)

    # Decode query for batch_size = 2
    q = torch.randn((2, num_heads, head_dim), dtype=dtype, device=device)
    block_tables = torch.tensor(
        [[5, 2, 9, -1], [11, 0, 7, 3]], dtype=torch.int32, device=device
    )
    context_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    triton_out = paged_attention_decode_triton(
        q, k_cache, v_cache, block_tables, context_lens, sm_scale=sm_scale
    )

    # Pure PyTorch reference attention per sequence
    ref_outs = []
    for b_idx in range(2):
        q_b = q[b_idx : b_idx + 1].unsqueeze(2)  # [1, num_heads, 1, head_dim]
        k_b = ref_k_list[b_idx].transpose(0, 1).unsqueeze(0)  # [1, num_kv_heads, slen, head_dim]
        v_b = ref_v_list[b_idx].transpose(0, 1).unsqueeze(0)
        k_exp = repeat_kv(k_b, num_kv_groups)
        v_exp = repeat_kv(v_b, num_kv_groups)

        scores = torch.matmul(q_b, k_exp.transpose(2, 3)) * sm_scale
        probs = torch.softmax(scores.float(), dim=-1).to(dtype)
        out_b = torch.matmul(probs, v_exp).squeeze(2)  # [1, num_heads, head_dim]
        ref_outs.append(out_b)

    ref_out = torch.cat(ref_outs, dim=0)
    max_diff = (triton_out - ref_out).abs().max().item()
    print(f"Max absolute difference (Triton PagedAttention vs PyTorch Reference): {max_diff:.6f}")
    assert torch.allclose(triton_out, ref_out, atol=1e-3, rtol=1e-3), (
        f"Triton PagedAttention kernel mismatch! Max diff: {max_diff}"
    )
    print("[SUCCESS] Triton `_reshape_and_cache_kernel` & `_paged_attention_decode_kernel` match PyTorch!\n")


def test_end_to_end_paged_generation(
    model_id: str = "Qwen/Qwen2.5-0.5B", max_new_tokens: int = 30
) -> None:
    """
    Verifies that end-to-end generation using `PagedLlamaForCausalLM` + Triton PagedAttention
    produces the exact same token sequence as Phase 1's `CustomLlamaForCausalLM.generate_naive()`.
    """
    print("=== Phase 4 Test 2: End-to-End Triton PagedAttention Generation Parity ===")
    torch.set_default_dtype(torch.float16)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = CustomLlamaForCausalLM.from_safetensors(
        model_id, dtype=torch.float16, device=device
    )

    prompt = "In operating systems, virtual memory paging works by"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 1. Phase 1 Naive Contiguous KV baseline
    naive_out, _ = base_model.generate_naive(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        max_seq_len=256,
        eos_token_id=tokenizer.eos_token_id,
    )
    naive_generated_ids = naive_out[0, input_ids.shape[1] :].tolist()

    # 2. Phase 4 Triton PagedAttention Engine (shares weights with base_model)
    paged_model = PagedLlamaForCausalLM(base_model)
    engine = OptimizedPagedEngine(
        model=paged_model,
        num_blocks=64,
        block_size=16,
        enable_prefix_caching=False,
        enable_cuda_graph=False,
    )
    stats = engine.generate(
        seq_id=1,
        prompt_token_ids=input_ids[0].tolist(),
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        verbose=True,
    )

    print(f"\nPrompt:        {prompt!r}")
    print(f"Naive Output:  {tokenizer.decode(naive_generated_ids, skip_special_tokens=True)!r}")
    print(f"Paged Output:  {tokenizer.decode(stats.generated_ids, skip_special_tokens=True)!r}")
    print(f"Physical Blocks Used: {stats.assigned_blocks}")

    assert stats.generated_ids == naive_generated_ids, (
        f"Token mismatch between Phase 1 Naive and Phase 4 Triton PagedAttention!\n"
        f"Naive: {naive_generated_ids}\nPaged: {stats.generated_ids}"
    )
    print("\n[SUCCESS] Phase 4 Triton PagedAttention matches Phase 1 Naive generation 100%!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()
    test_triton_paged_attention_kernel_unit()
    test_end_to_end_paged_generation(model_id=args.model, max_new_tokens=args.max_new_tokens)
