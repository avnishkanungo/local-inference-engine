import argparse
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import CustomLlamaForCausalLM


def test_phase1(model_id: str = "Qwen/Qwen2.5-0.5B", max_new_tokens: int = 25) -> None:
    assert torch.cuda.is_available(), "CUDA is required for Phase 1 test."
    torch.set_default_dtype(torch.float16)
    device = "cuda"

    prompt = "Provide the recursion based code for fibonacci numbers"
    print(f"=== Phase 1 Baseline Test ===")
    print(f"Model: {model_id}")
    print(f"Prompt: {prompt!r}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 1. Run HuggingFace Reference Model
    print("[1/2] Running HuggingFace AutoModelForCausalLM.generate()...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
    hf_model.eval()

    with torch.inference_mode():
        hf_prefill_logits = hf_model(input_ids).logits[:, -1, :].clone()
        hf_generated_ids = hf_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    hf_text = tokenizer.decode(hf_generated_ids[0], skip_special_tokens=True)
    print(f"HF Output:     {hf_text!r}")
    print(f"HF Token IDs:  {hf_generated_ids[0].tolist()}")

    # Free HF model from 4GB GTX 1650 VRAM before loading custom model
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Run Custom Safetensors Model & Naive While Loop
    print("\n[2/2] Loading CustomLlamaForCausalLM directly from .safetensors...")
    custom_model = CustomLlamaForCausalLM.from_safetensors(
        model_id,
        dtype=torch.float16,
        device=device,
    )

    custom_generated_ids, custom_prefill_logits = custom_model.generate_naive(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        max_seq_len=512,
        eos_token_id=tokenizer.eos_token_id,
    )

    custom_text = tokenizer.decode(custom_generated_ids[0], skip_special_tokens=True)
    print(f"Custom Output: {custom_text!r}")
    print(f"Custom IDs:    {custom_generated_ids[0].tolist()}")

    # 3. Assertions
    max_logit_diff = (hf_prefill_logits - custom_prefill_logits).abs().max().item()
    print(f"\nMax Prefill Logit Absolute Difference: {max_logit_diff:.6f}")

    assert torch.allclose(hf_prefill_logits, custom_prefill_logits, atol=5e-2, rtol=1e-2), (
        f"Logits diverged! Max diff: {max_logit_diff}"
    )
    assert torch.equal(hf_generated_ids, custom_generated_ids), (
        f"Generated tokens do not match!\nHF:     {hf_generated_ids.tolist()}\nCustom: {custom_generated_ids.tolist()}"
    )

    print("\n[SUCCESS] Baseline test passed! Custom model logits and generated tokens match HuggingFace.")


def test_rmsnorm_component() -> None:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from model import RMSNorm

    print("\n=== Test 2: Individual Component Verification (RMSNorm) ===")
    device = "cuda"
    x = torch.randn(2, 16, 896, device=device, dtype=torch.float16)
    hf_norm = Qwen2RMSNorm(896, eps=1e-6).to(device, dtype=torch.float16)
    our_norm = RMSNorm(896, eps=1e-6).to(device, dtype=torch.float16)

    diff = (hf_norm(x) - our_norm(x)).abs().max().item()
    print(f"RMSNorm max absolute diff vs HuggingFace: {diff}")
    assert diff == 0.0, f"Expected 0.0 diff for RMSNorm, got {diff}"
    print("[SUCCESS] RMSNorm is bitwise identical to HuggingFace!")


def test_multiple_prompts(model_id: str = "Qwen/Qwen2.5-0.5B") -> None:
    print("\n=== Test 3: Multiple Prompts & Step-by-Step Token Verification ===")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompts = [
        "In computer science, virtual memory is",
        "def fibonacci(n):\n    if n <= 1:\n        return",
        "The planets of the solar system in order from the Sun are",
    ]

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, attn_implementation="eager"
    ).to(device).eval()
    hf_outputs = {}
    with torch.inference_mode():
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
            out = hf_model.generate(
                ids, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
            hf_outputs[p] = out

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    custom_model = CustomLlamaForCausalLM.from_safetensors(
        model_id, dtype=torch.float16, device=device
    )
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        out, _ = custom_model.generate_naive(
            ids, max_new_tokens=20, max_seq_len=256, eos_token_id=tokenizer.eos_token_id
        )
        match = torch.equal(out, hf_outputs[p])
        print(f"\nPrompt: {p!r}")
        print(f"Exact Token Match: {match}")
        print(f"Generated: {tokenizer.decode(out[0], skip_special_tokens=True)!r}")
        assert match, "Mismatch detected!"

    del custom_model
    gc.collect()
    torch.cuda.empty_cache()
    print("\n[SUCCESS] All custom prompts matched HuggingFace token-for-token!")


def test_architecture_and_vram(model_id: str = "Qwen/Qwen2.5-0.5B") -> None:
    from model import ContiguousKVCache

    print("\n=== Test 4: Architecture & Contiguous KV Cache VRAM Footprint ===")
    device = "cuda"
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_before = torch.cuda.memory_allocated()
    model = CustomLlamaForCausalLM.from_safetensors(
        model_id, dtype=torch.float16, device=device
    )
    model_vram_mb = (torch.cuda.memory_allocated() - mem_before) / (1024 ** 2)
    print(f"Model Architecture:\n{model}\n")
    print(f"Model Config: {model.config}")
    print(f"Model Weights VRAM: {model_vram_mb:.2f} MB")

    cfg = model.config
    mem_before_kv = torch.cuda.memory_allocated()
    kv_cache = ContiguousKVCache(
        num_layers=cfg.num_hidden_layers,
        batch_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        max_seq_len=2048,
        head_dim=cfg.head_dim,
        dtype=torch.float16,
        device=device,
    )
    kv_vram_mb = (torch.cuda.memory_allocated() - mem_before_kv) / (1024 ** 2)
    print(
        f"Naive Contiguous KV Cache (batch=8, max_seq_len=2048) upfront VRAM: "
        f"{kv_vram_mb:.2f} MB (even for 0 tokens stored!)"
    )
    del kv_cache, model
    gc.collect()
    torch.cuda.empty_cache()
    print("\n[SUCCESS] All 4 Phase 1 test suites passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HF model ID (e.g. Qwen/Qwen2.5-0.5B or TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=25)
    args = parser.parse_args()
    test_phase1(model_id=args.model, max_new_tokens=args.max_new_tokens)
    test_rmsnorm_component()
    test_multiple_prompts(model_id=args.model)
    test_architecture_and_vram(model_id=args.model)
