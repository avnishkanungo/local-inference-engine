import sys
import torch
import triton
import transformers
import xxhash


def verify_environment() -> None:
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Triton version: {triton.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"xxhash version: {xxhash.VERSION}")

    # 1. Check CUDA availability
    assert torch.cuda.is_available(), "CUDA is not available! Check your PyTorch/CUDA installation."
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Detected GPU: {device_name} ({total_vram_gb:.2f} GB VRAM)")

    # 2. Assert GPU compute capability is exactly 7.5 (Turing / sm_75, e.g., GTX 1650)
    major, minor = torch.cuda.get_device_capability(0)
    compute_capability = f"{major}.{minor}"
    print(f"GPU Compute Capability: {compute_capability} (sm_{major}{minor})")
    assert (major, minor) == (7, 5), (
        f"Expected compute capability 7.5 (sm_75), but got {compute_capability}!"
    )

    # 3. Force PyTorch to use float16 instead of bfloat16 (no hardware bfloat16 on sm_75)
    torch.set_default_dtype(torch.float16)
    assert torch.get_default_dtype() == torch.float16, "Failed to set default dtype to torch.float16"

    # Quick sanity check tensor on GPU with float16
    test_tensor = torch.ones((4, 4), device="cuda")
    assert test_tensor.dtype == torch.float16, f"Expected float16 tensor, got {test_tensor.dtype}"
    print(f"Default PyTorch dtype forced to: {torch.get_default_dtype()}")
    print("\n[SUCCESS] Step 0 Environment & Hardware Verification passed!")


if __name__ == "__main__":
    verify_environment()
