# Antigravity Prompt: Custom LLM Inference Engine Build Plan

**Objective:** Build a custom LLM inference engine from scratch replicating vLLM's core optimizations (PagedAttention, Continuous Batching) optimized for a GTX 1650 (4GB VRAM, compute capability `sm_75`, no hardware bfloat16).

**Context Repositories:** Use the design philosophies of `nano-vllm` (a 1,200-line educational Python/Triton engine) and `MinivLLM` (which implements self-contained Triton kernels for PagedAttention).

*Instruction for Agent:* We will build this in strictly isolated phases. **Do not proceed to the next phase until I have tested and approved the current one.**

---

### Step 0: Environment & Hardware Verification

Before writing engine code, write a setup script to verify the local environment meets the exact requirements of `MinivLLM` and `nano-vllm`.

*   **Requirements:** Python 3.11, PyTorch 2.2+, and Triton.
*   **Task:** Create a `requirements.txt` containing `torch`, `transformers`, `triton`, and `xxhash` (required for prefix caching). 
*   **Verification Script:** Write a short `verify_env.py` script that checks if `torch.cuda.is_available()`, asserts the GPU compute capability is exactly 7.5, and forces PyTorch to use `float16` instead of `bfloat16`. 
*   **Approval Checkpoint:** *Wait for my confirmation that the environment is set up successfully.*

---

### Phase 1: The Model Scaffold & Naive Inference

Build the underlying model architecture without any vLLM optimizations to establish a correctness baseline.

*   **Task:** Implement a custom LLaMA-style model class in pure PyTorch (RMSNorm, RoPE, SwiGLU MLP). Load the `TinyLlama-1.1B` or `Qwen2.5-0.5B` safetensors directly into this custom class.
*   **The Loop:** Write a naive, manual `while` loop for generation that uses a standard, contiguous KV Cache (allocating the max sequence length upfront).
*   **Testing Phase 1:** Write a test script that runs a prompt through HuggingFace's standard `model.generate()` and our custom naive loop. Assert that the generated tokens and logits match exactly.
*   **Approval Checkpoint:** *Wait for my confirmation that the baseline tests pass.*

---

### Phase 2: The Block Manager (Virtual Memory for KV Cache)

Replicate `nano-vllm`'s memory management. We will build the page table in Python first before worrying about custom kernels.

*   **Task:** Implement a `BlockSpaceManager` class. It must divide the available VRAM into fixed-size blocks (e.g., 16 tokens per block).
*   **Page Tables:** Create the mapping logic: `Logical_Token_Index -> Physical_Block_Index`.
*   **Testing Phase 2:** Write a unit test that initializes the `BlockSpaceManager` with 100 available blocks. Simulate adding 3 requests of varying lengths, and print the page table to the console to verify that non-contiguous blocks are assigned correctly and freed when a simulated request ends.
*   **Approval Checkpoint:** *Wait for my confirmation that the block manager allocates and frees memory correctly.*

---

### Phase 3: Continuous Batching & The Scheduler

Shift the engine from batch-level execution to iteration-level execution.

*   **Task:** Implement a two-stage scheduler (`Prefill` and `Decode`). 
*   **Logic:** The scheduler must pull requests from a waiting queue into a running queue. At every generation step, if a sequence hits an EOS token, instantly evict it and allocate blocks for a new request from the waiting queue.
*   **Testing Phase 3:** Write a script passing three prompts of drastically different lengths (e.g., a 10-token output, a 50-token output, and a 150-token output). The logs must explicitly show the short prompt finishing, its memory blocks being freed, and a new prompt entering the batch while the long prompt is still generating.
*   **Approval Checkpoint:** *Wait for my confirmation that the scheduler logs show active continuous batching.*

---

### Phase 4: Custom Triton Kernels (PagedAttention)

Replace the standard PyTorch attention with Triton kernels, following the `MinivLLM` approach.

*   **Task:** Write a custom Triton kernel for PagedAttention. Since the target GPU is a GTX 1650 (`sm_75`), ensure the kernel respects tighter shared memory constraints and avoids instructions restricted to Ampere (`sm_80`). 
*   **Integration:** Replace the `torch.nn.MultiheadAttention` in Phase 1 with the Triton PagedAttention kernel, reading directly from the `BlockSpaceManager`'s page table.
*   **Testing Phase 4:** Provide `benchmark_decoding.py` (inspired by `MinivLLM`) to compare the token generation speed of the naive PyTorch loop from Phase 1 against the new Triton PagedAttention kernel.
*   **Approval Checkpoint:** *Wait for my confirmation that the Triton kernels compile and generate correct outputs.*

---

### Phase 5: Advanced Optimizations (CUDA Graphs & Prefix Caching)

Implement the final optimizations to reduce Python overhead and maximize memory efficiency.

*   **Task 1:** Implement prefix caching using `xxhash` to hash the token content of each block. If a new request shares a system prompt, point its page table to the existing physical blocks rather than recomputing them.
*   **Task 2:** Implement CUDA Graph capture for the decode step to eliminate the Python scheduling overhead during token generation.
*   **Testing Phase 5:** Send two prompts with identical, massive system prompts. The logs must prove that the second prompt skipped the prefill computation for the system prompt and reused the cached KV blocks.