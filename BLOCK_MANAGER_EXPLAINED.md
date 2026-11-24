# Deep Dive: Block Manager Architecture & Step-by-Step Flow

This document provides a comprehensive explanation of the **Block Manager** ([`block_manager.py`](block_manager.py)), why it is essential for Large Language Model (LLM) serving, how its end-to-end execution flow works, and which specific challenges it solves in **Phase 2** versus upcoming phases.

---

## 1. Why a Block Manager is Used: The Problem with Naive KV Caching

In autoregressive LLM inference, each newly generated token must attend to the Key ($K$) and Value ($V$) activation vectors of all preceding tokens. Storing these vectors across all layers is known as the **KV Cache**.

### The Phase 1 Baseline: `ContiguousKVCache`
In standard PyTorch implementations (such as Phase 1's [`ContiguousKVCache`](model.py)), tensors must be contiguous in memory. Consequently, memory for each sequence must be pre-allocated upfront:

```python
# Contiguous allocation upfront in VRAM:
k_cache = torch.zeros((num_layers, batch_size, num_kv_heads, max_seq_len, head_dim))
v_cache = torch.zeros((num_layers, batch_size, num_kv_heads, max_seq_len, head_dim))
```

This naive approach causes severe limitations, especially on hardware with constrained VRAM (like a 4 GB GTX 1650):

### 1. Large Space Over-Allocation & Internal Fragmentation
* Because the inference engine cannot predict whether a user prompt will generate 10 tokens or 2,048 tokens, it is forced to allocate for the worst-case maximum sequence length (`max_seq_len = 2048`).
* If a request terminates after generating only 30 tokens, **over 98% of its reserved memory remains completely unused** yet cannot be utilized by any other request.
* In Phase 1 tests, setting `batch_size = 8` and `max_seq_len = 2048` consumed **~192 MB of VRAM before a single token was generated**.

### 2. External Memory Fragmentation & Inflexible Batching
* In production serving, requests arrive dynamically with different prompt lengths and generate outputs at varying speeds.
* With contiguous tensors, when a short request completes early, its memory cannot simply be given to another growing sequence because sequences must remain strictly contiguous. Reclaiming or rearranging this memory requires expensive GPU memory copies and re-allocations.

### 3. Redundant Memory for Shared Prefixes
* If 10 separate requests share the exact same system prompt or few-shot examples (e.g., 500 tokens), a contiguous cache is forced to duplicate those 500 KV vectors 10 separate times in VRAM.

---

## 2. What Phase 2 Implements vs. Upcoming Phases

The implementation in [`block_manager.py`](block_manager.py) solves the memory management and fragmentation problems at the **logical control layer**:

| Dimension | Phase 2 (Current Scope) | Upcoming Phases (Phases 3–5) |
| :--- | :--- | :--- |
| **Primary Goal** | **Eliminates Internal & External Fragmentation logic** via virtual memory paging. Reduces unused KV cache memory to **< 4%** (at most 15 unused token slots in the last 16-token block). | **Continuous Batching & GPU Kernels**: Runs live multi-request scheduling and non-contiguous GPU attention. |
| **Multi-Request Handling** | **Logical Simulation**: Tracks multiple concurrent sequence page tables (`seq_id=101, 102, 103`), manages free/used block pools, and performs $O(1)$ allocation/deallocation. | **Production Scheduling (Phase 3)**: Request queuing, dynamic preemption, iteration-level continuous batching. |
| **Attention Computation** | None (Phase 1 still runs standard contiguous PyTorch attention). | **PagedAttention Kernel (Phase 4)**: Custom Triton/CUDA kernels reading the 2D block table tensor directly on the GPU. |
| **Prefix Caching** | Tracks `ref_count` and `block_hash` per physical block. | **Radix / Prefix Sharing (Phase 5)**: Reusing shared physical blocks across requests with identical prompt prefixes. |

---

## 3. Architecture: CPU Control Plane vs. GPU Data Plane

A common misconception is that the Block Manager runs inside CUDA. In reality, modern engines (vLLM, SGLang, TensorRT-LLM) split the responsibilities:

* **CPU Control Plane (Block Manager in Python/C++):**
  - Handles discrete, branch-heavy bookkeeping: free block queues (`deque`), dictionary page tables (`dict[seq_id, list[int]]`), reference counts, and hash lookups.
  - CPUs excel at pointer chasing, dictionary manipulation, and serial decision-making without stalling GPU compute pipelines.
* **GPU Data Plane (VRAM Buffer & CUDA/Triton Kernels):**
  - Allocates the large physical KV cache tensor once at engine initialization.
  - Executes the massively parallel mathematical operations (**PagedAttention** in Phase 4), reading the blocks indicated by the CPU's block table.

---

## 4. End-to-End Step-by-Step Flow

The complete lifecycle from engine startup to token generation proceeds through five distinct stages:

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. SYSTEM INITIALIZATION (Executed Once at Boot)                       │
 │                                                                        │
 │  GPU (VRAM): Allocates static physical KV buffer:                      │
 │    k_cache = [num_layers, num_blocks, block_size, num_heads, head_dim] │
 │                                                                        │
 │  CPU (BlockSpaceManager):                                              │
 │    - Initializes free pool: [P0, P1, P2, ..., P99]                     │
 │    - Initializes empty page tables: {}                                 │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 2. PROMPT ARRIVAL & PREFILL ALLOCATION                                 │
 │                                                                        │
 │  - Input prompt is tokenized into token IDs (e.g., 33 tokens).         │
 │  - CPU Block Manager computes required blocks:                         │
 │      ceil(33 / 16) = 3 blocks                                          │
 │  - Pops 3 blocks from free pool -> [P0, P1, P9]                        │
 │  - Page Table records: { seq_101: [0, 1, 9] }                          │
 │  - CPU sends to GPU:                                                   │
 │      • input_ids: [33 token IDs]                                       │
 │      • block_table tensor: [[0, 1, 9]]                                 │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 3. PREFILL MODEL FORWARD PASS (GPU)                                    │
 │                                                                        │
 │  - Model computes Query (Q), Key (K), and Value (V) projections.       │
 │  - GPU writes computed K & V vectors into assigned physical slots:     │
 │      • Tokens 0..15  -> Physical Block P0 (offset 0..15)               │
 │      • Tokens 16..31 -> Physical Block P1 (offset 0..15)               │
 │      • Token 32      -> Physical Block P9 (offset 0)                   │
 │  - PagedAttention computes self-attention and emits Token #34.         │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 4. AUTOREGRESSIVE DECODE LOOP (1 Token per Step)                       │
 │                                                                        │
 │  For each newly generated token, CPU calls `append_slot(seq_101)`:     │
 │                                                                        │
 │  Case A (Current block has space: len % 16 != 0):                      │
 │    - Token #35: written to next offset in P9 (offset 1).               │
 │    - 0 new blocks allocated!                                           │
 │                                                                        │
 │  Case B (Current block is full: len % 16 == 0):                        │
 │    - Token #49: P9 is full (16/16).                                    │
 │    - Pops a new block from free pool (e.g., P14).                      │
 │    - Page Table becomes: [0, 1, 9, 14] (Non-contiguous in VRAM).      │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 5. DEALLOCATION (Request Finished)                                     │
 │                                                                        │
 │  - Request hits EOS or reaches max_new_tokens.                         │
 │  - CPU calls `free(seq_101)`:                                          │
 │      • Removes seq_101 from page_tables dictionary.                    │
 │      • Decrements ref_count on blocks [0, 1, 9, 14].                   │
 │      • Blocks with ref_count == 0 are reset and pushed back to the     │
 │        free block pool for immediate reuse.                            │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Mathematical Address Resolution Formula

When the model needs to access the $K$ or $V$ vector for logical token index $i$ in sequence `seq_id`:

1. **Logical Block Index:**
   $$\text{logical\_block\_idx} = \lfloor i / \text{block\_size} \rfloor$$

2. **Block Offset:**
   $$\text{offset} = i \bmod \text{block\_size}$$

3. **Physical Block ID:**
   $$\text{phys\_block\_id} = \text{page\_tables}[\text{seq\_id}][\text{logical\_block\_idx}]$$

4. **Global Physical Slot in VRAM:**
   $$\text{global\_slot} = \text{phys\_block\_id} \times \text{block\_size} + \text{offset}$$

Because this translation is purely $O(1)$ arithmetic, it introduces zero algorithmic overhead during execution.

---

## 6. What Was Verified in `test_phase2.py`

The unit tests in [`test_phase2.py`](test_phase2.py) explicitly validate these principles:

1. **Prompt Allocation:** Allocated 3 requests of varying prompt lengths (20, 50, and 35 tokens), confirming exact block counts (2, 4, and 3 blocks respectively).
2. **Virtual Address Resolution:** Confirmed that arbitrary logical indices (e.g., token 19 of sequence 101) correctly resolve to their corresponding physical block and offset.
3. **Non-Contiguous Allocation:** Simulated token decoding across multiple sequences, proving that when sequence 101 overflows its initial blocks, it acquires block `P9`—resulting in a non-contiguous page table `[0, 1, 9]`.
4. **Immediate Deallocation:** Freed sequence 102 (freeing physical blocks `[2, 3, 4, 5]`), immediately returning them to the free pool while sequences 101 and 103 continued executing undisturbed.
5. **Padded Block Table Generation:** Exported the active page tables into a padded 2D integer tensor (`torch.int32`), ready to be passed directly to the Phase 4 Triton PagedAttention kernel.
