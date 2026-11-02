# Local-vLLM: Custom LLM Inference Engine (`sm_75` / GTX 1650)

A custom LLM inference engine built from scratch in PyTorch and Triton, replicating the core architectural optimizations of **vLLM** (`nano-vllm` and `MinivLLM` design philosophies) and specifically tuned for an **NVIDIA GeForce GTX 1650** (4 GB VRAM, Turing `sm_75`, `float16`).

---

## Project Phases Overview

| Phase | Module(s) | Test Script | Status |
| :--- | :--- | :--- | :--- |
| **Step 0: Environment & Hardware Verification** | `verify_env.py` | `uv run verify_env.py` | Completed |
| **Phase 1: Model Scaffold & Naive Inference** | `model.py` | `uv run test_phase1.py` | Completed |
| **Phase 2: The Block Manager (Paged KV Virtual Memory)** | `block_manager.py` | `uv run test_phase2.py` | Completed |
| **Phase 3: Continuous Batching & Scheduler** | *Upcoming* | *Upcoming* | Pending |
| **Phase 4: Custom Triton PagedAttention Kernels** | *Upcoming* | *Upcoming* | Pending |
| **Phase 5: Prefix Caching & CUDA Graphs** | *Upcoming* | *Upcoming* | Pending |

---

## Phase 2: The Block Manager (Virtual Memory for KV Cache)

### 1. What Phase 2 Does
Phase 2 implements [`BlockSpaceManager`](block_manager.py) and [`PhysicalBlock`](block_manager.py), bringing **Operating System virtual memory paging** to the LLM Key-Value (KV) Cache.

Instead of giving each request one giant, contiguous tensor slice in VRAM upfront, the Block Manager divides the KV cache pool into fixed-size **physical blocks** (each holding `block_size = 16` tokens) and maintains a **Page Table** per active sequence that maps logical token positions to physical blocks anywhere in memory.

---

### 2. Why a Block Manager Is Used (The Problem with Naive KV Caches)

In **Phase 1** ([`ContiguousKVCache`](model.py)), generating text required pre-allocating a contiguous tensor of shape:
```text
[num_layers, batch_size, num_kv_heads, max_seq_len, head_dim]
```
On a 4 GB GPU like the GTX 1650, this naive approach creates three severe bottlenecks:

1. **Internal Fragmentation (Reserved but Unused Memory):**
   Because we do not know ahead of time whether a prompt will generate 10 tokens or 2,048 tokens, a contiguous cache must reserve `max_seq_len` slots for every request in the batch upfront. In our Phase 1 VRAM test (`test_phase1.py`), reserving `batch_size=8, max_seq_len=2048` immediately consumed **192 MB of VRAM** even when **0 tokens** were stored.
2. **External Fragmentation & Rigid Batching:**
   When different requests have different prompt and output lengths, contiguous allocations cannot easily reclaim variable-sized gaps when a short request finishes early without expensive tensor copies or padding every sequence in the batch to the longest sequence.
3. **No Prefix Sharing:**
   If 10 requests share the same 500-token system prompt, contiguous per-request tensors force the engine to duplicate those 500 KV vectors 10 times in VRAM.

---

### 3. How the Block Manager Works

The `BlockSpaceManager` solves these problems by decoupling a sequence's **logical token view** (contiguous indices `0, 1, 2, ..., N-1`) from **physical VRAM storage** (non-contiguous 16-token blocks `P0, P1, ..., P99`).

```text
Logical View (Sequence 101 - 33 tokens):
  [ Logical Block 0 (tok 0..15) ] -> [ Logical Block 1 (tok 16..31) ] -> [ Logical Block 2 (tok 32) ]
                 |                                  |                                   |
                 v                                  v                                   v
Page Table (seq_id=101):  [ P0, P1, P9 ]  (Non-contiguous physical block indices!)
                 |                                  |                                   |
                 v                                  v                                   v
Physical KV Pool:
  [ P0 (16/16) ][ P1 (16/16) ][ P2..P8 (Owned by Seq 102 & 103) ][ P9 (1/16) ][ P10 ... ]
```

#### Core Mapping Formula (`resolve_logical_token`)
Given a sequence `seq_id` and a `logical_token_idx`, the Block Manager locates the exact KV slot in $O(1)$ time:

$$\text{logical\_block\_idx} = \lfloor \text{logical\_token\_idx} / \text{block\_size} \rfloor$$

$$\text{block\_offset} = \text{logical\_token\_idx} \bmod \text{block\_size}$$

$$\text{physical\_block\_id} = \text{page\_tables}[\text{seq\_id}][\text{logical\_block\_idx}]$$

$$\text{global\_physical\_slot} = \text{physical\_block\_id} \times \text{block\_size} + \text{block\_offset}$$

#### Request Lifecycle in `BlockSpaceManager`
1. **Prefill Allocation (`allocate(seq_id, token_ids)`):**
   * Computes the exact number of blocks required for the prompt: $\lceil \text{len(token\_ids)} / 16 \rceil$.
   * Pops that many block IDs from `free_block_ids`, sets their `ref_count = 1`, fills their token lists, and records the sequence's page table.
2. **Decode Slot Appending (`append_slot(seq_id, token_id)`):**
   * Called once per generated token during autoregressive decoding.
   * **Case A (`num_tokens % 16 != 0`):** The sequence's last physical block still has empty slots. The new token is placed into the next offset of that existing block (**0 new blocks allocated**).
   * **Case B (`num_tokens % 16 == 0`):** The sequence's last physical block is full (`16/16`). A fresh physical block is popped from `free_block_ids` and appended to `page_tables[seq_id]`. Because other sequences may have allocated blocks in the meantime, this new physical block ID does **not** need to be adjacent to the previous block.
3. **Deallocation (`free(seq_id)`):**
   * When a sequence hits `EOS` or `max_new_tokens`, `free(seq_id)` removes its page table entry and decrements `ref_count` on each of its physical blocks.
   * Any block whose `ref_count` reaches `0` is reset and immediately returned to `free_block_ids` for reuse by new or growing sequences.

---

### 4. What Phase 2 Adds to the Overall Inference Process

Implementing the Block Manager is the foundational bridge between naive PyTorch inference (Phase 1) and a production-grade vLLM engine (Phases 3–5):

* **Eliminates KV Cache Waste (<4% Internal Fragmentation):**
  Memory is allocated **on demand** at 16-token granularity. A sequence only ever leaves unused slots in its very last physical block (at most 15 token slots), allowing the 4 GB GTX 1650 to hold significantly more concurrent requests in VRAM.
* **Enables Continuous Batching (Phase 3):**
  Because physical blocks can be allocated and freed independently in $O(1)$ time without moving tensors in memory, the Phase 3 Scheduler can evict a finished 10-token request mid-batch and immediately hand its freed physical blocks to a new waiting request while a 150-token request keeps decoding.
* **Provides the Block Table Tensor for Triton PagedAttention (Phase 4):**
  `get_block_table_tensor(seq_ids)` exports the Python page tables into a padded 2D `int32` tensor (`[batch_size, max_blocks_per_seq]`). In Phase 4, custom Triton kernels will read this tensor directly on the GPU to gather non-contiguous KV blocks during attention computation.
* **Prepares for Reference-Counted Prefix Caching (Phase 5):**
  Each `PhysicalBlock` tracks `ref_count` and `block_hash`. In Phase 5, two requests sharing a long system prompt will hash their 16-token blocks with `xxhash` and point their page tables to the **exact same physical blocks** (`ref_count = 2`), skipping redundant prefill compute and halving KV memory usage.

---

## Running the Tests

All commands use `uv`:

```bash
# Step 0: Verify GPU (sm_75) and environment
uv run verify_env.py

# Phase 1: Run custom safetensors model & naive KV cache test suite
uv run test_phase1.py

# Phase 2: Run BlockSpaceManager virtual memory & page table unit tests
uv run test_phase2.py
```
