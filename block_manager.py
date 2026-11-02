from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class PhysicalBlock:
    """Represents a single fixed-size physical KV cache block in VRAM."""

    block_id: int
    block_size: int = 16
    ref_count: int = 0
    token_ids: List[int] = field(default_factory=list)
    block_hash: Optional[int] = None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_full(self) -> bool:
        return len(self.token_ids) >= self.block_size

    @property
    def remaining_slots(self) -> int:
        return self.block_size - len(self.token_ids)

    def reset(self) -> None:
        self.ref_count = 0
        self.token_ids.clear()
        self.block_hash = None


class BlockSpaceManager:
    """
    Phase 2: Virtual Memory Manager for Paged KV Cache (nano-vllm / MinivLLM style).

    Divides KV cache memory into fixed-size physical blocks (`block_size` tokens per block)
    and maintains per-sequence page tables mapping:
        Logical_Token_Index -> (Physical_Block_Index, Block_Offset)
    """

    def __init__(self, num_blocks: int = 100, block_size: int = 16) -> None:
        assert num_blocks > 0, "num_blocks must be positive"
        assert block_size > 0, "block_size must be positive"
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Physical block pool: indexed 0 .. num_blocks - 1
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=block_size) for i in range(num_blocks)
        ]
        # Free physical block indices queue
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # Page tables: seq_id -> list of physical block IDs (indexed by logical block index)
        self.page_tables: Dict[int, List[int]] = {}
        # Sequence token lengths: seq_id -> total logical tokens currently stored
        self.seq_num_tokens: Dict[int, int] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_block_ids)

    def _allocate_physical_block(self) -> PhysicalBlock:
        if not self.free_block_ids:
            raise MemoryError("Out of physical KV cache blocks in BlockSpaceManager!")
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        block.ref_count = 1
        block.token_ids = []
        block.block_hash = None
        return block

    def _free_physical_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count > 0, f"Double free detected on physical block {block_id}"
        block.ref_count -= 1
        if block.ref_count == 0:
            block.reset()
            self.free_block_ids.append(block_id)

    def can_allocate(self, num_tokens: int) -> bool:
        """Returns True if enough free physical blocks exist to store `num_tokens`."""
        needed_blocks = math.ceil(num_tokens / self.block_size)
        return needed_blocks <= self.num_free_blocks

    def allocate(self, seq_id: int, token_ids: Sequence[int]) -> List[int]:
        """
        Allocates physical blocks for a new sequence `seq_id` with initial `token_ids`.
        Returns the assigned page table (list of physical block IDs).
        """
        if seq_id in self.page_tables:
            raise ValueError(f"Sequence {seq_id} is already allocated in page_tables.")

        num_tokens = len(token_ids)
        needed_blocks = math.ceil(num_tokens / self.block_size)
        if needed_blocks > self.num_free_blocks:
            raise MemoryError(
                f"Cannot allocate {needed_blocks} blocks for seq_id={seq_id} "
                f"(only {self.num_free_blocks} free blocks available)."
            )

        assigned_blocks: List[int] = []
        for logical_block_idx in range(needed_blocks):
            start = logical_block_idx * self.block_size
            end = min(start + self.block_size, num_tokens)
            chunk = list(token_ids[start:end])

            block = self._allocate_physical_block()
            block.token_ids.extend(chunk)
            assigned_blocks.append(block.block_id)

        self.page_tables[seq_id] = assigned_blocks
        self.seq_num_tokens[seq_id] = num_tokens
        return list(assigned_blocks)

    def can_append_slot(self, seq_id: int) -> bool:
        """
        Returns True if `seq_id` can append 1 new token during decoding.
        A new physical block is only needed when the current last block is full.
        """
        if seq_id not in self.page_tables:
            raise KeyError(f"Unknown seq_id={seq_id}")
        curr_tokens = self.seq_num_tokens[seq_id]
        needs_new_block = (curr_tokens % self.block_size) == 0
        return (not needs_new_block) or (self.num_free_blocks >= 1)

    def append_slot(self, seq_id: int, token_id: int) -> Tuple[int, int, int]:
        """
        Appends a single newly generated token to `seq_id`.
        Allocates a new physical block if the last logical block is full.
        Returns (physical_block_id, block_offset, physical_slot_idx).
        """
        if seq_id not in self.page_tables:
            raise KeyError(f"Unknown seq_id={seq_id}")

        logical_token_idx = self.seq_num_tokens[seq_id]
        block_offset = logical_token_idx % self.block_size

        if block_offset == 0:
            # Current last block is full (or sequence had 0 tokens); allocate a new physical block
            new_block = self._allocate_physical_block()
            self.page_tables[seq_id].append(new_block.block_id)
            physical_block_id = new_block.block_id
        else:
            physical_block_id = self.page_tables[seq_id][-1]

        self.blocks[physical_block_id].token_ids.append(token_id)
        self.seq_num_tokens[seq_id] = logical_token_idx + 1
        physical_slot_idx = physical_block_id * self.block_size + block_offset
        return physical_block_id, block_offset, physical_slot_idx

    def resolve_logical_token(self, seq_id: int, logical_token_idx: int) -> Tuple[int, int, int]:
        """
        Core Page Table Mapping:
            Logical_Token_Index -> (Physical_Block_Index, Block_Offset, Physical_Slot_Index)
        """
        if seq_id not in self.page_tables:
            raise KeyError(f"Unknown seq_id={seq_id}")
        total_tokens = self.seq_num_tokens[seq_id]
        if not (0 <= logical_token_idx < total_tokens):
            raise IndexError(
                f"logical_token_idx {logical_token_idx} out of bounds for seq_id={seq_id} "
                f"with {total_tokens} tokens."
            )

        logical_block_idx = logical_token_idx // self.block_size
        block_offset = logical_token_idx % self.block_size
        physical_block_id = self.page_tables[seq_id][logical_block_idx]
        physical_slot_idx = physical_block_id * self.block_size + block_offset
        return physical_block_id, block_offset, physical_slot_idx

    def free(self, seq_id: int) -> List[int]:
        """
        Frees all physical blocks belonging to `seq_id` when the request finishes.
        Returns the list of physical block IDs that were released.
        """
        if seq_id not in self.page_tables:
            raise KeyError(f"Unknown seq_id={seq_id}")

        freed_blocks = list(self.page_tables.pop(seq_id))
        self.seq_num_tokens.pop(seq_id)
        for block_id in freed_blocks:
            self._free_physical_block(block_id)
        return freed_blocks

    def get_block_table_tensor(
        self, seq_ids: Sequence[int], device: str = "cpu"
    ) -> torch.Tensor:
        """
        Builds a padded 2D int32 block table tensor [batch_size, max_blocks_per_seq]
        for the given active `seq_ids` (padded with -1).
        """
        if not seq_ids:
            return torch.empty((0, 0), dtype=torch.int32, device=device)
        max_blocks = max(len(self.page_tables[sid]) for sid in seq_ids)
        table = torch.full((len(seq_ids), max_blocks), -1, dtype=torch.int32, device=device)
        for row_idx, sid in enumerate(seq_ids):
            blocks = self.page_tables[sid]
            table[row_idx, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=device)
        return table

    def format_page_table(self) -> str:
        """Returns a formatted console summary of current page tables and memory usage."""
        lines = [
            f"BlockSpaceManager Status: {self.num_used_blocks}/{self.num_blocks} blocks used "
            f"| {self.num_free_blocks} blocks free (block_size={self.block_size} tokens)"
        ]
        if not self.page_tables:
            lines.append("  [Page Table is empty - no active requests]")
            return "\n".join(lines)

        for seq_id, phys_blocks in self.page_tables.items():
            n_tok = self.seq_num_tokens[seq_id]
            last_block_fill = self.blocks[phys_blocks[-1]].num_tokens if phys_blocks else 0
            mapping_str = ", ".join(
                f"L{lb}->P{pb}({self.blocks[pb].num_tokens}/{self.block_size})"
                for lb, pb in enumerate(phys_blocks)
            )
            lines.append(
                f"  Seq {seq_id}: {n_tok:3d} tokens | {len(phys_blocks)} blocks "
                f"[last block: {last_block_fill}/{self.block_size} tokens] | {mapping_str}"
            )
        return "\n".join(lines)

    def print_page_table(self) -> None:
        print(self.format_page_table())

    @classmethod
    def calculate_num_blocks_from_vram(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        vram_utilization: float = 0.7,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[int, int]:
        """
        Calculates how many physical KV blocks fit in currently free GPU VRAM.
        Returns (num_blocks, bytes_per_block).
        """
        free_vram_bytes, _ = torch.cuda.mem_get_info()
        usable_bytes = int(free_vram_bytes * vram_utilization)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        # Each block stores K and V across all layers: 2 * num_layers * num_kv_heads * block_size * head_dim
        bytes_per_block = 2 * num_layers * num_kv_heads * block_size * head_dim * elem_size
        num_blocks = usable_bytes // bytes_per_block
        return num_blocks, bytes_per_block
