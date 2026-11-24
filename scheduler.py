from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence as SeqType, Tuple

import torch

from block_manager import BlockSpaceManager
from model import ContiguousKVCache, CustomLlamaForCausalLM


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


@dataclass
class SequenceRequest:
    """Represents a single inference request managed by the ContinuousBatchScheduler."""

    seq_id: int
    prompt: str
    prompt_token_ids: List[int]
    max_new_tokens: int
    eos_token_id: Optional[int] = None
    force_eos_at_max: bool = True
    status: SequenceStatus = SequenceStatus.WAITING
    output_token_ids: List[int] = field(default_factory=list)
    stop_reason: Optional[str] = None
    prefilled: bool = False

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]


@dataclass
class SchedulerStepResult:
    """Metadata returned by the scheduler for a single iteration."""

    prefill_seqs: List[SequenceRequest]
    decode_seqs: List[SequenceRequest]
    evicted_seqs: List[Tuple[SequenceRequest, List[int]]]  # (seq, freed_block_ids)


class ContinuousBatchScheduler:
    """
    Phase 3: Two-Stage Iteration-Level Scheduler (Prefill & Decode).

    Manages `waiting`, `running`, and `finished` queues on top of `BlockSpaceManager`:
      1. Pulls waiting requests into `running` whenever batch capacity and physical
         KV blocks permit, scheduling them for the `PREFILL` stage.
      2. Schedules already-prefilled active requests for the `DECODE` stage (1 token/step),
         appending a physical block slot in `BlockSpaceManager` per sequence.
      3. Immediately evicts any sequence that hits an `eos_token_id` (or `max_new_tokens`),
         freeing its physical blocks in `BlockSpaceManager` so waiting requests can enter
         the batch on the very next iteration.
    """

    def __init__(
        self,
        block_manager: BlockSpaceManager,
        max_running_seqs: int = 4,
        verbose: bool = True,
    ) -> None:
        self.block_manager = block_manager
        self.max_running_seqs = max_running_seqs
        self.verbose = verbose

        self.waiting: Deque[SequenceRequest] = deque()
        self.running: List[SequenceRequest] = []
        self.finished: List[SequenceRequest] = []
        self.step_counter: int = 0

    def add_request(self, request: SequenceRequest) -> None:
        request.status = SequenceStatus.WAITING
        self.waiting.append(request)
        if self.verbose:
            print(
                f"[Scheduler] + Queued Seq {request.seq_id} into WAITING "
                f"(prompt_tokens={request.num_prompt_tokens}, target_output={request.max_new_tokens})"
            )

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def schedule(self) -> Tuple[List[SequenceRequest], List[SequenceRequest]]:
        """
        Determines which sequences run in PREFILL and which run in DECODE for this iteration.
        Returns (prefill_seqs, decode_seqs).
        """
        self.step_counter += 1
        prefill_seqs: List[SequenceRequest] = []
        decode_seqs: List[SequenceRequest] = []

        # 1. Existing running sequences that are already prefilled will run a DECODE step.
        for seq in self.running:
            if seq.prefilled:
                decode_seqs.append(seq)

        # 2. Reserve awareness: each decode sequence at a block boundary (num_tokens % block_size == 0)
        #    needs 1 free block in this iteration.
        blocks_needed_for_decode = sum(
            1
            for seq in decode_seqs
            if (self.block_manager.seq_num_tokens[seq.seq_id] % self.block_manager.block_size) == 0
        )

        # 3. Pull requests from WAITING -> RUNNING for PREFILL if batch size & blocks allow.
        while self.waiting and len(self.running) < self.max_running_seqs:
            candidate = self.waiting[0]
            needed_blocks = (
                candidate.num_prompt_tokens + self.block_manager.block_size - 1
            ) // self.block_manager.block_size

            if needed_blocks + blocks_needed_for_decode <= self.block_manager.num_free_blocks:
                self.waiting.popleft()
                assigned_blocks = self.block_manager.allocate(
                    candidate.seq_id, candidate.prompt_token_ids
                )
                candidate.status = SequenceStatus.RUNNING
                self.running.append(candidate)
                prefill_seqs.append(candidate)
                if self.verbose:
                    print(
                        f"[Step {self.step_counter:03d} | PREFILL ADMIT] Seq {candidate.seq_id} "
                        f"pulled WAITING -> RUNNING | Allocated blocks {assigned_blocks} "
                        f"({self.block_manager.num_used_blocks}/{self.block_manager.num_blocks} used, "
                        f"{self.block_manager.num_free_blocks} free)"
                    )
            else:
                # Not enough free physical blocks yet; wait until a running sequence finishes.
                break

        return prefill_seqs, decode_seqs

    def update_after_step(
        self,
        new_tokens: Dict[int, int],
    ) -> List[Tuple[SequenceRequest, List[int]]]:
        """
        Appends newly generated tokens to active sequences, updates BlockSpaceManager slots
        for decode sequences, and immediately evicts any sequence that hits EOS or max_new_tokens.
        Returns list of (evicted_sequence, freed_block_ids).
        """
        evicted: List[Tuple[SequenceRequest, List[int]]] = []
        remaining_running: List[SequenceRequest] = []

        for seq in self.running:
            if seq.seq_id not in new_tokens:
                remaining_running.append(seq)
                continue

            next_token = new_tokens[seq.seq_id]
            was_prefill = not seq.prefilled
            seq.prefilled = True

            # Append token slot into BlockSpaceManager virtual page table
            phys_block, offset, _ = self.block_manager.append_slot(seq.seq_id, next_token)
            seq.output_token_ids.append(next_token)

            if not was_prefill and offset == 0 and self.verbose:
                print(
                    f"[Step {self.step_counter:03d} | BLOCK GROW]    Seq {seq.seq_id} "
                    f"crossed 16-token boundary ({seq.num_tokens} total tokens) -> "
                    f"assigned new physical block P{phys_block} "
                    f"(page_table={self.block_manager.page_tables[seq.seq_id]})"
                )

            # Check if sequence hit EOS or reached its target generation length
            hit_natural_eos = (
                seq.eos_token_id is not None and next_token == seq.eos_token_id
            )
            hit_max_tokens = seq.num_generated_tokens >= seq.max_new_tokens

            if hit_natural_eos or hit_max_tokens:
                if hit_max_tokens and seq.force_eos_at_max and seq.eos_token_id is not None:
                    # Mark final token as EOS if synthetic length limit triggers EOS eviction
                    seq.output_token_ids[-1] = seq.eos_token_id
                    hit_natural_eos = True

                seq.status = SequenceStatus.FINISHED
                seq.stop_reason = "EOS_TOKEN" if hit_natural_eos else "MAX_TOKENS"
                freed_blocks = self.block_manager.free(seq.seq_id)
                self.finished.append(seq)
                evicted.append((seq, freed_blocks))

                if self.verbose:
                    print(
                        f"[Step {self.step_counter:03d} | EVICT & FREE]  Seq {seq.seq_id} "
                        f"hit {seq.stop_reason} after {seq.num_generated_tokens} generated tokens! "
                        f"Evicted from RUNNING -> FINISHED | Freed blocks {freed_blocks} "
                        f"({self.block_manager.num_used_blocks}/{self.block_manager.num_blocks} used, "
                        f"{self.block_manager.num_free_blocks} free)"
                    )
            else:
                remaining_running.append(seq)

        self.running = remaining_running
        return evicted


class ContinuousBatchEngine:
    """
    Phase 3 Iteration-Level Execution Engine combining:
      - CustomLlamaForCausalLM (Phase 1 model)
      - BlockSpaceManager (Phase 2 virtual memory page table)
      - ContinuousBatchScheduler (Phase 3 two-stage Prefill & Decode scheduler)
    """

    def __init__(
        self,
        model: CustomLlamaForCausalLM,
        num_blocks: int = 32,
        block_size: int = 16,
        max_running_seqs: int = 4,
        max_seq_len: int = 512,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.device = str(next(model.parameters()).device)
        self.dtype = next(model.parameters()).dtype
        self.max_seq_len = max_seq_len
        self.verbose = verbose

        self.block_manager = BlockSpaceManager(num_blocks=num_blocks, block_size=block_size)
        self.scheduler = ContinuousBatchScheduler(
            block_manager=self.block_manager,
            max_running_seqs=max_running_seqs,
            verbose=verbose,
        )
        # Per-sequence active KV state while running (freed immediately when evicted)
        self._active_kv_caches: Dict[int, ContiguousKVCache] = {}

    def add_request(self, request: SequenceRequest) -> None:
        self.scheduler.add_request(request)

    @torch.inference_mode()
    def step(self) -> SchedulerStepResult:
        """
        Executes one iteration of Continuous Batching:
          Stage 1 (PREFILL): Processes any newly admitted requests from the waiting queue.
          Stage 2 (DECODE):  Advances all already-running sequences by 1 token.
          Stage 3 (EVICT):   Immediately evicts sequences hitting EOS and frees their KV blocks.
        """
        prefill_seqs, decode_seqs = self.scheduler.schedule()
        new_tokens: Dict[int, int] = {}

        # Stage 1: PREFILL newly admitted sequences
        for seq in prefill_seqs:
            cfg = self.model.config
            kv_cache = ContiguousKVCache(
                num_layers=cfg.num_hidden_layers,
                batch_size=1,
                num_kv_heads=cfg.num_key_value_heads,
                max_seq_len=max(self.max_seq_len, seq.num_prompt_tokens + seq.max_new_tokens + 8),
                head_dim=cfg.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self._active_kv_caches[seq.seq_id] = kv_cache

            input_ids = torch.tensor(
                [seq.prompt_token_ids], dtype=torch.long, device=self.device
            )
            logits = self.model.forward(input_ids, kv_cache=kv_cache, start_pos=0)
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            new_tokens[seq.seq_id] = next_token

        # Stage 2: DECODE existing running sequences (1 token each)
        for seq in decode_seqs:
            kv_cache = self._active_kv_caches[seq.seq_id]
            start_pos = seq.num_tokens - 1
            curr_input = torch.tensor(
                [[seq.last_token_id]], dtype=torch.long, device=self.device
            )
            logits = self.model.forward(curr_input, kv_cache=kv_cache, start_pos=start_pos)
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            new_tokens[seq.seq_id] = next_token

        # Stage 3: Update page tables & evict finished sequences
        evicted = self.scheduler.update_after_step(new_tokens)
        for evicted_seq, _ in evicted:
            self._active_kv_caches.pop(evicted_seq.seq_id, None)

        return SchedulerStepResult(
            prefill_seqs=prefill_seqs,
            decode_seqs=decode_seqs,
            evicted_seqs=evicted,
        )

    def run_to_completion(self, log_interval: int = 10) -> List[SequenceRequest]:
        """Runs the continuous batching loop until both waiting and running queues are empty."""
        while self.scheduler.has_unfinished_requests():
            result = self.step()
            step_num = self.scheduler.step_counter

            if self.verbose and (
                result.prefill_seqs
                or result.evicted_seqs
                or step_num % log_interval == 0
            ):
                running_summary = ", ".join(
                    f"Seq {s.seq_id}({s.num_generated_tokens}/{s.max_new_tokens} tok, "
                    f"blocks={self.block_manager.page_tables[s.seq_id]})"
                    for s in self.scheduler.running
                )
                waiting_summary = (
                    ", ".join(f"Seq {s.seq_id}" for s in self.scheduler.waiting) or "None"
                )
                print(
                    f"[Step {step_num:03d} | BATCH STATE]   "
                    f"Running=[{running_summary or 'None'}] | Waiting=[{waiting_summary}]"
                )

        return self.scheduler.finished
