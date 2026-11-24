import argparse
import torch
from transformers import AutoTokenizer

from model import CustomLlamaForCausalLM
from scheduler import ContinuousBatchEngine, SequenceRequest, SequenceStatus


def test_phase3(model_id: str = "Qwen/Qwen2.5-0.5B") -> None:
    assert torch.cuda.is_available(), "CUDA is required for Phase 3 test."
    torch.set_default_dtype(torch.float16)
    device = "cuda"

    print("=== Phase 3 Test: Continuous Batching & Two-Stage Scheduler (Prefill + Decode) ===")
    print(f"Model: {model_id}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = CustomLlamaForCausalLM.from_safetensors(
        model_id,
        dtype=torch.float16,
        device=device,
    )

    # Initialize ContinuousBatchEngine with max_running_seqs=2 so that:
    # - Seq 1 (10-token output) and Seq 2 (150-token output) start in the initial running batch.
    # - Seq 3 (50-token output) waits in the WAITING queue.
    # - As soon as Seq 1 finishes at Step 10 and frees its physical KV blocks,
    #   Seq 3 immediately enters the batch at Step 11 (PREFILL) while Seq 2 is still generating!
    engine = ContinuousBatchEngine(
        model=model,
        num_blocks=32,
        block_size=16,
        max_running_seqs=2,
        max_seq_len=512,
        verbose=True,
    )

    prompts_and_lengths = [
        (1, "The capital of France is", 10),
        (
            2,
            "Write a detailed step-by-step explanation of how operating system virtual memory works:",
            150,
        ),
        (
            3,
            "Explain the difference between a stack and a queue in computer science:",
            50,
        ),
    ]

    requests = []
    for seq_id, prompt, target_len in prompts_and_lengths:
        token_ids = tokenizer.encode(prompt)
        req = SequenceRequest(
            seq_id=seq_id,
            prompt=prompt,
            prompt_token_ids=token_ids,
            max_new_tokens=target_len,
            eos_token_id=tokenizer.eos_token_id,
            force_eos_at_max=True,
        )
        engine.add_request(req)
        requests.append(req)

    print("\n--- Starting Iteration-Level Continuous Batching Loop ---")
    finished_Order = []
    seq3_admitted_while_seq2_running = False
    seq1_evicted_step = None
    seq3_admitted_step = None

    while engine.scheduler.has_unfinished_requests():
        result = engine.step()
        step_num = engine.scheduler.step_counter

        for evicted_seq, _ in result.evicted_seqs:
            finished_Order.append(evicted_seq.seq_id)
            if evicted_seq.seq_id == 1:
                seq1_evicted_step = step_num

        for admitted_seq in result.prefill_seqs:
            if admitted_seq.seq_id == 3:
                seq3_admitted_step = step_num
                running_ids = [s.seq_id for s in engine.scheduler.running]
                if 2 in running_ids:
                    seq3_admitted_while_seq2_running = True

        if (
            result.prefill_seqs
            or result.evicted_seqs
            or step_num % 15 == 0
        ):
            running_summary = ", ".join(
                f"Seq {s.seq_id}({s.num_generated_tokens}/{s.max_new_tokens} tok, "
                f"blocks={engine.block_manager.page_tables[s.seq_id]})"
                for s in engine.scheduler.running
            )
            waiting_summary = (
                ", ".join(f"Seq {s.seq_id}" for s in engine.scheduler.waiting) or "None"
            )
            print(
                f"[Step {step_num:03d} | BATCH STATE]   "
                f"Running=[{running_summary or 'None'}] | Waiting=[{waiting_summary}]"
            )

    print("\n--- Continuous Batching Execution Summary ---")
    print(f"Completion Order: Seq {finished_Order[0]} -> Seq {finished_Order[1]} -> Seq {finished_Order[2]}")
    print(f"Seq 1 (10-tok) evicted at Step:  {seq1_evicted_step}")
    print(f"Seq 3 (50-tok) admitted at Step: {seq3_admitted_step} (while Seq 2 was at token 10/150)")
    print(f"Final BlockSpaceManager Free Blocks: {engine.block_manager.num_free_blocks}/{engine.block_manager.num_blocks}\n")

    for req in requests:
        decoded_text = tokenizer.decode(req.all_token_ids, skip_special_tokens=True)
        print(
            f"Seq {req.seq_id} ({req.num_generated_tokens} generated tokens, status={req.status.value}, "
            f"stop={req.stop_reason}):\n  {decoded_text!r}\n"
        )

    # Assertions verifying continuous batching behavior
    assert all(r.status == SequenceStatus.FINISHED for r in requests), "All requests must finish!"
    assert finished_Order == [1, 3, 2], (
        f"Expected completion order [1, 3, 2] (10-tok, then 50-tok, then 150-tok), got {finished_Order}"
    )
    assert seq3_admitted_while_seq2_running, (
        "Seq 3 should have entered the batch while Seq 2 (150-tok) was still generating!"
    )
    assert seq3_admitted_step == seq1_evicted_step + 1, (
        f"Seq 3 should enter on the very next step after Seq 1 eviction "
        f"(evicted={seq1_evicted_step}, admitted={seq3_admitted_step})"
    )
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks, (
        "All physical KV blocks must be freed after all sequences finish!"
    )

    print("[SUCCESS] Phase 3 Continuous Batching & Two-Stage Scheduler tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()
    test_phase3(model_id=args.model)
