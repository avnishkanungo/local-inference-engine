from block_manager import BlockSpaceManager


def test_block_space_manager() -> None:
    print("=== Phase 2 Unit Test: BlockSpaceManager (Virtual Memory for KV Cache) ===\n")

    # 1. Initialize BlockSpaceManager with 100 blocks (16 tokens per block)
    manager = BlockSpaceManager(num_blocks=100, block_size=16)
    print("[Step 1] Initialized BlockSpaceManager with 100 blocks (16 tokens/block):")
    manager.print_page_table()
    assert manager.num_free_blocks == 100
    assert manager.num_used_blocks == 0

    # 2. Simulate adding 3 requests of varying lengths
    #    - Request 101: 20 tokens -> requires ceil(20/16) = 2 blocks (P0, P1)
    #    - Request 102: 50 tokens -> requires ceil(50/16) = 4 blocks (P2, P3, P4, P5)
    #    - Request 103: 35 tokens -> requires ceil(35/16) = 3 blocks (P6, P7, P8)
    print("\n[Step 2] Allocating 3 requests of varying prompt lengths (20, 50, 35 tokens):")
    req1_tokens = list(range(1000, 1020))  # 20 tokens
    req2_tokens = list(range(2000, 2050))  # 50 tokens
    req3_tokens = list(range(3000, 3035))  # 35 tokens

    blocks_101 = manager.allocate(seq_id=101, token_ids=req1_tokens)
    blocks_102 = manager.allocate(seq_id=102, token_ids=req2_tokens)
    blocks_103 = manager.allocate(seq_id=103, token_ids=req3_tokens)

    manager.print_page_table()

    assert blocks_101 == [0, 1]
    assert blocks_102 == [2, 3, 4, 5]
    assert blocks_103 == [6, 7, 8]
    assert manager.num_used_blocks == 9
    assert manager.num_free_blocks == 91

    # 3. Verify Logical_Token_Index -> Physical_Block_Index mapping
    print("\n[Step 3] Verifying Logical_Token_Index -> Physical_Block_Index mappings:")
    for seq_id, sample_logical_idx in [(101, 0), (101, 19), (102, 16), (102, 49), (103, 34)]:
        phys_block, offset, phys_slot = manager.resolve_logical_token(seq_id, sample_logical_idx)
        print(
            f"  Seq {seq_id} | Logical Token {sample_logical_idx:2d} -> "
            f"Physical Block P{phys_block} (offset {offset:2d}, global slot {phys_slot:3d})"
        )

    assert manager.resolve_logical_token(101, 19) == (1, 3, 1 * 16 + 3)
    assert manager.resolve_logical_token(102, 49) == (5, 1, 5 * 16 + 1)
    assert manager.resolve_logical_token(103, 34) == (8, 2, 8 * 16 + 2)

    # 4. Simulate token generation (decode steps) causing NON-CONTIGUOUS physical block assignment
    #    Seq 101 currently has 20 tokens (P0 has 16, P1 has 4).
    #    Appending 13 more tokens to Seq 101 fills P1 (12 tokens) and overflows the 13th token
    #    into a newly allocated physical block P9 -> Seq 101 now owns non-contiguous blocks [0, 1, 9]!
    print(
        "\n[Step 4] Simulating decode token appends for Seq 101 (+13 tokens) and Seq 103 (+14 tokens) "
        "to demonstrate non-contiguous physical block allocation:"
    )
    for step in range(13):
        manager.append_slot(seq_id=101, token_id=9000 + step)
    for step in range(14):
        manager.append_slot(seq_id=103, token_id=8000 + step)

    manager.print_page_table()
    assert manager.page_tables[101] == [0, 1, 9], f"Expected non-contiguous [0, 1, 9], got {manager.page_tables[101]}"
    assert manager.page_tables[103] == [6, 7, 8, 10], f"Expected non-contiguous [6, 7, 8, 10], got {manager.page_tables[103]}"
    print(f"  -> Confirmed non-contiguous page table for Seq 101: {manager.page_tables[101]}")
    print(f"  -> Confirmed non-contiguous page table for Seq 103: {manager.page_tables[103]}")

    # 5. Simulate Request 102 finishing and freeing its physical blocks [2, 3, 4, 5]
    print("\n[Step 5] Simulating Request 102 finishing and freeing its blocks:")
    freed_102 = manager.free(seq_id=102)
    print(f"  Freed physical blocks from Seq 102: {freed_102}")
    manager.print_page_table()
    assert freed_102 == [2, 3, 4, 5]
    assert 102 not in manager.page_tables
    assert manager.num_used_blocks == 7
    assert manager.num_free_blocks == 93

    # 6. Check 2D padded block table tensor generation for Triton kernels
    block_tensor = manager.get_block_table_tensor([101, 103])
    print(f"\n[Step 6] Padded 2D Block Table Tensor for active sequences [101, 103]:\n{block_tensor}")
    assert block_tensor.shape == (2, 4)
    assert block_tensor[0].tolist() == [0, 1, 9, -1]
    assert block_tensor[1].tolist() == [6, 7, 8, 10]

    # 7. Free remaining requests and confirm all 100 blocks are back in the free pool
    print("\n[Step 7] Freeing remaining requests (Seq 101 and Seq 103):")
    manager.free(seq_id=101)
    manager.free(seq_id=103)
    manager.print_page_table()
    assert manager.num_free_blocks == 100
    assert manager.num_used_blocks == 0

    print("\n[SUCCESS] Phase 2 BlockSpaceManager unit tests passed!")


if __name__ == "__main__":
    test_block_space_manager()
