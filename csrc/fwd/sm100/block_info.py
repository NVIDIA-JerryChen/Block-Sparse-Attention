# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def get_n_block_idx(
        self,
        block_index: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        i: Int32,
    ) -> Int32:
        return block_index[batch_idx, head_idx, m_block, i]
