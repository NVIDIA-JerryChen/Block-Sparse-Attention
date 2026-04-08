# blk64 Refactor Progress (sla branch)

Goal: Simplify blk64 kernel, align code structure with FA Hopper.

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Remove templates + CLC + BOOL_SWITCH, SingleTileScheduler | DONE |
| 2 | flash.h + bsa_fwd_params + set_params_fprop, remove nested Args/Params | DONE |
| 3 | TORCH_LIBRARY binding, delete bindings.cpp, Python interface | DONE |
| 4 | 5D TMA for Q/O, remove V host transpose, no padding in C++ | DONE |
| 5 | Pipeline cleanup: SM100 PipelineUmmaAsync for SPO/OAcc, CUTLASS API throughout | DONE |

## Phase Details

### Phase 1: Simplify templates + remove CLC
- Remove `HasVarBlockNums`/`HasBlockSizes` template params (hardcode true)
- Remove CLC infrastructure (PipelineCLC, CLCTileScheduler)
- Rewrite `tile_scheduler.hpp` as `SingleTileScheduler` (FA Hopper pattern)
- Remove BOOL_SWITCH dispatch, single instantiation
- Delete `static_switch.h`, old instantiation files
- **Test:** `make setup && make tt BLK=64`

### Phase 2: flash.h + bsa_fwd_params
- Create `flash.h` with `bsa_fwd_params` (FA Hopper naming)
- Add `set_params_fprop()` function
- Remove `CollectiveMainloopFwd::Arguments/Params`
- Remove `CollectiveEpilogueFwd::Arguments/Params`
- Kernel Params = bsa_fwd_params ref + TMA descriptors + scheduler
- **Test:** `make setup && make tt BLK=64`

### Phase 3: Binding + Python interface
- Replace PYBIND11_MODULE with TORCH_LIBRARY
- Delete `bindings.cpp`
- Add `bsa_attn_fwd_blk64()` Python wrapper (bshd/bhsd, validation)
- **Test:** `make setup && make tt BLK=64`

### Phase 4: 5D TMA + remove host transforms
- Q: 5D TMA from BSHD (remove permute+pad+copy)
- O: 5D TMA store to BSHD (remove post-kernel permute)
- V: TMA reads (dim, token) directly — remove host sub-tile transpose
- Remove all padding from C++ (Python ensures alignment)
- **Test:** `make setup && make tt BLK=64`

### Phase 5: Pipeline cleanup
- Remove custom CTA-scope helpers from pipeline.hpp
- Replace with CUTLASS PipelineAsync API (consumer_release, producer_commit)
- Keep raw PTX for UMMA arrives, Q-ready barrier, p_lastsplit wait
- **Test:** `make setup && make tt BLK=64`
