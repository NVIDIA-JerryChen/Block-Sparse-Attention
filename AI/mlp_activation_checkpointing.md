# MLP activation recomputation

## Summary

QuACK's `MLP` supports three modes for trading memory vs compute:

| Mode | Forward GEMMs | Backward GEMMs | Total | Saved activations | How |
|---|---|---|---|---|---|
| **normal** | 2 | 4 | **6** | `x` + `preact` | default |
| **`torch.utils.checkpoint`** | 2 | 2 (replay) + 4 = 6 | **8** | `x` only | wraps entire MLP |
| **checkpoint + torch.compile** | 2 | 2 (replay) + 4 = 6 | **8** | `x` only | compile can't help |
| **`recompute=True`** | 2 | 1 (replay fc1) + 4 = 5 | **7** | `x` only | built-in, replays only fc1 |

`recompute=True` is strictly better than `torch.utils.checkpoint` — same memory savings,
one fewer GEMM in backward. Adding `torch.compile` on top of checkpoint doesn't help:
the compiler can't see inside custom `autograd.Function` subclasses to know that fc2's
replay is redundant (since `gemm_dact` already recomputes `postact` from `preact`).

## Background

Without recomputation, the MLP forward saves two things for backward:
- **fc1** (`LinearActFunc`): saves `x` (input to fc1) — a view of the original input, so ~0 extra
- **fc2** (`DActLinearFunc`): saves `preact` (fc1's pre-activation output) — this is the big one

Note: fc2 already partially recomputes — it stores `preact` instead of `postact` and recomputes
`postact` from `preact` during backward via `gemm_dact`. So the only "extra" activation held
across forward→backward is `preact`.

`torch.utils.checkpoint` avoids saving `preact` but replays the ENTIRE forward (fc1 + fc2 =
2 GEMMs) during backward. `recompute=True` is smarter: it replays only fc1 (1 GEMM) since
that's the only activation that needs recomputing. The backward `gemm_dact` already recomputes
`postact` from `preact`, so replaying fc2 is wasted work.

## GEMM kernel breakdown

Profiled on H100, bf16, shape 4×512×4096, hidden=16384
(see `scripts/mlp_checkpoint_bench.py --no-memory` to reproduce):

**normal** (6 GEMMs):
```
  fwd:  1× GemmAct (fc1: x @ W1 + act)  +  1× GemmDefault (fc2: postact @ W2)
  bwd:  1× GemmDAct (dout @ W2 + dact)  +  3× GemmDefault (dW2, dx, dW1)
```

**torch.utils.checkpoint** (8 GEMMs — same with or without torch.compile):
```
  fwd:  1× GemmAct + 1× GemmDefault
  bwd:  1× GemmAct + 1× GemmDefault (replay)  +  1× GemmDAct + 3× GemmDefault (bwd)
```
torch.compile cannot eliminate the replayed GemmAct + GemmDefault because the custom
autograd Functions are opaque graph nodes to the compiler.

**recompute=True** (7 GEMMs):
```
  fwd:  1× GemmAct + 1× GemmDefault
  bwd:  1× GemmDefault (replay fc1 only)  +  1× GemmDAct + 3× GemmDefault (bwd)
```

## Memory results (H100, bf16, 4×2048×4096 input, hidden=16384, 4 layers)

See `scripts/mlp_checkpoint_bench.py --no-kernels` to reproduce.

```
                                    normal    checkpoint    recompute
forward activation memory          1342 MB       268 MB       268 MB
peak memory                        3221 MB      3020 MB      3020 MB
```

- Forward savings = N_layers × preact_size (268 MB × 4 = 1074 MB)
- Both `checkpoint` and `recompute` have the same memory profile
- `recompute` saves 1 GEMM per layer vs `checkpoint`

## Usage

Built-in recompute (preferred):
```python
mlp = MLP(d_model, hidden, activation="swiglu", recompute=True)
# or functional:
out = mlp_func(x, w1, w2, activation="swiglu", recompute=True)
```

`torch.utils.checkpoint` (also works, but 1 extra GEMM per layer):
```python
# In a transformer block:
def forward(self, x):
    x = x + torch.utils.checkpoint.checkpoint(self.mlp, x, use_reentrant=False)
    return x
```

## `activation_memory_budget` (torch.compile SAC) — does NOT apply here

`torch._functorch.config.activation_memory_budget` is a separate mechanism:
- Requires `torch.compile` to trace the joint forward-backward graph
- Uses a min-cut partitioner to auto-decide which ops to save vs recompute (budget 0.0–1.0)
- Cannot see inside custom `autograd.Function` subclasses — QuACK's `LinearActFunc`,
  `DActLinearFunc`, etc. are opaque to the compiler

Since QuACK's MLP uses custom autograd Functions with hand-written `save_for_backward` /
`backward`, the `activation_memory_budget` knob has **no effect**. Use `recompute=True`
instead.

| Approach | Requires torch.compile | Works with custom autograd.Function | Granularity |
|---|---|---|---|
| `activation_memory_budget` | Yes | No (opaque) | Per-op (auto) |
| `torch.utils.checkpoint` | No | Yes | Per-layer (manual) |
| `checkpoint` + `torch.compile` | Yes | Yes, but can't optimize replay | Per-layer (manual) |
| `recompute=True` | No | N/A (built-in) | Per-MLP (optimal) |

## Zero-stride gradients (fixed)

`out.sum().backward()` produces an all-ones gradient via `expand()` with zero strides `(0, 0)`.
The GEMM kernels require contiguous strides (TMA needs a known contiguous dimension).

Fixed in `linear.py` via `_ensure_contiguous(dout)` in both `LinearFunc.backward` and
`DActLinearFunc.backward`. The helper checks `stride(-1) != 1` to avoid copies in the common
case; under `torch.compile` it calls `.contiguous()` unconditionally (dynamo can't inspect
strides on fake tensors).
