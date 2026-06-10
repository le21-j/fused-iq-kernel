# Tutorial: fused-iq-kernel

A **fused complex-IQ classification layer** written four ways — PyTorch eager,
`torch.compile`, hand-written Triton, and hand-written CUDA C++ — registered as
proper PyTorch custom ops and benchmarked head-to-head with an honest report
card (including a roofline analysis on the actual profiled GPU).

**Source spec:** `../portfolio-projects.md` (Project 3) · **State:** see `Active:` line in `CLAUDE.md`

## Core abstractions (planned — chapters land as prompts PASS)

```mermaid
flowchart TD
    A0["IQ layout decision (interleaved vs planar)"] --> A1["Reference op (eager + compile)"]
    A0 --> A2["Triton kernel"]
    A0 --> A3["CUDA C++ kernel"]
    A1 --> A4["Custom-op registration (torch.library)"]
    A2 --> A4
    A3 --> A4
    A4 --> A5["Parity tests, 4-way benchmark, roofline"]
    A6["Agent harness & GPU gates"] -.-> A0
```

## Chapters

*None yet — Prompt 1 not started.* Chapters are added at each prompt EXIT,
following the per-abstraction tutorial format of P1
(`../neural-channel-estimator/docs/`).

Spec-locked decisions worth knowing up front:

1. **IQ memory layout** is decided once in Prompt 1 (`docs/design.md`) — every
   later kernel must match it.
2. Registration uses the **direct `torch.library.Library` API** with a
   `register_fake` meta-kernel (not the decorator API).
3. Prompts 2/3/5 are **GPU_STEP** on this macOS machine: kernels get written
   locally, compiled/profiled only on a remote CUDA host. No fabricated timings.

---
*Maintained in the style of [PocketFlow-Tutorial-Codebase-Knowledge](https://github.com/The-Pocket/PocketFlow-Tutorial-Codebase-Knowledge); updated at each prompt EXIT.*
