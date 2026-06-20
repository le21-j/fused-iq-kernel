# fused-iq-kernel

Fused 1D complex-IQ classification layer as hand-written **Triton** and **CUDA C++** kernels, registered as PyTorch custom ops and benchmarked 4-way (eager / `torch.compile` / Triton / CUDA) with an honest report card + roofline on the actual profiled GPU.

Spec of record: `../portfolio-projects.md` (Project 3). State: see the `Active:` line in `CLAUDE.md`.

## Status
macOS arm64 (no CUDA): Prompt 1 PASSED (CPU baselines, PROVISIONAL timings). Prompts 2/3/5 are **GPU_STEP** — kernels written locally, compiled/profiled on a remote CUDA host. No fabricated timings.

## Quickstart
```bash
make            # list targets
pytest -q       # parity + shape tests (CUDA-only tests skip on no-CUDA hosts)
```

## Layout
- `kernel_triton/`, `kernel_cuda/`, `kernel_w4a16/` — the kernels
- `baseline/`, `bench/` — reference op + 4-way benchmark + roofline
- `tests/` — parity / shape EXIT checks
- `docs/` — PocketFlow-style chapters (start at `docs/index.md`)

## Honesty
macOS arm64, no CUDA. GPU work parks as `GPU_STEP: ready for remote build`; gated numbers are never fabricated.
