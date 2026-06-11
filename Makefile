.PHONY: build test bench profile

test bench:
	@echo "not implemented — run /run-prompt N in Claude Code"

build:
	@if command -v nvcc > /dev/null 2>&1; then \
		.venv/bin/python setup.py build_ext --inplace; \
	else \
		echo "GPU_STEP: requires Linux + CUDA host."; \
		exit 2; \
	fi

profile:
	@echo "GPU_STEP: requires Linux + CUDA host. Write-only on macOS."
