# Repository Guidelines

## Project Structure & Module Organization

mini-vllm targets Python 3.12. `mini_llm/` contains model configuration, layers, Qwen implementation, weight loading, KV cache, and generation. `tests/` holds correctness tests. `benchmarks/inference/` provides workload preparation, backend execution, and output comparison; `mha_mqa_lab/` contains attention experiments. Keep service configurations in `configs/inference/` and architecture, reproduction guides, and reports in `docs/`. Generated inputs, model links, logs, results, and traces belong in Git-ignored `artifacts/`.

## Build, Test, and Development Commands

Run commands from the repository root:

- `uv sync --frozen --extra test`: install locked runtime and test dependencies.
- `uv build`: build the setuptools source distribution and wheel.
- `.venv/bin/python -m mini_llm.preflight`: verify CUDA, BF16 support, and cached model files.
- `.venv/bin/python -m mini_llm.local_generate --prompt "Explain KV cache." --max-new-tokens 32`: run custom inference.
- `.venv/bin/python -m pytest tests -q`: run the test suite; add `-p no:cacheprovider` when cache writes are restricted.
- `.venv/bin/python benchmarks/inference/run_inference.py --help`: inspect evaluation options.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and type hints for public APIs. Match surrounding code. Format changed Python files with `ruff format` and check them with `ruff check`; Ruff is a separate development tool, currently absent from project dependencies.

Name components by function, such as `run_inference.py` or `vllm-single-gpu.yaml`. Keep schedules in planning documents. Prefer explicit relative paths over custom exported environment variables.

## Testing Guidelines

Use pytest, `tests/test_*.py` filenames, and descriptive `test_*` functions. Extend nearby tests for observable behavior: block boundaries, allocation/release, EOS, context limits, and Hugging Face oracle consistency. No percentage coverage threshold is configured. Some tests require CUDA and cached weights; report skips separately from successful GPU validation.

## Commit & Pull Request Guidelines

History uses imperative prefixes: `feat:`, `fix:`, and `docs:`; for example, `feat: implement block manager`. Keep commits focused.

PRs should explain the problem, resulting behavior, relevant issues, validation commands, and results. Performance changes need baseline comparisons, hardware, model revision, workload, and timing definitions. Update affected documentation and configurations.

## Models & Configuration

Preserve the pinned model revision in `ModelSettings`. PyTorch comes from the `pytorch-cu130` index in `pyproject.toml`; validate GPU compatibility when updating it. Keep vLLM in its own environment. Commit reusable tools and configurations; retain weights, credentials, and raw traces locally. Publish verified findings with reproduction instructions in `docs/reports/`.
