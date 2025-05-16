### Project Foundations
1. Build for reproducibility, scalability, and rapid experimentation—standardize configs, dependencies, and environments across every stage.

### Core Tooling
1. Track experiments in **MLflow**, version data & pipelines with **DVC**, and run everything inside **Docker**.
2. Keep all dependencies in `pyproject.toml`; install them once per container.
3. Run **Pytest** on every new or modified script.

### Code Style
1. Follow PEP 8 (4-space indents), add type hints, and include concise docstrings.
2. Record events and errors with the `logging` module; avoid `print`.
3. Prefer NumPy/Pandas vectorization and keep functions small and testable.

### Repository Layout
1. `data/` for datasets, `models/` for architectures, `experiments/` for runs, `utils/` for helpers, and `notebooks/` for exploration.
2. Centralize parameters in `config.yaml` or `config.json`; place environment-specific values in `.env`.

### Model & Data Practices
1. Log every model run to MLflow and assert data integrity with unit tests.

### Testing & QA
1. Cover preprocessing, augmentation, and utilities with Pytest; validate outputs against agreed bounds.

### Copilot Prompts
1. Generate modular PyTorch code with MLflow hooks and DVC-ready pipeline snippets.
2. Provide routines for data prep, augmentation, evaluation, and visualization.
3. Scaffold Docker and CI templates with linting and type-checking integrations.
