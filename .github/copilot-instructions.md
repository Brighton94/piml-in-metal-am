1. Build for reproducibility, scalability, and rapid experimentation—standardize configs, dependencies, and environments across every stage.
2. Track experiments in **MLflow**, version data & pipelines with **DVC**, and run everything inside **Docker**.
3. Keep all dependencies in `pyproject.toml`; install them once per container.
4. Run **Pytest** on every new or modified script.
5. Follow PEP 8 (4-space indents), add type hints, and include concise docstrings.
6. Record events and errors with the `logging` module; avoid `print`.
7. Prefer NumPy/Pandas vectorization and keep functions small and testable.
8. Use `data/` for datasets, `models/` for architectures, `experiments/` for runs, `utils/` for helpers, and `notebooks/` for exploration.
9. Centralize parameters in `config.yaml` or `config.json`; place environment-specific values in `.env`.
10. Log every model run to MLflow and assert data integrity with unit tests.
11. Cover preprocessing, augmentation, and utilities with Pytest; validate outputs against agreed bounds.
12. Generate modular PyTorch code with MLflow hooks and DVC-ready pipeline snippets.
13. Provide routines for data prep, augmentation, evaluation, and visualization.
14. Scaffold Docker and CI templates with linting and type-checking integrations.
15. Whenever a new Python module is added—or an existing one lacks coverage—create a matching `tests/test_<module>.py` file with Pytest skeletons that hit edge cases and typical inputs.
16. Whenever the public API, configuration, or workflow changes, update the corresponding section of `README.md` in the same commit, adding a concise description and usage example.
