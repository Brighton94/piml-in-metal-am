# copilot-instructions.md

## Project Overview
This project—**Physics Informed Machine Learning for Metal AM**—aims to predict spatter and recoater streaking in metal 3D printing processes. It leverages both image and physics-based data to train Vision Transformer (ViT) models. The project is designed with reproducibility, scalability, and extensive experimentation in mind. Project configurations, dependency management, and environment setup are standardized through modern tools to ensure consistency across deployments and experiments.

## Tech Stack & Tools
- **PyTorch**: Core framework for building and training deep learning models (e.g., Vision Transformers).
- **Hugging Face Transformers**: Provides pre-built ViT architectures and utilities for model configuration.
- **MLflow**: Manages experiment tracking, logging of hyperparameters, metrics, and model lifecycle.
- **DVC**: Tracks data versioning and end-to-end pipelines, ensuring reproducibility of data preprocessing and model training stages.
- **Docker**: Offers containerized environments that guarantee consistency across development, testing, and production setups.
- **Additional Dependencies** (specified in pyproject.toml):
  - **Numpy, Pandas, H5py**: For efficient data manipulation and numerical computations.
  - **Matplotlib & Seaborn**: For visualization of results and analysis of data trends.
  - **Scikit-learn**: For classical ML preprocessing and evaluation.
  - **OpenCV-Python**: For advanced image processing tasks.
  - **Jupyter & IPython Kernel**: For interactive experimentation and prototyping.
  - **Python-Dotenv**: For secure and flexible environment variable management.
  - **Ttkbootstrap**: (if applicable) for enhancing GUI elements.
- **Development & Testing Tools**:
  - **Pytest**: For unit and integration testing.
  - **Ruff**: To enforce code quality, style guidelines (PEP8), and static type checking.
  - **Pre-commit Hooks**: Integrated via tools like `ruff` and `pre-commit` to ensure all committed code meets quality standards.

## Coding Style & Conventions
- **Indentation & Formatting**:
  - Use **4 spaces** per indentation level.
  - Follow [PEP8](https://peps.python.org/pep-0008/) for all Python code.
- **Type Hints & Documentation**:
  - Use type hints for every function signature.
  - Provide comprehensive docstrings for public functions, classes, and modules.
- **Logging & Debugging**:
  - Log key processing events and errors using the built-in `logging` module instead of print statements.
- **Efficiency**:
  - Favor vectorized operations (e.g., using NumPy, pandas) over explicit loops when possible.
- **Modular & Testable Design**:
  - Write small, composable functions that can be independently tested.

**Best Practice Tip:** Integrate linters (like Ruff and Pylint) and type checkers (Mypy) into your development workflow to catch errors and enforce coding standards early.

## File/Directory Structure Conventions
- **Data Management**:
  - Keep raw and processed datasets in the `data/` directory, with full tracking via DVC.
- **Source Code & Experimentation**:
  - Place model definitions under `models/` and core experiments or training routines in `experiments/`.
  - Store reusable utilities in the `utils/` directory.
  - Shift exploratory notebooks (EDA/prototyping) to the `notebooks/` directory, with critical code migrated to well-documented Python files.
- **Configuration & Environment**:
  - Use centralized configuration files (e.g., `config.yaml` or `config.json`) to manage training parameters, data paths, and experiment settings.
  - Environment-specific settings (like dataset paths and user IDs) should be managed via `.env` files (as described in the README) rather than hard-coded.

## Model & Data Conventions
- **Model Architecture**:
  - Leverage Vision Transformer (ViT) models from Hugging Face for image processing tasks.
  - Standardize image preprocessing (resizing, normalization) before training.
- **Experiment Tracking**:
  - Register and log all trained models in MLflow, ensuring traceability of experiments.
- **Data Integrity**:
  - Check for missing values, shape mismatches, and maintain consistency across data inputs using unit tests and assertions.

**Best Practice Tip:** Automate and version-control data pipelines using DVC to backtrack and reproduce results reliably.

## Testing & Quality Assurance
- **Unit & Integration Testing**:
  - Write comprehensive unit tests for utility functions (data preprocessing, augmentation, etc.) using Pytest.
  - Validate model outputs (e.g., predictions and confidence scores) against pre-defined bounds.
- **Code Quality Checks**:
  - Enforce coding standards by incorporating automatic linting (Ruff, Pylint) and static type checking (Mypy) in your CI/CD pipelines.

**Best Practice Tip:** Integrate continuous integration (CI) to automatically run tests and quality checks on each commit to maintain robust, error-free code.

## Operational Guidelines
- **Environment Setup**:
  - Follow containerization best practices using Docker to ensure that development, testing, and production environments remain consistent. Verify host configurations (e.g., display settings in Linux) as described in the README.
- **Dependency Management**:
  - Use `pyproject.toml` (with Hatchling as the build backend) to manage project dependencies consistently.
- **Configuration**:
  - Externalize configuration settings, such as dataset paths and user-specific settings, in environment variables or dedicated configuration files instead of embedding them in code.

## Copilot Guidance
Copilot should assist by:
- Generating clean, modular PyTorch model definitions and training loop templates.
- Suggesting integration points for MLflow logging (capturing hyperparameters, metrics, and artifacts) during experiments.
- Helping structure DVC pipelines to handle data versioning and reproducibility seamlessly.
- Producing routines for data augmentation and preprocessing, leveraging libraries such as NumPy, pandas, and OpenCV.
- Assisting in writing robust helper functions for dataset loading, visualization, and model evaluation.
- Generating Dockerfile and docker-compose templates that reflect environment variables and container setup best practices.
- Providing suggestions and boilerplate code for comprehensive testing using Pytest, along with guidelines to integrate linting and type-checking tools.
