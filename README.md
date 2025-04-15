# Physics-Informed Machine Learning for Metal AM

This project uses physics-informed machine learning models and Vision Transformers for segmentation of additive manufacturing data. 

## Setup Instructions

### Environment Variables

Create a `.env` file in `.devcontainer` with these variables:

```bash
# Get the dataset root from environment variable or use a default
HOST_DATASET_PATH=/path/to/your/dataset
```
NB:
- On MacOS, typically UID=501 and GID=20
- On Linux, typically UID=1000 and GID=1000
- On Windows, typically UID=1000 and GID=1000 (if using WSL)

You can find these by running `id -u` and `id -g` in the terminal (WSL terminal on Windows). 
