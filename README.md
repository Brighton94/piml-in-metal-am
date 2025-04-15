# Physics-Informed Machine Learning for Metal AM: A MLOps Template for Researchers

ON GOING project: This project uses physics-informed machine learning models and Vision Transformers for segmentation of additive manufacturing data. 

This repo currently serves as a machine learning operations for research projects template. 

## Setup Instructions

### Linux Display Configuration

If you're using Linux and want to run GUI applications inside the container, you need to:

1. Run this command on your host machine before starting the container:
   ```bash
   xhost +local:
	```

2. Verify your DISPLAY environment variable:
   ```bash
   echo $DISPLAY
   ```
   This value should match what's passed to the container in docker-compose.yml.

### Configuration

#### Environment Variables

Create a `.env` file in `.devcontainer` with these variables:

```bash
# Get the dataset root from environment variable or use a default
HOST_DATASET_PATH==/path/to/your/dataset
```
NB:
- On MacOS, typically UID=501 and GID=20
- On Linux, typically UID=1000 and GID=1000
- On Windows, typically UID=1000 and GID=1000 (if using WSL)

You can find these by running `id -u` and `id -g` in the terminal (WSL terminal on Windows). 
