# Physics-Informed Machine Learning for Metal AM

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
- On MacOS, typically USER_ID=501 and USER_GID=20
- On Linux, typically USER_ID=1000 and USER_GID=1000
- On Windows, typically USER_ID=1000 and USER_GID=1000 (if using WSL)

You can find these by running `id -u` and `id -g` in the terminal (WSL terminal on Windows)

#### Container Configuration

Modify these files according to your system:

1. In docker-compose.yml:
   - Change the volume mount path to match your external drive location:
     ```yaml
     # Change this line to your external drive location
     - "/path/to/your/external/drive/l-pbf-dataset:/external/l-pbf-dataset:ro"
     ```
   - Adjust USER_UID and USER_GID in the build args if your host UID/GID differs from 1000

2. In devcontainer.json:
   - Modify the "remoteEnv" EXTERNAL_DRIVE_PATH if needed

#### Dataset Paths

The system looks for datasets in multiple locations:
- Primary: `$DATASET_ROOT` (defaults to "/data")
- Secondary: `$EXTERNAL_DRIVE_PATH` (defaults to "/external/l-pbf-dataset")

Currently available dataset keys:
- "tcr_phase1_build1"
- "tcr_phase1_build2"
