"""
Example script demonstrating how to:
1) Load streaking and spatter masks from a Peregrine-style HDF5 dataset
2) Compute coverage per layer
3) Overlay masks on original camera images
4) Plot coverage vs. layer
5) Visualize selected layers
"""
import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from config import get_dataset_path, CLASS_ID_STREAK, CLASS_ID_SPATTER

# Get dataset path using the configuration system
HDF5_PATH = get_dataset_path("tcr_phase1_build2")
if HDF5_PATH is None:
    raise FileNotFoundError("Dataset not found. Please check the configuration and data directory.")

LAYER_TO_SHOW = 50

def main():
    # Open the HDF5 file
    with h5py.File(HDF5_PATH, "r") as build:
        # Extract spatter and recoater streaking anomalies
        spatter = build["slices/segmentation_results/8"][
            LAYER_TO_SHOW, ...
        ]  # Spatter anomalies
        streaking = build["slices/segmentation_results/3"][
            LAYER_TO_SHOW, ...
        ]  # Recoater streaking anomalies

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Spatter anomaly visualization
        axes[0].imshow(spatter, cmap="inferno", interpolation="none")
        axes[0].set_title("Spatter Anomalies")
        axes[0].axis("off")

        # Recoater streaking anomaly visualization
        axes[1].imshow(streaking, cmap="plasma", interpolation="none")
        axes[1].set_title("Recoater Streaking Anomalies")
        axes[1].axis("off")

        # Display the plots
        plt.show()


if __name__ == "__main__":
    main()
