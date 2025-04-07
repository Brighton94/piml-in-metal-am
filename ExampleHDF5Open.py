# Date: 2023-09-04
# Description: example script to load an exported HDF5 file
# -----------------------------------------------------------------------------
# Load external modules
import h5py
import matplotlib.cm as cm  # used for vector graphics
import matplotlib.collections as collections  # used for vector graphics
import matplotlib.colors as mcolors  # used for vector graphics
import matplotlib.pyplot as plt
import numpy as np
from config import get_dataset_path

# Needed to enable plotting if using the Spyder IDE
try:
    from IPython import get_ipython  # needed to run magic commands

    ipython = get_ipython()  # needed to run magic commands
    ipython.magic("matplotlib qt")  # display figures in a separate window
except Exception as e:
    print(f"An error occurred: {e}")

# Parameters
path_file = get_dataset_path("tcr_phase1_build1")
if path_file is None:
    raise FileNotFoundError("Dataset not found. Please check the configuration and data directory.")
probe_layer = 50  # layer number to extract

# -----------------------------------------------------------------------------

# Check the HDF5 file
print("\nChecking the HDF5 contents...\n")
with h5py.File(path_file, "r") as build:

    def print_attrs(name, obj):
        # Access datasets
        if isinstance(build[name], h5py.Dataset):
            # Report dataset path
            print("--", name)

            # Display slice-type data
            if name.split("/")[0] == "slices":
                plt.figure(name)
                plt.imshow(
                    build[name][probe_layer, ...], cmap="jet", interpolation="none"
                )
                plt.show()  # Display the figure

            # Display scan path data
            elif name == "scans/%i" % (probe_layer):
                x = build["scans/%i" % (probe_layer)][:, 0:2]
                y = build["scans/%i" % (probe_layer)][:, 2:4]
                t = build["scans/%i" % (probe_layer)][:, 4]
                colorizer = cm.ScalarMappable(
                    norm=mcolors.Normalize(np.min(t), np.max(t)), cmap="jet"
                )
                line_collection = collections.LineCollection(
                    np.stack([x, y], axis=-1), colors=colorizer.to_rgba(t)
                )
                fig = plt.figure(name)
                ax = fig.add_subplot()
                plt.axis("scaled")
                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(y.min(), y.max())
                ax.add_collection(line_collection)
                plt.show()  # Display the figure

            # Display temporal data
            elif name.split("/")[0] == "temporal":
                plt.figure(name)
                plt.scatter(np.arange(build[name].shape[0]), build[name])
                plt.show()  # Display the figure

            # Display part and sample data
            elif (name.split("/")[0] == "parts") | (name.split("/")[0] == "samples"):
                for i in range(1, min(build[name].shape[0], 10), 1):
                    print("  %i " % (i), build[name][i])

            # Display reference images and micrographs
            elif (name.split("/")[0] == "reference_images") | (
                name.split("/")[0] == "micrographs"
            ):
                plt.figure(name)
                plt.imshow(build[name][...], interpolation="none")
                plt.show()  # Display the figure

        # Access attributes
        for key in obj.attrs:
            print(f"{name}/{key}:", str(build[name].attrs[key]).split("\n")[0])

    # Walk through the HDF5 file and display a sub-set of the data
    print("\n---HDF5 CONTENTS---\n")
    for key in build.attrs:
        print("%s:" % (key), str(build.attrs[key]).split("\n")[0])  # top-level metadata
    build.visititems(print_attrs)  # nested levels
