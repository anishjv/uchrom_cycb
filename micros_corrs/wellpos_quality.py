import re 
import os
import glob
import tifffile as tif
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

def quality_control_plotting(img_dict: dict, cmap: Optional[str] = "gray"):

    '''
    Function to plot slices from IXN movies - helps
    for checking if any wells lost focus during imaging
    ----------------------------------------------------
    INPUTS:
        img_dict: dict, dictionary containing images to plot
        cmap: str, tells matplotlib how to color the images
    '''

    n_rows = len(img_dict)
    n_cols = 3

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        squeeze=False
    )

    for row, (key, stack) in enumerate(sorted(img_dict.items())):

        for col in range(3):
            ax = axes[row, col]
            ax.imshow(stack[col], cmap=cmap)

        axes[row, 0].set_ylabel(key)

    plt.tight_layout()
    return fig


def main():

    root_dir = Path("/nfs/turbo/umms-ajitj/anishjv/cyclinb_analysis/20251028-cycb-gsk")
    chromatin_dir = root_dir

    stub_re = re.compile(r"[A-H](?:[1-9]|0[1-9]|1[0-2])_s\d{1,2}")

    inference_dirs = [
        obj.path for obj in os.scandir(root_dir)
        if "_inference" in obj.name and obj.is_dir()
    ]

    print(f"Found {len(inference_dirs)} inference directories.")

    all_chromatin_files = list(chromatin_dir.glob("*Texas Red.tif"))

    img_dict = {}
    for dir_path in sorted(inference_dirs):

        match = stub_re.search(str(dir_path))
        if not match:
            continue

        name_stub = match.group()
        prefix = f"{name_stub}_"

        matches = [p for p in all_chromatin_files if prefix in p.name]

        if len(matches) == 0:
            print(f"No chromatin file for {name_stub}")
            continue
        if len(matches) > 1:
            print(f"Multiple chromatin files for {name_stub}")
            continue

        with tif.TiffFile(matches[0]) as tf:
            z = len(tf.pages)
            planes = [1, z // 2, z - 2] #second, middle, second to last
            stack = np.stack([tf.pages[i].asarray() for i in planes])

        img_dict[name_stub] = stack

    fig = quality_control_plotting(img_dict)
    fig.savefig(root_dir / "qc_images.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
    

