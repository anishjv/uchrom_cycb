import pandas as pd
import numpy as np
from typing import Optional, List, Union
import re
import h5py
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('/Users/whoisv/')
from uchrom_cycb.changept import changept

def aggregate_clean_dfs(
    paths: list[str], 
    datewell_keep: Optional[list[str]]=None,
    pos_avoid: Optional[list[str]]=None
    ) -> tuple[pd.DataFrame]:

    '''
    Aggregates chromatin.xlsx dataframes
    --------------------------------------
    INPUTS:
        paths: list[str], list of paths to chromatin.xlsx files
        datewell_keep[str], datewells to aggregate
    OUTPUTS:

    '''

    dfs = []
    qc_dfs = []
    failed_records = []

    for f in paths:

        date = re.search(r"20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])", str(f)).group()
        well = re.search(r"[A-H]([1-9]|0[1-9]|1[0-2])_s(\d{1,2})", str(f)).group()

        if datewell_keep: #keep certain wells
            if not any(stub in date + well[0] for stub in datewell_keep):
                continue

        if pos_avoid: #avoid certain positions
            if any(stub in (date + '_' + well) for stub in pos_avoid):
                continue

        print(f"Aggregating {date}, {well}")

        df = pd.read_excel(f)
        df_qc = pd.read_excel(f, sheet_name=1)

        df["date"] = date
        df["well"] = well
        df["date-well"] = df["date"] + df["well"].str[0]

        df_qc["date"] = date
        df_qc["well"] = well
        df_qc["date-well"] = df["date"] + df["well"].str[0]

        dfs.append(df)

        qc_mask = (
            (df_qc["num_dead_flags"] <= 5) &
            (df_qc["plate_removal_freq"] >= 0.5) &
            (df_qc["range_criterion"].astype(int) == 1)
        )

        # keep only PASSING rows
        qc_dfs.append(df_qc.loc[qc_mask].copy())

        failed_rows = df_qc.loc[~qc_mask, ["cell_id"]].copy()
        failed_rows["date"] = date
        failed_rows["well"] = well
        failed_records.append(failed_rows)

    df_agg = pd.concat(dfs, ignore_index=True)
    df_agg_qc = pd.concat(qc_dfs, ignore_index=True)
    df_failed = pd.concat(failed_records, ignore_index=True)

    failed_index = pd.MultiIndex.from_frame(df_failed[["cell_id", "date", "well"]])
    agg_index = pd.MultiIndex.from_frame(df_agg[["cell_id", "date", "well"]])

    df_agg_c = df_agg[~agg_index.isin(failed_index)]

    return df_agg_c, df_agg_qc


def find_slowdeg_regime(group: pd.DataFrame):

    '''
    Appends two rows ('slowdeg_regime', 'regime_time') to the 
    aggregate dataframe from aggregate_clean_dfs()
    ---------------------------------------------------------
    INPUTS:
        group: pd.DataFrame, grouped dataframe
    '''

    group = group.sort_values("frame")

    mask = group["semantic_smoothed"] == 1
    y = group.loc[mask, "cycb_intensity"]

    slowdeg_regime = pd.Series(0, index=group.index, dtype=int)
    regime_time = pd.Series(np.nan, index=group.index, dtype=float)

    if len(y) < 3:
        return pd.DataFrame({
            "slowdeg_regime": slowdeg_regime,
            "regime_time": regime_time
        })

    cp, i_max, i_min = changept(y.to_numpy())

    frames = group.loc[mask, "frame"].to_numpy()

    cp_frame  = frames[cp]
    max_frame = frames[i_max]

    start = min(cp_frame, max_frame)
    end   = max(cp_frame, max_frame)

    in_regime = (
        (group["frame"] >= start) &
        (group["frame"] <= end)
    )

    slowdeg_regime.loc[in_regime] = 1

    # count frames since regime start
    regime_time.loc[in_regime] = group.loc[in_regime, "frame"] - start

    return pd.DataFrame({
        "slowdeg_regime": slowdeg_regime,
        "regime_time": regime_time
    })


def save_chromatin_crops(
    cell: np.ndarray,
    tophat_cell: np.ndarray,
    cell_mask: np.ndarray,
    labeled_chromosomes: np.ndarray,
    removal_mask: np.ndarray,
) -> List[np.ndarray]:
    """
    Create a stack of chromatin crops for a single cell across metaphase timepoints.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray, raw grayscale crop of the cell
        tophat_cell: np.ndarray, preprocessed cell image (unused as base; raw cell is used)
        cell_mask: np.ndarray, binary mask of the cell
        labeled_chromosomes: np.ndarray, labeled chromosome regions
        removal_mask: np.ndarray, mask for removed regions
    OUTPUTS:
        crop_stack: list of np.ndarray containing:
            [0] raw cell (grayscale)
            [1] metaphase plate overlay (RGBA)
            [2] colored overlay of unaligned chromosomes + cell mask (RGBA)
    """

    from uchrom_cycb.segment_chromatin import get_largest_signal_regions

    H, W = cell.shape
    raw_img = cell.copy()  # raw grayscale base

    # Helper function for blending
    def blend_overlay(base_rgba, overlay_rgba):
        """Alpha blend overlay_rgba onto base_rgba."""
        alpha = overlay_rgba[..., 3:4]
        base_rgba[..., :3] = (1 - alpha) * base_rgba[..., :3] + alpha * overlay_rgba[
            ..., :3
        ]
        base_rgba[..., 3] = np.clip(base_rgba[..., 3] + alpha[..., 0], 0, 1)
        return base_rgba

    #  Metaphase overlay
    labeled_regions, max_lbl = get_largest_signal_regions(tophat_cell, cell, cell_mask)
    if max_lbl is not None:
        metaphase_mask = labeled_regions == max_lbl
    metaphase_overlay = np.zeros((H, W, 4), dtype=float)
    if metaphase_mask is not None:
        mask_rgba = np.zeros((H, W, 4), dtype=float)
        mask_rgba[metaphase_mask.astype(bool)] = [0, 0, 1, 0.2]  # blue alpha 0.2
        metaphase_overlay = blend_overlay(metaphase_overlay, mask_rgba)
    if removal_mask is not None:
        removal_rgba = np.zeros((H, W, 4), dtype=float)
        removal_rgba[removal_mask.astype(bool)] = [1, 0, 0, 0.2]  # red alpha 0.2
        metaphase_overlay = blend_overlay(metaphase_overlay, removal_rgba)

    #  Unaligned chromosomes overlay
    unaligned_overlay = np.zeros((H, W, 4), dtype=float)
    colors = [
        [1, 0, 0, 0.2],
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.2],
        [1, 1, 0, 0.2],
        [1, 0, 1, 0.2],
        [0, 1, 1, 0.2],
        [1, 0.5, 0, 0.2],
        [0.5, 0, 1, 0.2],
        [0, 0.5, 0, 0.2],
        [0.5, 0.5, 0, 0.2],
    ]

    unique_labels = np.unique(labeled_chromosomes[labeled_chromosomes > 0])
    for i, lbl in enumerate(unique_labels):
        mask = labeled_chromosomes == lbl
        chrom_rgba = np.zeros((H, W, 4), dtype=float)
        chrom_rgba[mask] = colors[i % len(colors)]
        unaligned_overlay = blend_overlay(unaligned_overlay, chrom_rgba)

    # Overlay faint white cell outline last
    cell_rgba = np.zeros((H, W, 4), dtype=float)
    cell_rgba[cell_mask.astype(bool)] = [1, 1, 1, 0.1]
    unaligned_overlay = blend_overlay(unaligned_overlay, cell_rgba)

    return [raw_img, metaphase_overlay, unaligned_overlay]


def visualize_chromatin_hd5(
    file_path: str,
    frames: list[int],
    num_frames_deg: int | None = None,
    max_cells: int = 20,
    chunk_size: int = 6,
):
    """
    Visualize chromatin crops from an HDF5 file, chunked for easier viewing.

    Displays:
      Row 0: raw grayscale images
      Row 1: metaphase overlay
      Row 2: unaligned overlay + column number labels

    Each figure shows up to `chunk_size` cells (columns).
    """
    with h5py.File(file_path, "r") as f:
        cell_groups = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
        num_images = len(cell_groups)

        if num_frames_deg:
            offset = (
                num_frames_deg - num_images
            )  # all mitotic timepoints get an image, but my degradation measurements only from the CycB peak onwards (a few timepoints into mitosis)
            print(
                f"Saved stacks contain {num_images} cells; "
                f"only {num_frames_deg} were valid degradation frames. Adjusting indices accordingly."
            )

        n_cells = min(len(cell_groups), max_cells)
        n_chunks = math.ceil(n_cells / chunk_size)

        figs = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_cells)
            chunk_groups = cell_groups[start:end]
            n_cols = len(chunk_groups)

            fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9), squeeze=False)

            for col, name in enumerate(chunk_groups):
                grp = f[name]
                raw_img = grp["0"][()].astype(float) / 255
                metaphase_overlay = grp["1"][()]
                unaligned_overlay = grp["2"][()]

                # Scale overlay RGB to darken
                meta_disp = metaphase_overlay.copy()
                meta_disp[..., :3] *= 3
                unal_disp = unaligned_overlay.copy()
                unal_disp[..., :3] *= 3

                # Top row: raw
                axes[0, col].imshow(raw_img, cmap="gray")
                axes[0, col].axis("off")

                # Middle row: metaphase overlay
                axes[1, col].imshow(raw_img, cmap="gray")
                axes[1, col].imshow(meta_disp)
                axes[1, col].axis("off")

                # Bottom row: unaligned overlay + label
                ax = axes[2, col]
                ax.imshow(raw_img, cmap="gray")
                ax.imshow(unal_disp)
                ax.tick_params(left=False, bottom=False, labelleft=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                rel_label = start + col + offset
                abs_label = frames[start+col]
                label = f'{rel_label},{abs_label}'
                ax.set_xlabel(
                    label, 
                    fontsize=16
                    )
                ax.xaxis.set_label_position("bottom")

            plt.tight_layout()
            plt.show()  # each chunk displays separately
            figs.append(fig)

        return figs
