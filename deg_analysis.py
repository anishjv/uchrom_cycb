import pandas as pd
import numpy as np
from typing import Optional, List
import re
import h5py
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('/Users/whoisv/')
from uchrom_cycb.changept import deriv_changept
from scipy.ndimage import binary_dilation
from skimage.restoration import denoise_tv_chambolle

def aggregate_clean_dfs(
    paths: list[str], 
    datewell_keep: Optional[list[str]]=None,
    pos_avoid: Optional[list[str]]=None,
    filters: Optional[list[str]]=None
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
        df_qc["date-well"] = df_qc["date"] + df_qc["well"].str[0]
        dfs.append(df)

        #TODO: integrate exited mitosis flag in degradation.py
        ends_in_mitosis = (
            df
            .sort_values(["cell_id", "frame"])
            .groupby("cell_id")["semantic_smoothed"]
            .last()
            .eq(1)
        )

        df_qc["ends_in_mitosis"] = df_qc["cell_id"].map(ends_in_mitosis).fillna(False)

        QC_RULES = {
            "num_dead_flags": lambda d: d["num_dead_flags"] <= 5,
            "plate_gsk": lambda d: d["plate_removal_freq"] >= 0.5,
            "plate_noc": lambda d: d["plate_removal_freq"] < 0.5,
            "range_criterion": lambda d: d["range_criterion"].astype(int) == 1,
            "exited_mitosis": lambda d: ~d["ends_in_mitosis"],
            "time_in_mitosis_noc": lambda d: d["time_in_mitosis"] >= 60 #this is 60*4 240 minutes; emperically determined
        }

        qc_mask = pd.Series(True, index=df_qc.index)

        if filters:
            for f_name in filters:
                try:
                    qc_mask &= QC_RULES[f_name](df_qc)
                except KeyError:
                    raise ValueError(f"Unknown QC filter: {f_name}")

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


def cp_statistics(
    group:pd.Grouper, 
    min_prominence:float=0.25, 
    rel_height:float=0.9, 
    dilate:bool=False
    ) -> pd.Series:

    """
    Wrapper to handle pandas groups and return 'frame' of the changepoint and CycB at changept
    NOTE: params are correct for non time-corrected d/dt traces, units = [intensity/4min]
    -------------------------------------------------------------------------------------------
    INPUTS:
        group: pd.Grouper, grouped (by cell) df_agg from aggregate_clean_dfs()
        min_prominence: float, minumum prominence to be considered a peak
        rel_height: float, percentage of peak height to traverse down to be considered the peak's "base"
        dilate: bool, whether or not to extent semantic smoothed, guards against Cell-APP ending mitosis early
    OUTPUTS:
        pd.Series containing:
            cp_frame: int, frame of fast phase onset
            cp_intensity: float, Cyclin B intensity at frame of fast phase onset
    """

    index = ["cp_frame", "cp_intensity", "fast_phs_deg_rate"]

    group = group.sort_values("frame")

    mask = group["semantic_smoothed"].fillna(0).astype(bool)
    if dilate:
        mask = pd.Series(
            binary_dilation(mask.values, iterations=20),
            index=group.index
        )

    active_phase = group.loc[mask]

    if active_phase.empty or "cycb_deg_rate" not in active_phase:
        return pd.Series([np.nan, np.nan, np.nan], index=index)

    deriv = active_phase["cycb_deg_rate"].to_numpy()

    cp_idx, peak_idx = deriv_changept(
        deriv,
        min_prominence=min_prominence,
        rel_height=rel_height
    )

    if cp_idx is None or peak_idx is None:
        return pd.Series([np.nan, np.nan, np.nan], index=index)

    cp_row = active_phase.iloc[cp_idx]
    peak_value = deriv[peak_idx]

    return pd.Series(
        [cp_row["frame"], cp_row["cycb_intensity"], peak_value],
        index=index
    )


def max_cycb_statistics(group: pd.DataFrame) -> pd.Series:
    """
    Wrapper to handle pandas groups and return 'frame' of the max CycB and CycB at max frame
    -------------------------------------------------------------------------------------------
    INPUTS:
        group: pd.Grouper, grouped (by cell) df_agg from aggregate_clean_dfs()
    OUTPUTS:
        pd.Series containing:
            max_cycb_frame: int, frame of maximum Cyclin B
            max_cycb_intensity: float, Cyclin B at max_cycb_frame
    """

    # Isolate mitotic region
    group = group.sort_values("frame")
    mitotic_region = group[group["semantic_smoothed"] == 1]
    
    if mitotic_region.empty:
        return pd.Series([np.nan, np.nan], index=["max_cycb_frame", "max_cycb_intensity"])

    # Find the row with the maximum intensity
    # .idxmax() returns the index of the first occurrence of the maximum
    max_idx = mitotic_region["cycb_intensity"].idxmax()
    max_row = mitotic_region.loc[max_idx]

    return pd.Series(
        [max_row["frame"], max_row["cycb_intensity"]], 
        index=["max_cycb_frame", "max_cycb_intensity"]
    )


def deg_rate_statistics(group: pd.DataFrame) -> pd.Series:

    """
    Computes the mean and variance of the Cyclin B degradation rate 
    specifically during the identified degradation window (deg_semantic == 1).
    -------------------------------------------------------------------------------------------
    INPUTS:
        group: pd.DataFrame, a grouped dataframe (per cell) from df_agg
    OUTPUTS:
        pd.Series containing:
            avg_deg_rate: float, mean of cycb_deg_rate during deg_semantic phase
            var_deg_rate: float, variance of cycb_deg_rate during deg_semantic phase
    """

    # 1. Isolate the degradation region
    deg_region = group[group["deg_semantic"] == 1]
    
    # Define the index once to keep it DRY (Don't Repeat Yourself)
    stat_index = ["avg_deg_rate", "var_deg_rate", "min_deg_rate", "max_deg_rate", "range_deg_rate"]

    # 2. Handle cases where no degradation window was identified
    # Added a third np.nan to match the new range_rate column
    if deg_region.empty or deg_region["cycb_deg_rate"].isna().all():
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=stat_index)

    # 3. Calculate statistics
    avg_rate = deg_region["cycb_deg_rate"].mean()
    var_rate = deg_region["cycb_deg_rate"].var()
    max_rate = deg_region["cycb_deg_rate"].max()
    min_rate = deg_region["cycb_deg_rate"].min()
    range_rate = max_rate - min_rate

    # Return all three
    return pd.Series([avg_rate, var_rate, min_rate, max_rate, range_rate], index=stat_index)


def add_deg_semantic(df_agg: pd.DataFrame, df_agg_qc: pd.DataFrame) -> pd.DataFrame:
    """
    Creates  'deg_semantic' and 'deg_time' columns in df_agg based on max Cyclin B and changepoint frames.
    deg_semantic == 1 for (max_cycb_frame < frame <= cp_frame)
    -------------------------------------------------------------------------------------------
    INPUTS:
        df_agg: pd.DataFrame, per-timepoint output of aggregate_clean_dfs()
        df_agg_qc: pd.DataFrame, per-cell output of aggregate_clean_dfs()
    OUTPUTS:
        pd.Series containing:
            max_cycb_frame: int, frame of maximum Cyclin B
            max_cycb_intensity: float, Cyclin B at max_cycb_frame
    """

    cell_keys = ["date", "well", "cell_id"]
    bounds = df_agg_qc[cell_keys + ["max_cycb_frame", "cp_frame"]]

    # 1. Merge bounds
    df_merged = df_agg.merge(bounds, on=cell_keys, how="left")

    # 2. Identify the degradation phase (semantic == 1)
    df_merged["deg_semantic"] = (
        (df_merged["frame"] > df_merged["max_cycb_frame"]) & 
        (df_merged["frame"] <= df_merged["cp_frame"])
    ).astype(int)

    # 3. Calculate absolute time in phase (0 to N-1)
    mask = df_merged["deg_semantic"] == 1
    df_merged["deg_time"] = df_merged[mask].groupby(cell_keys).cumcount()
    df_merged["deg_time"] = df_merged["deg_time"].fillna(0)

    # 4. Calculate relative time in phase (0.0 to 1.0)
    # We find the max deg_time for each cell and divide
    group_max = df_merged.groupby(cell_keys)["deg_time"].transform("max")
    
    # We use .where to avoid division by zero if a phase is only 1 frame long
    df_merged["deg_time_rel"] = (df_merged["deg_time"] / group_max).fillna(0)

    # 5. Clean up
    return df_merged.drop(columns=["max_cycb_frame", "cp_frame"])


def add_addl_metrics(
    df_agg: pd.DataFrame, df_agg_qc: pd.DataFrame, noc:Optional[bool]=False
    ) ->tuple[pd.DataFrame]:

    """
    Function to add extra metrics to chromatin.xlsx dataframes
    -----------------------------------------------------------
    INPUTS:
        df_agg: pd.DataFrame, per-timepoint output of aggregate_clean_dfs()
        df_agg_qc: pd.DataFrame, per-cell output of aggregate_clean_dfs()
    """

    denoise_weight = 5 if not noc else 51
    to_dilate = False if not noc else True

    df_agg["cycb_smoothed"] = (
    df_agg
    .groupby(["cell_id", "date", "well"], sort=False)["cycb_intensity"]
    .transform(lambda x: denoise_tv_chambolle(x, weight = denoise_weight))
    )


    df_agg["cycb_deg_rate"] = (
        df_agg
        .groupby(["cell_id", "date", "well"], sort=False)["cycb_smoothed"]
        .transform(lambda x: -1*np.gradient(x))
    )

    # 1. Get Peak Info
    max_info = (
        df_agg.groupby(["date", "well", "cell_id"])
        .apply(max_cycb_statistics, include_groups=False)
        .reset_index()
    )
    df_agg_qc = df_agg_qc.merge(max_info, on=["date", "well", "cell_id"], how="left")

    # 2. Get Changepoint Info (using your existing calculate_cp)
    cp_info = (
        df_agg.groupby(["date", "well", "cell_id"])
        .apply(cp_statistics, dilate = to_dilate, include_groups=False)
        .reset_index()
    )
    df_agg_qc = df_agg_qc.merge(cp_info, on=["date", "well", "cell_id"], how="left")

    # 3. Generate the actual trace in the time-series data
    df_agg = add_deg_semantic(df_agg, df_agg_qc)

    # 4. Generate degradation rate info
    deg_info = (
        df_agg.groupby(["date", "well", "cell_id"])
        .apply(deg_rate_statistics, include_groups=False)
        .reset_index()
    )
    df_agg_qc = df_agg_qc.merge(deg_info, on=["date", "well", "cell_id"], how="left")

    return df_agg, df_agg_qc

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
