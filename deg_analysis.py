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
from scipy.ndimage import binary_dilation, median_filter
from skimage.restoration import denoise_tv_chambolle
from scipy.signal import peak_widths, find_peaks, peak_prominences
import warnings
warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
warnings.filterwarnings("ignore", message="some peaks have a width of 0")


def get_cycb_widths(group: pd.DataFrame) -> float:
    """
    Calculates CycB peak width. Returns 10000.0 if no peak is found.
    """
    trace = group['cycb_intensity'].values
    
    # Handle empty, all-NaN, or constant traces to avoid PeakPropertyWarning
    if len(trace) < 3 or np.all(np.isnan(trace)) or np.ptp(trace) == 0:
        return 10000.0
        
    peak_pos = np.nanargmax(trace)

    widths, _, _, _ = peak_widths(trace, [peak_pos], rel_height=0.25)

    if int(widths[0]) <= 3:
        return 10000.0
    else:
        return widths[0]


def aggregate_clean_dfs(
    paths: list[str], 
    datewell_keep: Optional[list[str]]=None,
    pos_avoid: Optional[list[str]]=None,
    filters: Optional[list[str]]=None,
    return_failed: bool = False
    ) -> tuple[pd.DataFrame, ...]:

    '''
    Aggregates chromatin.xlsx dataframes
    --------------------------------------
    INPUTS:
        paths: list[str], list of paths to chromatin.xlsx files
        datewell_keep: Optional[list[str]], datewells to aggregate
        pos_avoid: Optional[list[str]], positions to avoid
        filters: Optional[list[str]], QC filters to apply
        return_failed: bool, if True, returns passing and failing dfs
    OUTPUTS:
        If return_failed is False: (df_agg_c, df_agg_qc)
        If return_failed is True:  (df_agg_c, df_agg_qc, df_failed_c, df_failed_qc)
    '''

    dfs = []
    qc_dfs = []
    failed_qc_dfs = []

    for f in paths:
        date_match = re.search(r"20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])", str(f))
        well_match = re.search(r"[A-H]([1-9]|0[1-9]|1[0-2])_s(\d{1,2})", str(f))
        
        if not (date_match and well_match):
            continue
            
        date, well = date_match.group(), well_match.group()

        if datewell_keep and not any(stub in date + well[0] for stub in datewell_keep):
            continue
        if pos_avoid and any(stub in (date + '_' + well) for stub in pos_avoid):
            continue

        print(f"Aggregating {date}, {well}")

        df = pd.read_excel(f)
        df_qc = pd.read_excel(f, sheet_name=1)

        # Basic ID tagging
        for d in [df, df_qc]:
            d["date"], d["well"] = date, well
            d["date-well"] = d["date"] + d["well"].str[0]

        # Feature Calculations
        
        # 1. Mitosis Exit Check
        ends_in_mitosis = (
            df.sort_values(["cell_id", "frame"])
            .groupby("cell_id")["semantic_smoothed"]
            .last().eq(1)
        )
        df_qc["ends_in_mitosis"] = df_qc["cell_id"].map(ends_in_mitosis).fillna(False)

        # 2. cellapp-start: Cyclin B Peak Sync Check
        # Does the global CycB max occur while semantic_smoothed == 1?
        # This guards against Cell-APP starting mitosis late
        idx_max_cycb = df.dropna(subset=['cycb_intensity']).groupby('cell_id')['cycb_intensity'].idxmax()
        cycb_max_is_mitotic = df.loc[idx_max_cycb].set_index('cell_id')['semantic_smoothed'].eq(1)
        df_qc["cycb_max_in_mitosis"] = df_qc["cell_id"].map(cycb_max_is_mitotic).fillna(False)

        # 3. cellapp-end: Cyclin B Decay Width Check (The Scipy Integration)
        # Calculate the 'physical' width of the CycB peak per cell
        # this guards against Cell-APP ending mitosis early
        cycb_widths = (
            df.groupby('cell_id')[['cycb_intensity']] # Explicitly select column
            .apply(get_cycb_widths, include_groups=False) # Silence deprecation warning
        )
        df_qc["cycb_decay_width"] = df_qc["cell_id"].map(cycb_widths).fillna(0.0)


        QC_RULES = {
            "num_dead_flags": lambda d: d["num_dead_flags"] <= 5,
            "plate_gsk": lambda d: d["plate_removal_freq"] >= 0.5,
            "plate_noc": lambda d: d["plate_removal_freq"] < 0.5,
            "range_criterion": lambda d: d["range_criterion"].astype(int) == 1,
            "exited_mitosis": lambda d: ~d["ends_in_mitosis"],
            "time_in_mitosis_noc": lambda d: d["time_in_mitosis"] >= 40,
            "cellapp_start": lambda d: d["cycb_max_in_mitosis"],
            "cellapp_end": lambda d: (d["time_in_mitosis"] / d["cycb_decay_width"]) >= 0.75
        }

        qc_mask = pd.Series(True, index=df_qc.index)
        if filters:
            for f_name in filters:
                if f_name in QC_RULES:
                    qc_mask &= QC_RULES[f_name](df_qc)
                else:
                    raise ValueError(f"Unknown QC filter: {f_name}")

        # Store results
        dfs.append(df)
        qc_dfs.append(df_qc.loc[qc_mask].copy())
        failed_qc_dfs.append(df_qc.loc[~qc_mask].copy())

    df_agg = pd.concat(dfs, ignore_index=True)
    df_agg_qc = pd.concat(qc_dfs, ignore_index=True)
    df_failed_qc = pd.concat(failed_qc_dfs, ignore_index=True)

    failed_index = pd.MultiIndex.from_frame(df_failed_qc[["cell_id", "date", "well"]])
    agg_index = pd.MultiIndex.from_frame(df_agg[["cell_id", "date", "well"]])

    df_agg_c = df_agg[~agg_index.isin(failed_index)]
    
    if return_failed:
        df_failed_c = df_agg[agg_index.isin(failed_index)]
        return df_agg_c, df_agg_qc, df_failed_c, df_failed_qc
        
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
    if dilate:
        mask = group["semantic_contig"].values.astype(bool)
        mask = binary_dilation(mask, iterations=20)
        active_phase = group[mask]
    else:
        active_phase = group[group["semantic_contig"] == 1]

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

def area_jump_statistics(
    group: pd.DataFrame,
    min_height: float = 1,
) -> pd.Series:

    """
    Wrapper to handle pandas groups and return frames corresponding to area collapse dynamics
    -------------------------------------------------------------------------------------------
    For each cell, defines a search window spanning from 2× mitotic duration before mitotic onset
    to the midpoint of mitosis, then identifies the largest negative excursion in area derivative
    (i.e., the steepest drop in area). The dominant peak is selected based on absolute height
    (not prominence), and the onset/stabilization of the collapse are defined as the points where
    the signal falls to 50% of peak height on the left and right flanks.

    -------------------------------------------------------------------------------------------
    INPUTS:
        group: pd.DataFrame, grouped (by cell) dataframe containing:
            - 'frame': int, time index
            - 'semantic_contig': int/bool, contiguous mitotic annotation (1 = mitosis)
            - 'area_derivative': float, time derivative of cell area
        min_height: float, minimum peak height in -area_derivative required to consider
                    a candidate area collapse event

    OUTPUTS:
        pd.Series containing:
            t_areajump: int, frame corresponding to left 75% height crossing of dominant peak
                        (interpreted as onset of area collapse)
            t_areastable: int, frame corresponding to right 50% height crossing of dominant peak
                          (interpreted as post-collapse stabilization)
            jump_height: float, height of the detected peak in -area_derivative
                         (proxy for magnitude of area collapse)
    """

    index = ["t_areajump", "t_areastable", "jump_height"]

    group = group.sort_values("frame").reset_index(drop=True)

    cycb = group['cycb_smoothed'].to_numpy()
    sem = group["semantic_contig"].to_numpy()
    area_deriv = group["area_derivative"].to_numpy()
    frames = group["frame"].to_numpy()

    # --- identify mitotic region ---
    mito_idx = np.where(sem == 1)[0]
    if len(mito_idx) == 0:
        return pd.Series([np.nan, np.nan, np.nan], index=index)
    start_mito = mito_idx[0]

    # --- define search window ---
    start_idx = max(0, start_mito - 25)
    end_idx = np.argmax(cycb)

    if end_idx <= start_idx:
        return pd.Series([np.nan, np.nan, np.nan], index=index)

    # invert to detect drops as peaks
    window_signal = -area_deriv[start_idx:end_idx]

    # --- peak detection (height-based) ---
    peaks, props = find_peaks(window_signal, height=min_height)
    if len(peaks) == 0:
        return pd.Series([np.nan, np.nan, np.nan], index=index)

    heights = props["peak_heights"]

    # --- select tallest peak ---
    best = np.argmax(heights)
    peak_idx = peaks[best]

    # --- compute 75% height crossings ---
    widths, width_heights, left_ips, right_ips = peak_widths(
        window_signal,
        [peak_idx],
        rel_height=0.75 #X% of the way down
    )

    left_ip = left_ips[0]
    right_ip = right_ips[0]

    # --- map back to global indices ---
    global_left = int(np.floor(start_idx + left_ip))
    global_right = int(np.ceil(start_idx + right_ip))

    return pd.Series(
        [
            frames[global_left],
            frames[global_right],
            heights[best],
        ],
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
    mitotic_region = group[group["semantic_contig"] == 1]
    
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
        (df_merged["frame"] < df_merged["cp_frame"])
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


def compute_semantic_contig(df: pd.DataFrame) -> pd.Series:
    """
    Identifies the largest contiguous block of semantic_smoothed == 1 and
    returns a binary indicator marking membership in that block.
    -----------------------------------------------------------------------------
    INPUTS:
        df: pd.DataFrame
            Must contain:
                - 'semantic_smoothed' (binary or boolean-like)
                - sorted or unsorted by frame (function handles sorting implicitly
                  if needed upstream; assumes row order corresponds to time or
                  has consistent indexing within a cell group)
    OUTPUTS:
        pd.Series (int)
            Binary series of same length as df:
                1 → frame belongs to the largest contiguous semantic == 1 block
                0 → all other frames
            If no semantic == 1 region exists, returns all zeros.
    """

    mask = df["semantic_smoothed"] == 1

    if not mask.any():
        return pd.Series(0, index=df.index)

    block_id = (mask != mask.shift()).cumsum()

    blocks = df[mask].groupby(block_id[mask])

    if len(blocks) == 0:
        return pd.Series(0, index=df.index)

    largest_block = max(blocks, key=lambda x: len(x[1]))[1]

    contig = pd.Series(0, index=df.index)
    contig.loc[largest_block.index] = 1

    return contig


def synth_rate_statistics(group, df_agg_qc, min_seg=3):
    """
    Estimate Cyclin B synthesis/degradation rates using smoothed
    derivative measurements ('cycb_deg_rate') in matched windows
    before and after NEB.

    Strategy:
    1. Detect likely NEB timing from the largest positive jump in
       Cyclin B intensity near t_areajump.

    2. Define matched windows:
           back  = pre-NEB
           front = post-NEB

       while excluding frames immediately surrounding NEB.

    3. Compute mean derivative-based rates within each window.
    ------------------------------------------------------------------
    REQUIRED COLUMNS:
        group:
            - frame
            - cycb_smoothed
            - cycb_deg_rate

        df_agg_qc:
            - t_areajump
            - max_cycb_frame

    RETURNS:
        pd.Series containing:
            t_nucbrk
            ksynth_back
            ksynth_front
            std_back
            std_front
            back_start
            back_end
            front_start
            front_end
            back_window_size
            front_window_size
    """

    return_index = [
        "t_nucbrk",
        "ksynth_back",
        "ksynth_front",
        "std_back",
        "std_front",
        "back_start",
        "back_end",
        "front_start",
        "front_end",
        "back_window_size",
        "front_window_size"
    ]

    nan_return = pd.Series(
        [np.nan] * len(return_index),
        index=return_index
    )

    group = group.sort_values("frame")

    # Lookup QC metrics
    key_cols = ["cell_id", "date", "well"]
    key = tuple(group.iloc[0][key_cols])

    qc = df_agg_qc.set_index(key_cols).loc[key]

    t_areajump = qc["t_areajump"]
    t_max = qc["max_cycb_frame"]

    if np.isnan(t_areajump) or np.isnan(t_max):
        return nan_return

    frames = group["frame"].to_numpy()
    y = group["cycb_smoothed"].to_numpy()

    # Detect NEB-associated influx jump
    search_mask = (
        (frames >= t_areajump - 10) &
        (frames <= t_areajump + 10)
    )

    if np.sum(search_mask) < min_seg:
        return nan_return

    search_frames = frames[search_mask]
    search_y = y[search_mask]

    diffs = np.diff(search_y)

    if len(diffs) == 0:
        return nan_return

    # largest positive jump = likely NEB-associated influx
    jump_idx = np.argmax(diffs)
    t_nucbrk = search_frames[jump_idx + 1]

    # Define post-NEB window
    front_start = t_nucbrk + 2

    front_len = int(
        (t_max - front_start) / 4
    )

    front_end = front_start + front_len

    front_mask = (
        (frames >= front_start) &
        (frames <= front_end)
    )

    # Define matched pre-NEB window
    back_end = t_nucbrk - 4
    back_start = back_end - front_len

    back_mask = (
        (frames >= back_start) &
        (frames <= back_end)
    )

    # Ensure enough frames
    if (
        np.sum(front_mask) < min_seg or
        np.sum(back_mask) < min_seg
    ):
        return nan_return

    # Extract derivative traces
    rate_front = group.loc[
        front_mask,
        "cycb_deg_rate"
    ].to_numpy()

    rate_back = group.loc[
        back_mask,
        "cycb_deg_rate"
    ].to_numpy()

    # remove NaNs/infs
    rate_front = rate_front[np.isfinite(rate_front)]
    rate_back = rate_back[np.isfinite(rate_back)]

    if (
        len(rate_front) < min_seg or
        len(rate_back) < min_seg
    ):
        return nan_return

    ksynth_front = -1*np.mean(rate_front)
    ksynth_back = -1*np.mean(rate_back)

    std_front = -1*np.std(rate_front)
    std_back = -1*np.std(rate_back)


    return pd.Series(
        [
            t_nucbrk,
            ksynth_back,
            ksynth_front,
            std_back,
            std_front,
            back_start,
            back_end,
            front_start,
            front_end,
            len(rate_back),
            len(rate_front)
        ],
        index=return_index
    )


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

    to_dilate = False if not noc else True

    df_agg["cycb_smoothed"] = (
        df_agg
        .groupby(["cell_id", "date", "well"], sort=False)["cycb_intensity"]
        .transform(lambda x: denoise_tv_chambolle(x, weight = 5))
    )

    df_agg["cycb_deg_rate"] = (
        df_agg
        .groupby(["cell_id", "date", "well"], sort=False)["cycb_smoothed"]
        .transform(lambda x: -1*np.gradient(x))
    )

    df_agg["area_smoothed"] = (
        df_agg
        .groupby(["cell_id", "date", "well"], sort=False)["cell_area"]
        .transform(lambda x: denoise_tv_chambolle(x*0.3387**2, weight = 50))
    )

    df_agg["area_derivative"] = (
        df_agg
        .groupby(["cell_id", "date", "well"], sort=False)["area_smoothed"]
        .transform(lambda x: np.gradient(x))
    )

    df_agg["semantic_contig"] = (
        df_agg
        .groupby(["cell_id", "date", "well"], sort=False)
        .apply(compute_semantic_contig)
        .reset_index(level=[0,1,2], drop=True)
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

    jump_info = (
        df_agg.groupby(["date", "well", "cell_id"])
        .apply(area_jump_statistics, include_groups=False)
        .reset_index()
    )
    df_agg_qc = df_agg_qc.merge(jump_info, on=["date", "well", "cell_id"], how="left")

    synth_info = (
        df_agg.groupby(["date", "well", "cell_id"])
        .apply(lambda g: synth_rate_statistics(g, df_agg_qc))
        .reset_index()
    )
    df_agg_qc = df_agg_qc.merge(synth_info, on=["date", "well", "cell_id"], how="left")


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
