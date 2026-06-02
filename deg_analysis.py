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
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import find_boundaries
from scipy.signal import peak_widths, find_peaks

import warnings
warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
warnings.filterwarnings("ignore", message="some peaks have a width of 0")


def validate_cyclin_b_trace(
    trace: np.ndarray,
    semantic: np.ndarray,
):
    """
    Validate whether a trace exhibits expected Cyclin B dynamics around the
    semantic == 1 interval.
    ---------------------------------------------------------------------------
    INPUTS:
        trace:
            np.ndarray of Cyclin B intensities over time.

        semantic:
            np.ndarray of semantic state labels over time.
            semantic == 1 is assumed to form one contiguous block of frames.

    OUTPUTS:
        peaks_criterion:
            True if the maximum Cyclin B intensity occurs within the
            semantic == 1 block.

        range_criterion:
            True if max(trace) - min(trace) > 10.

        hysteresis_criterion:
            True if the mean Cyclin B intensity in the 1-3 frames BEFORE the
            semantic == 1 block is greater than the mean intensity in the
            1-3 frames AFTER the semantic == 1 block.

            Returns False if insufficient frames exist on either side.
    """

    semantic_1_idx = np.where(semantic == 1)[0]

    # semantic==1 block must exist
    if len(semantic_1_idx) == 0:
        return False, False, False

    start = semantic_1_idx[0]
    end = semantic_1_idx[-1]

    # Criterion 1:
    # Max Cyclin B occurs during semantic==1 block
    max_frame = np.argmax(trace)
    peaks_criterion = start <= max_frame <= end

    # Criterion 2:
    # Trace dynamic range > 10
    trace_range = np.max(trace) - np.min(trace)
    range_criterion = trace_range > 10

    # Criterion 3:
    # Mean before semantic==1 block > mean after block
    n = len(trace)

    pre_start = max(start - 16, 0)
    pre_end = max(start - 10, 0)

    post_start = min(end + 1, n)
    post_end = min(end + 6, n)

    valid_windows = (
        (pre_end-pre_start) > 2 and
        (post_end-post_start) > 2
    )

    if valid_windows:
        pre_vals = trace[pre_start:pre_end]
        post_vals = trace[post_start:post_end]

        hysteresis_criterion = np.mean(pre_vals) > np.mean(post_vals)
    else:
        hysteresis_criterion = False

    return peaks_criterion, range_criterion, hysteresis_criterion

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
        
        #TODO: rerun deg so that these columns don't exist
        # Remove old columns first
        cols_to_replace = [
            "peaks_criterion",
            "range_criterion",
            "hysteresis_criterion",
        ]

        df_qc = df_qc.drop(
            columns=[c for c in cols_to_replace if c in df_qc.columns]
        )

        # Basic ID tagging
        for d in [df, df_qc]:
            d["date"], d["well"] = date, well
            d["date-well"] = d["date"] + d["well"].str[0]

        # Compute largest contiguous mitotic region
        df["semantic_contig"] = (
            df
            .groupby(["cell_id"], sort=False)
            .apply(compute_semantic_contig, include_groups=False)
            .reset_index(level=0, drop=True)
        )

        # Feature Calculations
        criteria = (
            df.sort_values(["cell_id", "frame"])
            .groupby("cell_id")
            .apply(
                lambda g: pd.Series(
                    validate_cyclin_b_trace(
                        trace=g["cycb_intensity"].values,
                        semantic=g["semantic_contig"].values,
                    ),
                    index=[
                        "peaks_criterion",
                        "range_criterion",
                        "hysteresis_criterion"
                    ],
                ), include_groups=False
            )
            .reset_index()
        )

        # Merge criteria into df_qc
        df_qc = df_qc.merge(criteria, on="cell_id", how="left")

        QC_RULES = {
            "num_dead_flags": lambda d: d["num_dead_flags"] <= 5,
            "plate_gsk": lambda d: d["plate_removal_freq"] >= 0.5,
            "plate_noc": lambda d: d["plate_removal_freq"] < 0.5,
            "range_criterion": lambda d: d["range_criterion"].astype(int) == 1,
            "exited_mitosis": lambda d: d["ends_in_mitosis"].astype(int) == 0,
            "time_in_mitosis_noc": lambda d: d["time_in_mitosis"] >= 40,
            "cellapp_start": lambda d: d["peaks_criterion"].astype(int) == 1,
            "cellapp_end": lambda d: d["hysteresis_criterion"].astype(int) == 1,
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
    group: pd.DataFrame,
    min_prominence: float = 0.25,
    rel_height: float = 0.9,
    cp_remediate: bool = False,
) -> pd.Series:

    index = [
        "cp_frame",
        "cp_intensity",
        "ap_frame",
        "ap_intensity",
        "fast_phs_frame",
        "fast_phs_intensity",
        "fast_phs_deg_rate",
    ]

    nan_series = pd.Series([np.nan] * len(index), index=index)

    group = group.sort_values("frame").reset_index(drop=True)

    sem_mask = group["semantic_contig"].values.astype(bool)
    sem_indices = np.flatnonzero(sem_mask)

    if len(sem_indices) == 0:
        return nan_series

    start_idx = sem_indices[0]
    end_idx = sem_indices[-1]

    half_idx = (start_idx + end_idx) // 2

    # expand search window to prevent semantic cutoffs ruining detection
    search_end = min(end_idx + 25, len(group) - 1)

    active_phase = group.iloc[half_idx : search_end + 1]

    if active_phase.empty or "cycb_deg_rate" not in active_phase:
        return nan_series

    deriv = active_phase["cycb_deg_rate"].to_numpy()

    cp_idx, peak_idx, ap_idx = deriv_changept(
        deriv, min_prominence=min_prominence, rel_height=rel_height
    )

    if cp_idx is None or peak_idx is None or ap_idx is None:
        return nan_series

    # Only attempt changepoint remediation if dilation is enabled
    if cp_remediate:
        # Check if the 5 frames preceding cp_idx have a negative mean degradation rate
        # If so, cell likely died
        start_lookback = max(0, cp_idx - 5)

        if (
            cp_idx > 0
            and np.mean(deriv[start_lookback:cp_idx]) < 0
        ):

            # Find the "nearest" index *before* cp_idx where deriv > 0
            prior_positive_indices = np.flatnonzero(deriv[:cp_idx] > 0)

            if len(prior_positive_indices) >= 3:
                cp_idx = prior_positive_indices[-3]

            else:
                cp_idx = np.nan

            # Set peak and anaphase indices to NaN as the cell died
            peak_idx = np.nan
            ap_idx = np.nan

    # Safely handle extraction for ap and peak rows since they could now be NaN
    if pd.isna(peak_idx):
        peak_frame = np.nan
        peak_intensity = np.nan
        peak_value = np.nan
    else:
        peak_row = active_phase.iloc[int(peak_idx)]
        peak_frame = peak_row["frame"]
        peak_intensity = peak_row["cycb_intensity"]
        peak_value = deriv[int(peak_idx)]

    if pd.isna(ap_idx):
        ap_frame = np.nan
        ap_intensity = np.nan
    else:
        ap_row = active_phase.iloc[int(ap_idx)]
        ap_frame = ap_row["frame"]
        ap_intensity = ap_row["cycb_intensity"]

    if pd.isna(cp_idx):
        cp_frame = np.nan
        cp_intensity = np.nan
    else:
        cp_row = active_phase.iloc[int(cp_idx)]
        cp_frame = cp_row["frame"]
        cp_intensity = cp_row["cycb_intensity"]

    return pd.Series(
        [
            cp_frame,
            cp_intensity,
            ap_frame,
            ap_intensity,
            peak_frame,
            peak_intensity,
            peak_value,
        ],
        index=index,
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
    (not prominence), and the onset of the collapse is defined as the point where
    the signal reaches 75% of peak height on the left flank.

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
            jump_height: float, height of the detected peak in -area_derivative
                         (proxy for magnitude of area collapse)
    """

    index = ["t_areajump", "jump_height"]

    group = group.sort_values("frame").reset_index(drop=True)

    cycb = group['cycb_smoothed'].to_numpy()
    sem = group["semantic_contig"].to_numpy()
    area_deriv = group["area_derivative"].to_numpy()
    frames = group["frame"].to_numpy()

    # --- identify mitotic region ---
    mito_idx = np.where(sem == 1)[0]
    if len(mito_idx) == 0:
        return pd.Series([np.nan, np.nan], index=index)

    start_mito = mito_idx[0]

    # --- define search window ---
    start_idx = max(0, start_mito - 25)
    end_idx = np.argmax(cycb)

    if end_idx <= start_idx:
        return pd.Series([np.nan, np.nan], index=index)

    # invert to detect drops as peaks
    window_signal = -area_deriv[start_idx:end_idx]

    # --- peak detection (height-based) ---
    peaks, props = find_peaks(window_signal, height=min_height)

    if len(peaks) == 0:
        return pd.Series([np.nan, np.nan], index=index)

    heights = props["peak_heights"]

    # --- select tallest peak ---
    best = np.argmax(heights)
    peak_idx = peaks[best]

    # --- compute 75% height crossings ---
    widths, width_heights, left_ips, right_ips = peak_widths(
        window_signal,
        [peak_idx],
        rel_height=0.75
    )

    left_ip = left_ips[0]

    # --- map back to global indices ---
    global_left = int(np.floor(start_idx + left_ip))

    return pd.Series(
        [
            frames[global_left],
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
    stat_index = ["avg_deg_rate", "var_deg_rate", "min_deg_rate", "max_deg_rate"]

    # 2. Handle cases where no degradation window was identified
    # Added a third np.nan to match the new range_rate column
    if deg_region.empty or deg_region["cycb_deg_rate"].isna().all():
        return pd.Series([np.nan, np.nan, np.nan, np.nan], index=stat_index)

    # 3. Calculate statistics
    avg_rate = deg_region["cycb_deg_rate"].mean()
    var_rate = deg_region["cycb_deg_rate"].var()
    max_rate = deg_region["cycb_deg_rate"].max()
    min_rate = deg_region["cycb_deg_rate"].min()

    # Return all three
    return pd.Series([avg_rate, var_rate, min_rate, max_rate], index=stat_index)


def add_deg_semantic(df_agg: pd.DataFrame, df_agg_qc: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 'deg_semantic', 'deg_time', and 'deg_time_rel' columns.

    Standard case:
        deg_semantic == 1 for:
            max_cycb_frame < frame < cp_frame

    If ends_in_mitosis == 1:
        deg_semantic == 1 for:
            frame > max_cycb_frame through end of trace
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

    bounds = df_agg_qc[
        cell_keys + ["max_cycb_frame", "cp_frame", "ends_in_mitosis"]
    ]

    # 1. Merge metadata
    df_merged = df_agg.merge(bounds, on=cell_keys, how="left")

    # 2. Define degradation phase
    normal_mask = (
        (df_merged["frame"] > df_merged["max_cycb_frame"]) &
        (df_merged["frame"] < df_merged["cp_frame"])
    )

    mitotic_end_mask = (
        (df_merged["ends_in_mitosis"] == 1) &
        (df_merged["frame"] > df_merged["max_cycb_frame"])
    )

    df_merged["deg_semantic"] = (
        normal_mask | mitotic_end_mask
    ).astype(int)

    # 3. Absolute degradation time
    mask = df_merged["deg_semantic"] == 1

    df_merged["deg_time"] = np.nan
    df_merged.loc[mask, "deg_time"] = (
        df_merged.loc[mask]
        .groupby(cell_keys)
        .cumcount()
    )

    df_merged["deg_time"] = df_merged["deg_time"].fillna(0)

    # 4. Relative degradation time
    group_max = (
        df_merged
        .groupby(cell_keys)["deg_time"]
        .transform("max")
    )

    df_merged["deg_time_rel"] = (
        df_merged["deg_time"] / group_max.where(group_max > 0)
    ).fillna(0)

    # 5. Cleanup
    return df_merged.drop(
        columns=["max_cycb_frame", "cp_frame", "ends_in_mitosis"]
    )


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
        "front_end"
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
        (t_max - front_start) / 3
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

    ksynth_front = -1 * np.mean(rate_front)
    ksynth_back = -1 * np.mean(rate_back)

    std_front = -1 * np.std(rate_front)
    std_back = -1 * np.std(rate_back)

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
            front_end
        ],
        index=return_index
    )


def add_addl_metrics(
    df_agg: pd.DataFrame, df_agg_qc: pd.DataFrame, noc:Optional[bool]=False, type:Optional[str]="none"
    ) ->tuple[pd.DataFrame]:

    """
    Function to add extra metrics to chromatin.xlsx dataframes
    -----------------------------------------------------------
    INPUTS:
        df_agg: pd.DataFrame, per-timepoint output of aggregate_clean_dfs()
        df_agg_qc: pd.DataFrame, per-cell output of aggregate_clean_dfs()
    """

    cp_remediate = False if not noc else True
    match type:
        case "rpe":
            weight = 10
        case "oe":
            weight = 100
        case "none":
            weight = 5

    df_agg["cycb_smoothed"] = (
        df_agg
        .groupby(["cell_id", "date", "well"], sort=False)["cycb_intensity"]
        .transform(lambda x: denoise_tv_chambolle(x, weight = weight))
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
        .apply(cp_statistics, cp_remediate = cp_remediate, include_groups=False)
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
    cell_mask: np.ndarray,
    labeled_chromosomes: Optional[np.ndarray] = None,
    removal_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a single composited RGBA crop for one cell timepoint.
    ---------------------------------------------------------------------------
    INPUTS:
        cell:
            np.ndarray, float32 grayscale cell image in [0, 1].

        cell_mask:
            np.ndarray, boolean cell boundary mask.

        labeled_chromosomes:
            Optional np.ndarray, labeled unaligned chromosome regions (mitotic
            frames only). Pass None for non-mitotic frames.

        removal_mask:
            Optional np.ndarray, boolean mask covering the metaphase plate
            (mitotic frames only). Pass None for non-mitotic frames.

    OUTPUTS:
        composite:
            np.ndarray (H, W, 4) float32 RGBA composed of:
                - grayscale base
                - metaphase plate in transparent gray (if removal_mask given)
                - per-chromosome colored overlays (if labeled_chromosomes given)
                - cell boundary as a white outline
    """

    H, W = cell.shape

    def blend_overlay(base_rgba, overlay_rgba):
        alpha = overlay_rgba[..., 3:4]
        blended = base_rgba.copy()
        blended[..., :3] = (1 - alpha) * base_rgba[..., :3] + alpha * overlay_rgba[..., :3]
        blended[..., 3] = np.clip(base_rgba[..., 3] + alpha[..., 0], 0, 1)
        return blended

    # Grayscale base as RGBA (cell is already normalized to [0, 1])
    composite = np.zeros((H, W, 4), dtype=float)
    composite[..., :3] = cell[..., np.newaxis]
    composite[..., 3] = 1.0

    # Metaphase plate: transparent gray
    if removal_mask is not None:
        removal_rgba = np.zeros((H, W, 4), dtype=float)
        removal_rgba[removal_mask.astype(bool)] = [0.6, 0.6, 0.6, 0.3]
        composite = blend_overlay(composite, removal_rgba)

    # Unaligned chromosomes overlay
    if labeled_chromosomes is not None:
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
            composite = blend_overlay(composite, chrom_rgba)

    # Cell boundary as white outline
    border = find_boundaries(cell_mask.astype(bool), mode='inner')
    border_rgba = np.zeros((H, W, 4), dtype=float)
    border_rgba[border] = [1, 1, 1, 0.8]
    composite = blend_overlay(composite, border_rgba)

    return composite.astype(np.float32)


def visualize_chromatin_hd5(
    file_path: str,
    frames: list[int],
    max_cells: int = 60,
    chunk_size: int = 12,
):
    """
    Visualize chromatin time-series from an HDF5 cell stack, organized in chunks.

    Each HDF5 file contains cell groups (cell_0, cell_1, ...) with one composited
    RGBA image per timepoint (all frames, not just mitotic ones). Each group has a
    single dataset '0' containing the composited image.

    Cells are displayed in groups of `chunk_size` columns per figure, one row per
    chunk.

    -------------------------------------------------------------------------------------------
    INPUTS:
        file_path : str
            Path to HDF5 file containing cell image stacks.

        frames : list[int]
            Frame numbers for all timepoints stored in the file (one per cell group).

        max_cells : int, default=60
            Maximum number of timepoints to visualize from the file.

        chunk_size : int, default=12
            Number of timepoints displayed per figure.

    -------------------------------------------------------------------------------------------
    OUTPUTS:
        figs : list[matplotlib.figure.Figure]
            List of figure handles, one per chunk.

    SIDE EFFECTS:
        Displays figures interactively via plt.show().
    """

    with h5py.File(file_path, "r") as f:
        cell_groups = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
        n_cells = min(len(cell_groups), max_cells)
        n_chunks = math.ceil(n_cells / chunk_size)

        figs = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_cells)
            chunk_groups = cell_groups[start:end]
            n_cols = len(chunk_groups)

            fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3), squeeze=False)

            for col, name in enumerate(chunk_groups):
                grp = f[name]
                composite = grp["0"][()]

                ax = axes[0, col]
                ax.imshow(composite)
                ax.tick_params(left=False, bottom=False, labelleft=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xlabel(frames[start + col], fontsize=12)
                ax.xaxis.set_label_position("bottom")

            plt.tight_layout()
            plt.show()
            figs.append(fig)

        return figs
