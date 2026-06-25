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
from scipy.ndimage import median_filter
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

        # Smooth before QC criteria so validate_cyclin_b_trace operates on smoothed trace
        df["cycb_smoothed"] = (
            df.groupby("cell_id", sort=False)["cycb_intensity"]
            .transform(lambda x: adaptive_bilateral_1d(
                x, sigma_min=1.0, sigma_range_min=3.0, min_int=10, gain=1.0, read_noise_var=4.0
            ))
        )

        # Feature Calculations
        criteria = (
            df.sort_values(["cell_id", "frame"])
            .groupby("cell_id")
            .apply(
                lambda g: pd.Series(
                    validate_cyclin_b_trace(
                        trace=g["cycb_smoothed"].values,
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
        "fast_phs_end_frame",
        "fast_phs_end_intensity",
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
        peak_intensity = peak_row["cycb_smoothed"]
        peak_value = deriv[int(peak_idx)]

    if pd.isna(ap_idx):
        fast_phs_end_frame = np.nan
        fast_phs_end_intensity = np.nan
    else:
        ap_row = active_phase.iloc[int(ap_idx)]
        fast_phs_end_frame = ap_row["frame"]
        fast_phs_end_intensity = ap_row["cycb_smoothed"]

    if pd.isna(cp_idx):
        cp_frame = np.nan
        cp_intensity = np.nan
    else:
        cp_row = active_phase.iloc[int(cp_idx)]
        cp_frame = cp_row["frame"]
        cp_intensity = cp_row["cycb_smoothed"]

    return pd.Series(
        [
            cp_frame,
            cp_intensity,
            fast_phs_end_frame,
            fast_phs_end_intensity,
            peak_frame,
            peak_intensity,
            peak_value,
        ],
        index=index,
    )


def _find_dominant_drop(
    signal: np.ndarray,
    min_height: float = 1.0,
) -> Optional[tuple[int, float]]:
    """
    In a windowed -area_derivative signal, find the tallest peak.
    --------------------------------------------------------------
    INPUTS:
        signal: np.ndarray, windowed inverted area derivative (-area_derivative)
        min_height: float, minimum peak height to qualify
    OUTPUTS:
        Optional[tuple[int, float]]:
            (peak_idx, peak_height) local to signal; None if no qualifying peak found
    """
    peaks, props = find_peaks(signal, height=min_height)
    if len(peaks) == 0:
        return None
    best = int(np.argmax(props["peak_heights"]))
    return int(peaks[best]), float(props["peak_heights"][best])


def area_jump_statistics(
    group: pd.DataFrame,
    min_height: float = 1,
) -> pd.Series:

    """
    Wrapper to handle pandas groups and return frames corresponding to area collapse dynamics
    -------------------------------------------------------------------------------------------
    Identifies two distinct area drop events in -area_derivative:

    1. Rounding-up peak: steepest area drop as the cell rounds up entering mitosis.
       Searched from 25 frames before mitotic onset to the CycB peak.

    2. Division peak: sharp area drop at anaphase as tracking switches from mother to
       daughter cell. Searched from max(CycB peak, end_mito - 10) to
       min(trace end, end_mito + 10).

    Both peaks use identical base logic: left and right bases are the 75% height
    crossings from peak_widths(rel_height=0.75).

    -------------------------------------------------------------------------------------------
    INPUTS:
        group: pd.DataFrame, grouped (by cell) dataframe containing:
            - 'frame': int, time index
            - 'semantic_contig': int/bool, contiguous mitotic annotation (1 = mitosis)
            - 'cycb_smoothed': float, smoothed Cyclin B signal
            - 'area_derivative': float, time derivative of cell area
        min_height: float, minimum peak height in -area_derivative required to consider
                    a candidate area collapse event

    OUTPUTS:
        pd.Series containing:
            roundup_start_frame: int, left 75% height crossing of the rounding-up peak
            roundup_end_frame: int, right 75% height crossing of the rounding-up peak
            roundup_frame: int, frame of the rounding-up peak
            anaphase_start_frame: int, left 75% height crossing of the division peak
            anaphase_end_frame: int, right 75% height crossing of the division peak
            anaphase_frame: int, frame of the division peak
    """

    index = [
        "roundup_start_frame", "roundup_end_frame", "roundup_frame",
        "anaphase_start_frame", "anaphase_end_frame", "anaphase_frame",
    ]
    nan_series = pd.Series([np.nan] * 6, index=index)

    group = group.sort_values("frame").reset_index(drop=True)

    cycb = group['cycb_smoothed'].to_numpy()
    sem = group["semantic_contig"].to_numpy()
    area_deriv = group["area_derivative"].to_numpy()
    frames = group["frame"].to_numpy()

    mito_idx = np.where(sem == 1)[0]
    if len(mito_idx) == 0:
        return nan_series

    start_mito = mito_idx[0]
    end_mito = mito_idx[-1]
    cycb_peak_idx = int(np.argmax(cycb))

    # ── rounding-up peak ────────────────────────────────────────────────────
    roundup_start = roundup_end = roundup_frame = np.nan

    round_start = max(0, start_mito - 25)
    round_end = cycb_peak_idx

    if round_end > round_start:
        window = -area_deriv[round_start:round_end]
        result = _find_dominant_drop(window, min_height)
        if result is not None:
            peak_idx, _ = result
            _, _, left_ips, right_ips = peak_widths(window, [peak_idx], rel_height=0.75)
            roundup_start = frames[round_start + int(np.floor(left_ips[0]))]
            roundup_frame = frames[round_start + peak_idx]
            roundup_end = frames[round_start + int(np.ceil(right_ips[0]))]

    # ── division peak (anaphase) ─────────────────────────────────────────────
    anaphase_start = anaphase_end = anaphase_frame = np.nan

    ana_start = max(cycb_peak_idx, end_mito - 10)
    ana_end = min(len(frames) - 1, end_mito + 10)

    if ana_end > ana_start:
        window = -area_deriv[ana_start : ana_end + 1]
        result = _find_dominant_drop(window, min_height)
        if result is not None:
            peak_idx, _ = result
            _, _, left_ips, right_ips = peak_widths(window, [peak_idx], rel_height=0.75)
            anaphase_start = frames[ana_start + int(np.floor(left_ips[0]))]
            anaphase_frame = frames[ana_start + peak_idx]
            anaphase_end = frames[ana_start + int(np.ceil(right_ips[0]))]

    return pd.Series(
        [roundup_start, roundup_end, roundup_frame,
         anaphase_start, anaphase_end, anaphase_frame],
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
    max_idx = mitotic_region["cycb_smoothed"].idxmax()
    max_row = mitotic_region.loc[max_idx]

    return pd.Series(
        [max_row["frame"], max_row["cycb_smoothed"]],
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
       Cyclin B intensity near roundup_start_frame.

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
            - roundup_start_frame
            - max_cycb_frame

    RETURNS:
        pd.Series containing:
            roundup_frame_refined
            ksynth_back
            ksynth_front
            ksynth_std_back
            ksynth_std_front
            ksynth_back_start
            ksynth_back_end
            ksynth_front_start
            ksynth_front_end
    """

    return_index = [
        "roundup_frame_refined",
        "ksynth_back",
        "ksynth_front",
        "ksynth_std_back",
        "ksynth_std_front",
        "ksynth_back_start",
        "ksynth_back_end",
        "ksynth_front_start",
        "ksynth_front_end"
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

    t_roundup_start = qc["roundup_start_frame"]
    t_max = qc["max_cycb_frame"]

    if np.isnan(t_roundup_start) or np.isnan(t_max):
        return nan_return

    frames = group["frame"].to_numpy()
    y = group["cycb_smoothed"].to_numpy()

    # Detect NEB-associated influx jump
    search_mask = (
        (frames >= t_roundup_start - 10) &
        (frames <= t_roundup_start + 10)
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
    roundup_frame_refined = search_frames[jump_idx + 1]

    # Define post-NEB window
    ksynth_front_start = roundup_frame_refined + 2

    front_len = int(
        (t_max - ksynth_front_start) / 3
    )

    ksynth_front_end = ksynth_front_start + front_len

    front_mask = (
        (frames >= ksynth_front_start) &
        (frames <= ksynth_front_end)
    )

    # Define matched pre-NEB window
    ksynth_back_end = roundup_frame_refined - 4
    ksynth_back_start = ksynth_back_end - front_len

    back_mask = (
        (frames >= ksynth_back_start) &
        (frames <= ksynth_back_end)
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

    ksynth_std_front = -1 * np.std(rate_front)
    ksynth_std_back = -1 * np.std(rate_back)

    return pd.Series(
        [
            roundup_frame_refined,
            ksynth_back,
            ksynth_front,
            ksynth_std_back,
            ksynth_std_front,
            ksynth_back_start,
            ksynth_back_end,
            ksynth_front_start,
            ksynth_front_end
        ],
        index=return_index
    )


def adaptive_bilateral_1d(x, sigma_min=1.0, sigma_range_min=3.0, min_int=10, gain=1.0, read_noise_var=4.0):
    """
    Smooths a 1D array using a bilateral filter whose spatial and range sigmas
    both scale with local shot noise variance. The range sigma prevents cross-edge
    averaging at sharp transitions while remaining permissive enough at high
    intensities where true shot noise is large.
    -----------------------------------------------------------------------------
    INPUTS:
        x              : npt.NDArray, 1D fluorescence intensity array
        sigma_min      : float, minimum spatial sigma (at min_int). Default 1.0.
        sigma_range_min: float, range sigma at min_int — sets edge sensitivity at
                         the slow→fast transition. Default 3.0.
        min_int        : float, intensity defining the transition zone. Default 10.
        gain           : float, camera counts per photoelectron. Default 1.0.
        read_noise_var : float, readout noise variance. Default 4.0.
    OUTPUTS:
        smoothed : npt.NDArray, smoothed trace, same shape as x.
    """
    x = np.asarray(x)
    n = len(x)
    if n <= 1:
        return x.copy()

    pilot         = median_filter(x, size=min(3, n))
    pilot_clipped = np.maximum(0, pilot)

    var_pilot = (gain * pilot_clipped) + read_noise_var
    var_min   = (gain * min_int) + read_noise_var

    noise_ratio = var_pilot / var_min

    sigmas       = np.maximum(sigma_min,       sigma_min       * noise_ratio)
    sigmas       = np.minimum(max(sigma_min, n / 24), sigmas)
    sigma_ranges = np.maximum(sigma_range_min, sigma_range_min * noise_ratio)

    smoothed = np.zeros_like(x, dtype=float)

    for i in range(n):
        sigma       = sigmas[i]
        sigma_range = sigma_ranges[i]
        radius      = int(np.ceil(4 * sigma))
        start       = max(0, i - radius)
        end         = min(n, i + radius + 1)

        t          = np.arange(start, end) - i
        spatial_w  = np.exp(-t**2 / (2 * sigma**2))
        range_w    = np.exp(-(x[start:end] - x[i])**2 / (2 * sigma_range**2))
        weights    = spatial_w * range_w

        weights_sum = np.sum(weights)
        weights     = weights / weights_sum if weights_sum > 0 else np.ones_like(weights) / len(weights)

        smoothed[i] = np.sum(x[start:end] * weights)

    return smoothed


def floor_cycb_statistics(group: pd.DataFrame, anaphase_end_frame, window: int = 30) -> pd.Series:
    """
    Find the Cyclin B floor after anaphase end
    -------------------------------------------
    INPUTS:
        group: pd.DataFrame, per-cell rows containing 'frame' and 'cycb_smoothed'
        anaphase_end_frame: int or float, frame marking the end of the anaphase area drop
        window: int, number of frames after anaphase_end_frame to search for the minimum
    OUTPUTS:
        pd.Series containing:
            floor_cycb_frame: int, frame of minimum cyclin B in the search window
            floor_cycb_intensity: float, smoothed cyclin B value at that frame
    """
    index = ["floor_cycb_frame", "floor_cycb_intensity"]
    nan_series = pd.Series([np.nan, np.nan], index=index)

    if pd.isna(anaphase_end_frame):
        return nan_series

    group = group.sort_values("frame")
    mask = (group["frame"] >= anaphase_end_frame) & (group["frame"] <= anaphase_end_frame + window)
    window_df = group[mask]

    if window_df.empty:
        return nan_series

    min_idx = window_df["cycb_smoothed"].idxmin()
    return pd.Series(
        [window_df.loc[min_idx, "frame"], window_df.loc[min_idx, "cycb_smoothed"]],
        index=index,
    )


def add_addl_metrics(
    df_agg: pd.DataFrame, df_agg_qc: pd.DataFrame, noc: Optional[bool] = False
) -> tuple[pd.DataFrame]:

    """
    Function to add extra metrics to chromatin.xlsx dataframes
    -----------------------------------------------------------
    INPUTS:
        df_agg: pd.DataFrame, per-timepoint output of aggregate_clean_dfs()
        df_agg_qc: pd.DataFrame, per-cell output of aggregate_clean_dfs()
    """

    cp_remediate = False if not noc else True

    if "cycb_smoothed" not in df_agg.columns:
        df_agg["cycb_smoothed"] = (
            df_agg
            .groupby(["cell_id", "date", "well"], sort=False)["cycb_intensity"]
            .transform(lambda x: adaptive_bilateral_1d(
                x, sigma_min=1.0, sigma_range_min=3.0, min_int=10, gain=1.0, read_noise_var=4.0
            ))
        )

    df_agg["cycb_deg_rate"] = (
        df_agg
        .groupby(["cell_id", "date", "well"], sort=False)["cycb_smoothed"]
        .transform(lambda x: -1*np.gradient(x))
    )

    df_agg["area_smoothed"] = (
        df_agg
        .groupby(["cell_id", "date", "well"], sort=False)["cell_area"]
        .transform(lambda x: adaptive_bilateral_1d(
                x*0.3387**2, sigma_min=1.0, sigma_range_min=3.0, min_int=10, gain=1.0, read_noise_var=4.0
            ))
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

    # 5. Get Cyclin B floor after anaphase
    qc_indexed = df_agg_qc.set_index(["date", "well", "cell_id"])
    floor_info = (
        df_agg.groupby(["date", "well", "cell_id"])
        .apply(
            lambda g: floor_cycb_statistics(
                g,
                qc_indexed.loc[
                    (g.name[0], g.name[1], g.name[2]), "anaphase_end_frame"
                ] if g.name in qc_indexed.index else np.nan,
            ),
            include_groups=False,
        )
        .reset_index()
    )
    df_agg_qc = df_agg_qc.merge(floor_info, on=["date", "well", "cell_id"], how="left")

    synth_info = (
        df_agg.groupby(["date", "well", "cell_id"])
        .apply(lambda g: synth_rate_statistics(g, df_agg_qc))
        .reset_index()
    )
    df_agg_qc = df_agg_qc.merge(synth_info, on=["date", "well", "cell_id"], how="left")


    return df_agg, df_agg_qc

def save_chromatin_sources(
    cell: np.ndarray,
    cell_mask: np.ndarray,
    labeled_chromosomes: Optional[np.ndarray] = None,
    removal_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Package the four source arrays for one cell timepoint as a dict of
    minimally-typed arrays ready to write as named HDF5 datasets.
    ---------------------------------------------------------------------------------------------------------------
    Stores sources rather than a rendered composite so that the display is
    lossless and re-analysis (e.g. filter_plate_bleedout) can run directly
    off the HDF5 without recomputing from the raw movies. The composite is
    fully recoverable from these arrays via render_chromatin_composite().
    cell is stored as float16 (sufficient for [0, 1] data and half the size
    of float32). Masks are stored as bool (gzip compresses sparse boolean
    arrays very efficiently). labeled is stored as int16 (ample for any
    realistic number of objects in a single crop).
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray, float grayscale crop normalized to [0, 1]
        cell_mask: np.ndarray, boolean cell boundary mask
        labeled_chromosomes: Optional[np.ndarray], int labeled unaligned
                             chromosome regions; None for non-mitotic frames
        removal_mask: Optional[np.ndarray], boolean metaphase plate mask;
                      None for non-mitotic frames
    OUTPUTS:
        src: dict mapping dataset name -> array at minimal dtype:
             "cell"         -> float16 (always present)
             "cell_mask"    -> bool    (always present)
             "labeled"      -> int16   (present iff labeled_chromosomes is not None)
             "removal_mask" -> bool    (present iff removal_mask is not None)
    """
    src = {
        "cell":      cell.astype(np.float16),
        "cell_mask": cell_mask.astype(bool),
    }
    if labeled_chromosomes is not None:
        src["labeled"] = labeled_chromosomes.astype(np.int16)
    if removal_mask is not None:
        src["removal_mask"] = removal_mask.astype(bool)
    return src


def render_chromatin_composite(
    cell: np.ndarray,
    cell_mask: np.ndarray,
    labeled_chromosomes: Optional[np.ndarray] = None,
    removal_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct the RGBA composite from source arrays saved by
    save_chromatin_sources.
    ---------------------------------------------------------------------------------------------------------------
    This is a pure rendering function: it applies the same sequence of
    alpha-blended overlays that save_chromatin_crops used to produce, so
    the output is pixel-identical to what save_chromatin_crops would have
    returned for the same inputs. Kept separate from save_chromatin_sources
    so that storage and display concerns do not mix.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray, float grayscale crop in [0, 1] (may be float16
              as loaded from HDF5; imshow handles both)
        cell_mask: np.ndarray, boolean cell boundary mask
        labeled_chromosomes: Optional[np.ndarray], int labeled unaligned
                             chromosome regions; None for non-mitotic frames
        removal_mask: Optional[np.ndarray], boolean metaphase plate mask;
                      None for non-mitotic frames
    OUTPUTS:
        composite: np.ndarray (H, W, 4) float32 RGBA, identical to what
                   the old save_chromatin_crops would have returned
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
            [0.357, 0.498, 0.749, 0.6],  # #5B7FBF — steel blue
            [0.302, 0.659, 0.627, 0.6],  # #4DA8A0 — teal
            [0.337, 0.706, 0.263, 0.6],  # #56B443 — mid green  ← your existing green
            [0.482, 0.408, 0.784, 0.6],  # #7B68C8 — periwinkle
            [0.690, 0.376, 0.565, 0.6],  # #B06090 — mauve
            [0.290, 0.565, 0.769, 0.6],  # #4A90C4 — sky blue
            [0.427, 0.722, 0.478, 0.6],  # #6DB87A — sage green
            [0.608, 0.447, 0.722, 0.6],  # #9B72B8 — soft violet
            [0.353, 0.659, 0.753, 0.6],  # #5AA8C0 — cerulean
            [0.769, 0.478, 0.667, 0.6],  # #C47AAA — dusty pink
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
    center: Optional[int] = None,
    display_data: Optional[list[list]] = None,
    overlays: bool = True,
):
    """
    Visualize chromatin time-series from an HDF5 cell stack, organized in chunks.

    Each HDF5 file contains cell groups (cell_0, cell_1, ...) with named source
    datasets per timepoint (all frames, not just mitotic ones). Every group has
    'cell' (float16) and 'cell_mask' (bool); mitotic frames also have 'labeled'
    (int16) and 'removal_mask' (bool). The composite is rendered at display time
    via render_chromatin_composite, so the HDF5 is lossless and directly usable
    for re-analysis.

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

        center : Optional[int], default=None
            Frame number to center the display window around. When provided,
            displays `max_cells` timepoints centered on the group whose frame
            number is closest to `center`.

        display_data : Optional[list[list]], default=None
            Additional data lists to show on the x-axis label alongside the frame
            number. Each list must have the same length as `frames`. Float values
            are rounded to the nearest integer. The x-axis label for each timepoint
            will read: frame, val1, val2, ...

        overlays : bool, default=True
            If True, renders the full composite (cell boundary, chromosome labels,
            metaphase plate) via render_chromatin_composite. If False, displays
            only the raw grayscale 'cell' array — faster because it skips reading
            cell_mask, labeled, and removal_mask from HDF5 and skips all blending.

    -------------------------------------------------------------------------------------------
    OUTPUTS:
        figs : list[matplotlib.figure.Figure]
            List of figure handles, one per chunk.

    SIDE EFFECTS:
        Displays figures interactively via plt.show().
    """

    if display_data is not None:
        for i, lst in enumerate(display_data):
            if len(lst) != len(frames):
                raise ValueError(
                    f"display_data[{i}] has length {len(lst)} but frames has length {len(frames)}."
                )

    def _fmt(val):
        if isinstance(val, float):
            return str(round(val))
        return str(val)

    with h5py.File(file_path, "r") as f:
        all_groups = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))

        if center is not None:
            center_pos = min(
                range(len(all_groups)),
                key=lambda i: abs(frames[int(all_groups[i].split("_")[-1])] - center),
            )
            half = max_cells // 2
            lo = max(0, center_pos - half)
            hi = min(len(all_groups), lo + max_cells)
            lo = max(0, hi - max_cells)
            cell_groups = all_groups[lo:hi]
        else:
            cell_groups = all_groups[:max_cells]
        n_cells = len(cell_groups)
        n_chunks = math.ceil(n_cells / chunk_size)

        figs = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_cells)
            chunk_groups = cell_groups[start:end]
            n_cols = len(chunk_groups)

            fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3), squeeze=False, dpi=48)

            for col, name in enumerate(chunk_groups):
                grp = f[name]
                if overlays:
                    cell      = grp["cell"][()].astype(np.float32)
                    cell_mask = grp["cell_mask"][()].astype(bool)
                    labeled   = grp["labeled"][()].astype(np.int16)  if "labeled"      in grp else None
                    removal   = grp["removal_mask"][()].astype(bool) if "removal_mask" in grp else None
                    img = render_chromatin_composite(cell, cell_mask, labeled, removal)
                else:
                    img = grp["cell"][()].astype(np.float32)

                ax = axes[0, col]
                ax.imshow(img, cmap="gray" if not overlays else None)
                ax.tick_params(left=False, bottom=False, labelleft=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                group_idx = int(name.split("_")[-1])

                label_parts = [str(frames[group_idx])]
                if display_data is not None:
                    label_parts += [_fmt(lst[group_idx]) for lst in display_data]
                ax.set_xlabel(", ".join(label_parts), fontsize=12)
                ax.xaxis.set_label_position("bottom")

            plt.tight_layout()
            #plt.show()
            figs.append(fig)

        return figs
