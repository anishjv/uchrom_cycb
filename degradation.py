import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile as tiff
from scipy.signal import find_peaks

from typing import Optional
import glob
import re
from segment_chromatin import unaligned_chromatin, ChromatinSegConfig
import sys
import h5py


def resolve_channel_columns(analysis_df: pd.DataFrame, wl: str) -> dict[str, str]:
    """
    Map a channel to the analysis.xlsx columns holding its raw intensity and corrections.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        analysis_df: pd.DataFrame, analysis.xlsx output from `https://github.com/ajitpj/cellapp-analysis`
        wl: str, channel name (e.g. 'GFP', 'Texas Red')
    OUTPUTS:
        columns: dict[str, str], {'intensity', 'bkg', 'shading', 'offset'} -> column name in analysis_df
    RAISES:
        KeyError, if any required column for the channel is missing
    """
    columns = {
        "intensity": f"{wl}",
        "bkg": f"{wl}_bkg_corr",
        "shading": f"{wl}_int_corr",
        "offset": f"{wl}_offset",
    }
    # Files processed before bkg_plotting went multi-channel hold a single GFP-derived 'offset'
    # column; it is only valid for GFP, so other channels must have their own '{wl}_offset'.
    if columns["offset"] not in analysis_df.columns and wl == "GFP":
        columns["offset"] = "offset"

    missing = [col for col in columns.values() if col not in analysis_df.columns]
    if missing:
        raise KeyError(f"channel '{wl}' is missing column(s) {missing}")
    return columns


def output_column_names(wl: str) -> dict[str, str]:
    """
    Output (degradation_data) column names for a channel's per-frame traces.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        wl: str, channel name
    OUTPUTS:
        names: dict[str, str], {'intensity', 'intensity_raw', 'bkg', 'shading', 'offset'} -> output column name
    """
    # GFP keeps the pre-multichannel names because deg_analysis.py and other consumers read them
    if wl == "GFP":
        return {
            "intensity": "cycb_intensity",
            "intensity_raw": "cycb_intensity_raw",
            "bkg": "bkg",
            "shading": "shading",
            "offset": "offset",
        }
    return {key: f"{wl}_{key}" for key in ["intensity", "intensity_raw", "bkg", "shading", "offset"]}


def channel_trace_columns(
    channels: list[str],
    i: int,
    intensity_traces: dict[str, list[npt.NDArray]],
    raw_intensity_traces: dict[str, list[npt.NDArray]],
    bkg_traces: dict[str, list[npt.NDArray]],
    shading_traces: dict[str, list[npt.NDArray]],
    offset_traces: dict[str, list[npt.NDArray]],
) -> tuple[dict[str, npt.NDArray], dict[str, npt.NDArray]]:
    """
    Build the per-channel output columns for the i-th retained cell.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        channels: list[str], channels analyzed
        i: int, index of the cell in the retrieve_traces outputs
        intensity_traces ... offset_traces: dict[str, list[npt.NDArray]], per-channel outputs of retrieve_traces
    OUTPUTS:
        intensity_cols: dict[str, npt.NDArray], {output column: trace} for corrected and raw intensity
        correction_cols: dict[str, npt.NDArray], {output column: trace} for bkg, shading, and offset
    """
    intensity_cols = {}
    correction_cols = {}
    for wl in channels:
        names = output_column_names(wl)
        intensity_cols[names["intensity"]] = intensity_traces[wl][i]
        intensity_cols[names["intensity_raw"]] = raw_intensity_traces[wl][i]
        correction_cols[names["bkg"]] = bkg_traces[wl][i]
        correction_cols[names["shading"]] = shading_traces[wl][i]
        correction_cols[names["offset"]] = offset_traces[wl][i]
    return intensity_cols, correction_cols


def retrieve_traces(
    analysis_df: pd.DataFrame,
    channels: list[str],
    frame_interval: int,
    remove_end_mitosis: Optional[bool] = False,
) -> tuple[dict[str, list[npt.NDArray]], list[npt.NDArray], list, list[npt.NDArray], list, dict[str, list[npt.NDArray]], dict[str, list[npt.NDArray]], dict[str, list[npt.NDArray]], dict[str, list[npt.NDArray]]]:
    """
    Retrieve corrected intensity traces for each channel plus shared semantic traces, enforcing selection rules and experiment length constraints.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        analysis_df: pd.DataFrame, analysis.xlsx output from `https://github.com/ajitpj/cellapp-analysis`
        channels: list[str], channel names to process (e.g. ['GFP', 'Texas Red'])
        frame_interval: int, time between successive frames; sets t_char = 20 // frame_interval
        remove_end_mitosis: bool, if True exclude traces that end mitotic at frame exp_length-1; otherwise allow a single end-plateau
    OUTPUTS:
        intensity_traces: dict[str, list[npt.NDArray]], per channel, corrected intensities ((intensity - bkg) * shading) - offset
        semantic_traces: list[npt.NDArray], unpadded semantic traces
        frame_traces: list[npt.NDArray], actual frame numbers for each trace
        area_traces: list[npt.NDArray], cell area per frame for each trace
        ids: list, particle identifiers retained
        bkg_traces: dict[str, list[npt.NDArray]], per channel, per-frame background correction values ({wl}_bkg_corr)
        shading_traces: dict[str, list[npt.NDArray]], per channel, per-frame shading correction factors ({wl}_int_corr)
        offset_traces: dict[str, list[npt.NDArray]], per channel, per-frame offsets ({wl}_offset, or legacy 'offset' for GFP)
        raw_intensity_traces: dict[str, list[npt.NDArray]], per channel, uncorrected intensities ({wl}) prior to bkg/shading/offset correction
        Per-channel lists are index-aligned with ids.
    RAISES:
        KeyError, if a required column for any channel is missing (see resolve_channel_columns)
    """

    # Resolve up front so a missing column fails before any traces are built
    channel_columns = {wl: resolve_channel_columns(analysis_df, wl) for wl in channels}

    ids = []
    semantic_traces = []
    frame_traces = []
    area_traces = []
    intensity_traces = {wl: [] for wl in channels}
    raw_intensity_traces = {wl: [] for wl in channels}
    bkg_traces = {wl: [] for wl in channels}
    shading_traces = {wl: [] for wl in channels}
    offset_traces = {wl: [] for wl in channels}
    t_char = 20 // frame_interval

    for id in analysis_df["particle"].unique():
        cell_df = analysis_df.query(f"particle=={id}")
        area = cell_df["area"].to_numpy()
        semantic = cell_df["semantic_smoothed"].to_numpy()
        frames = cell_df["frame"].to_numpy()

        # Always remove traces that start in mitosis
        if semantic[0] == 1:
            continue
        # If remove_end_mitosis is True and the trace ends in mitosis at the final frame, skip
        if remove_end_mitosis and semantic[-1] == 1:
            continue

        # Peak detection array: pad to allow end-plateau to be counted as a peak
        semantic_for_detection = np.append(semantic, np.zeros(3))

        # Detect mitosis events; require exactly one event to satisfy
        _, props = find_peaks(semantic_for_detection, width=t_char)
        if props["widths"].size != 1:
            continue

        for wl, cols in channel_columns.items():
            intensity = cell_df[cols["intensity"]].to_numpy()
            bkg = cell_df[cols["bkg"]].to_numpy()
            shading = cell_df[cols["shading"]].to_numpy()
            offset = cell_df[cols["offset"]].to_numpy()

            intensity_traces[wl].append(((intensity - bkg) * shading) - offset)
            raw_intensity_traces[wl].append(intensity)
            bkg_traces[wl].append(bkg)
            shading_traces[wl].append(shading)
            offset_traces[wl].append(offset)

        semantic_traces.append(semantic)
        frame_traces.append(frames)
        area_traces.append(area)
        ids.append(id)

    return intensity_traces, semantic_traces, frame_traces, area_traces, ids, bkg_traces, shading_traces, offset_traces, raw_intensity_traces


def area_model(N, A_max, f, beta):
    "Poisson germ-grain area model with a noise term (beta)"
    return A_max * (1 - (1 - f)**N) + beta


def predict_integer_chromosomes(
    A_obs: float, 
    A_max: float,
    f: float,
    beta: float,
    sigma_0: float,
    gamma:float,
    max_n: int = 75 #pseudo-triploid ~69
    ):
    """
    Defines a probability distribution over possible numbers of unaligned chromosomes 
    given an observed unaligned chromatin area measurement
    """
    n_options = np.arange(0, max_n + 1)
    
    # Calculates predicted means for every possible integer n
    mu_n = area_model(n_options, A_max, f, beta)
    
    # Calculates predicted sigmas for every possible integer n
    sigma_n = sigma_0 * (n_options + 1)**gamma
    
    # Calculate the log-likelihood for each n
    # (Leaving out constants like sqrt(2pi) as they don't change the argmax)
    log_likelihoods = -np.log(sigma_n) - 0.5 * ((A_obs - mu_n) / sigma_n)**2
    
    # Find the most likely integer (the Mode)
    n_best = n_options[np.argmax(log_likelihoods)]
    
    # Calculate a Credible Interval (normalized probabilities)
    probs = np.exp(log_likelihoods - np.max(log_likelihoods)) # Undo the log
    probs /= np.sum(probs) # Normalize
    
    # Cumulative distribution to find the 95% range
    cumulative_prob = np.cumsum(probs)
    n_low = n_options[np.searchsorted(cumulative_prob, 0.16)]
    n_high = n_options[np.searchsorted(cumulative_prob, 0.84)]
    
    return int(n_best), (int(n_low), int(n_high)), probs


def cycb_chromatin_batch_analyze(
    positions: list,
    analysis_paths: list,
    instance_paths: list,
    chromatin_paths: list,
    frame_interval_minutes: float = 4.0,
    config: Optional[object] = None,
    version: Optional[str] = None,
    channels: Optional[list[str]] = None
) -> tuple[pd.DataFrame]:
    """
    Batch analyze chromatin segmentation for multiple positions.
    --------------------------------------------------------------------
    INPUTS:
        positions: list, position identifiers
        analysis_paths: list, paths to analysis Excel files
        instance_paths: list, paths to instance movie files
        chromatin_paths: list, paths to chromatin movie files
        frame_interval_minutes: float, time between frames in minutes
        config: Optional[ChromatinSegConfig], configuration parameters
        version: Optional[str], suffix appended to output filenames
        channels: Optional[list[str]], fluorescence channels to extract traces for; defaults to ['GFP']
    OUTPUTS:
        None (saves Excel files to disk)
    """

    if config is None:
        config = ChromatinSegConfig()
    if channels is None:
        channels = ["GFP"]

    for name_stub, analysis_path, instance_path, chromatin_path in zip(
        positions, analysis_paths, instance_paths, chromatin_paths
    ):

        save_dir = os.path.dirname(analysis_path)
        if not os.path.isdir(save_dir):
            save_dir = os.getcwd()
        vis_dir = os.path.join(save_dir, f"visualizations_{version}" if version else "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        try:
            instance = tiff.imread(instance_path)
            chromatin = tiff.imread(chromatin_path)
            analysis_df = pd.read_excel(analysis_path)
        except FileNotFoundError:
            print(
                f"Could not find either the instance movie, chromatin movie, or analysis dataframe for {analysis_path}"
            )
            continue

        analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        analysis_df.dropna(inplace=True)

        print(f"Working on position: {name_stub}")

        try:
            (
                intensity_traces, semantic_traces, frame_traces,
                cell_area_traces, ids, bkg_traces, shading_traces, offset_traces,
                raw_intensity_traces
            ) = retrieve_traces(analysis_df, channels, int(frame_interval_minutes))
        except KeyError as err:
            print(f"Skipping position {name_stub}: {err.args[0]} in {analysis_path}")
            continue

        channel_names = [output_column_names(wl) for wl in channels]
        intensity_names = [names[key] for names in channel_names for key in ("intensity", "intensity_raw")]
        correction_names = [names[key] for names in channel_names for key in ("bkg", "shading", "offset")]

        degradation_data = pd.DataFrame(
            {
                "cell_id": [],
                "frame": [],
                **{col: [] for col in intensity_names},
                "semantic_smoothed": [],
                'cell_area': [],
                **{col: [] for col in correction_names},
                "u_area": [],
                "u_area_intensity": [],
                "t_area": [],
                "t_area_intensity": [],
                "mtphs_plate_width": [],
                "u_chrom_num": [],
                "u_chrom_num_low": [],
                "u_chrom_num_high": [],
            }
        )

        cell_summary_data = []

        for i, cell_id in enumerate(ids):

            time_in_mitosis = np.sum(semantic_traces[i])

            ends_in_mitosis = (semantic_traces[i][-1] == 1)

            data_tuple = unaligned_chromatin(
                cell_id, analysis_df, instance, chromatin, config=config
            )

            if data_tuple is None:
                continue

            (
                u_area_trace,
                u_area_int_trace,
                u_num_trace,
                t_area_trace,
                t_area_int_trace,
                width_trace,
                visualization_stacks,
                removal_freq
            ) = data_tuple

            u_num_trace = []
            u_num_low_trace = []
            u_num_high_trace = []

            for meas in u_area_trace:
                meas *= 0.3387**2
                u_num, (u_num_low, u_num_high), _ = predict_integer_chromosomes(
                    meas,
                    108.4674,
                    0.056312,
                    1.9318,
                    8.15458,
                    0.25272
                )

                u_num_trace.append(u_num)
                u_num_low_trace.append(u_num_low)
                u_num_high_trace.append(u_num_high)

            frames = frame_traces[i]
            intensity_cols, correction_cols = channel_trace_columns(
                channels, i, intensity_traces, raw_intensity_traces,
                bkg_traces, shading_traces, offset_traces
            )

            cell_data = {
                "cell_id": [cell_id] * len(frames),
                "frame": frames,
                **intensity_cols,
                "semantic_smoothed": semantic_traces[i],
                "cell_area": cell_area_traces[i],
                **correction_cols,
                "u_area": u_area_trace,
                "u_area_intensity": u_area_int_trace,
                "a_area": np.asarray(t_area_trace) - np.asarray(u_area_trace),
                "a_area_intensity": np.asarray(t_area_int_trace) - np.asarray(u_area_int_trace),
                "t_area": t_area_trace,
                "t_area_intensity": t_area_int_trace,
                "mtphs_plate_width": width_trace,
                "u_chrom_num": u_num_trace,
                "u_chrom_num_low": u_num_low_trace,
                "u_chrom_num_high": u_num_high_trace,
            }

            degradation_data = pd.concat(
                [degradation_data, pd.DataFrame(cell_data)],
                ignore_index=True
            )

            cell_summary_data.append(
                {
                    "cell_id": cell_id,
                    "track_len": len(frames),
                    "track_start": frames[0],
                    "track_end": frames[-1],
                    "plate_removal_freq": removal_freq,
                    "time_in_mitosis": time_in_mitosis,
                    "ends_in_mitosis": ends_in_mitosis
                }
            )

            file_path = os.path.join(vis_dir, f'cell_{cell_id}.h5')
            with h5py.File(file_path, "w") as f:
                for i, src in enumerate(visualization_stacks):
                    grp = f.create_group(f"cell_{i}")
                    for name, arr in src.items():
                        grp.create_dataset(
                            name, data=arr, compression="gzip", compression_opts=4
                        )

        cell_summary_df = pd.DataFrame(cell_summary_data)

        analysis_info = pd.DataFrame(
            {
                "instance_path": [instance_path],
                "analysis_path": [analysis_path],
                "chromatin_path": [chromatin_path],
                "frame_interval_minutes": [frame_interval_minutes],
                "n_cells": [len(ids)],
                "total_frames": [len(degradation_data)],
            }
        )

        if hasattr(config, "__dict__"):
            config_info = pd.DataFrame([config.__dict__])
        else:
            config_info = pd.DataFrame()

        analysis_config_df = (
            pd.concat([analysis_info, config_info], axis=1)
            if not config_info.empty else analysis_info
        )

        if version:
            name_stub += f'_{version}'

        save_path = os.path.join(save_dir, f"{name_stub}_cycb_chromatin.xlsx")

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            degradation_data.to_excel(writer, sheet_name="degradation_data", index=False)
            cell_summary_df.to_excel(writer, sheet_name="cell_summary", index=False)
            analysis_config_df.to_excel(writer, sheet_name="analysis_config", index=False)


def cycb_batch_analyze_nochrom(
    positions: list,
    analysis_paths: list,
    instance_paths: list,
    frame_interval_minutes: float = 4.0,
    version: Optional[str] = None,
    channels: Optional[list[str]] = None
) -> None:

    """
    Batch analyze fluorescence traces (no chromatin segmentation) for multiple positions.
    --------------------------------------------------------------------
    INPUTS:
        positions: list, position identifiers
        analysis_paths: list, paths to analysis Excel files
        instance_paths: list, paths to instance movie files
        frame_interval_minutes: float, time between frames in minutes
        version: Optional[str], suffix appended to output filenames
        channels: Optional[list[str]], fluorescence channels to extract traces for; defaults to ['GFP']
    OUTPUTS:
        None (saves Excel files to disk)
    """

    if channels is None:
        channels = ["GFP"]

    for name_stub, analysis_path, instance_path in zip(
        positions,
        analysis_paths,
        instance_paths
    ):

        save_dir = os.path.dirname(analysis_path)
        if not os.path.isdir(save_dir):
            save_dir = os.getcwd()

        try:
            analysis_df = pd.read_excel(analysis_path)

        except FileNotFoundError:
            print(f"Could not load files for {analysis_path}")
            continue

        analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        analysis_df.dropna(inplace=True)

        print(f"Working on position: {name_stub}")

        try:
            (
                intensity_traces,
                semantic_traces,
                frame_traces,
                cell_area_traces,
                ids,
                bkg_traces,
                shading_traces,
                offset_traces,
                raw_intensity_traces
            ) = retrieve_traces(
                analysis_df,
                channels,
                int(frame_interval_minutes)
            )
        except KeyError as err:
            print(f"Skipping position {name_stub}: {err.args[0]} in {analysis_path}")
            continue

        channel_names = [output_column_names(wl) for wl in channels]
        intensity_names = [names[key] for names in channel_names for key in ("intensity", "intensity_raw")]
        correction_names = [names[key] for names in channel_names for key in ("bkg", "shading", "offset")]

        degradation_data = pd.DataFrame({
            "cell_id": [],
            "frame": [],
            **{col: [] for col in intensity_names},
            "semantic_smoothed": [],
            "cell_area": [],
            **{col: [] for col in correction_names},
            "u_area": [],
            "u_area_intensity": [],
            "a_area": [],
            "a_area_intensity": [],
            "t_area": [],
            "t_area_intensity": [],
            "mtphs_plate_width": [],
            "u_chrom_num": [],
            "u_chrom_num_low": [],
            "u_chrom_num_high": [],
        })

        cell_summary_data = []

        for i, cell_id in enumerate(ids):

            time_in_mitosis = np.sum(
                semantic_traces[i]
            )

            ends_in_mitosis = (semantic_traces[i][-1] == 1)

            frames = frame_traces[i]
            n = len(frames)
            intensity_cols, correction_cols = channel_trace_columns(
                channels, i, intensity_traces, raw_intensity_traces,
                bkg_traces, shading_traces, offset_traces
            )

            nan_trace = [np.nan] * n

            cell_data = {
                "cell_id": [cell_id] * n,
                "frame": frames,
                **intensity_cols,
                "semantic_smoothed": semantic_traces[i],
                "cell_area": cell_area_traces[i],
                **correction_cols,
                "u_area": nan_trace,
                "u_area_intensity": nan_trace,
                "a_area": nan_trace,
                "a_area_intensity": nan_trace,
                "t_area": nan_trace,
                "t_area_intensity": nan_trace,
                "mtphs_plate_width": nan_trace,
                "u_chrom_num": nan_trace,
                "u_chrom_num_low": nan_trace,
                "u_chrom_num_high": nan_trace,
            }

            degradation_data = pd.concat(
                [degradation_data, pd.DataFrame(cell_data)],
                ignore_index=True
            )

            cell_summary_data.append({
                "cell_id": cell_id,
                "track_len": len(frames),
                "track_start": frames[0],
                "track_end": frames[-1],
                "plate_removal_freq": np.nan,
                "time_in_mitosis": time_in_mitosis,
                "ends_in_mitosis": ends_in_mitosis
            })

        cell_summary_df = pd.DataFrame(cell_summary_data)

        analysis_info = pd.DataFrame({
            "instance_path": [instance_path],
            "analysis_path": [analysis_path],
            "chromatin_path": [None],
            "frame_interval_minutes": [frame_interval_minutes],
            "n_cells": [len(ids)],
            "total_frames": [len(degradation_data)],
        })

        if version:
            name_stub += f"_{version}"

        save_path = os.path.join(save_dir, f"{name_stub}_cycb_only.xlsx")

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            degradation_data.to_excel(writer, sheet_name="degradation_data", index=False)
            cell_summary_df.to_excel(writer, sheet_name="cell_summary", index=False)
            analysis_info.to_excel(writer, sheet_name="analysis_config", index=False)


if __name__ == "__main__":

    version = '0.3'

    root_dir = Path("/nfs/turbo/umms-ajitj/anishjv/cyclinb_analysis/20250621-cycb-noc")

    # Fluorescence channels to extract traces for. GFP (Cyclin B) is required and keeps the
    # legacy output columns; every other channel gets '{channel}_intensity', '{channel}_bkg', etc.
    channels = ["GFP"]

    # True: skip chromatin segmentation (writes *_cycb_only.xlsx).
    # False: segment unaligned chromatin from the chromatin_channel movie (writes *_cycb_chromatin.xlsx).
    cycb_only = False
    chromatin_channel = "Texas Red"

    if "GFP" not in channels:
        print("channels must include 'GFP' (Cyclin B)")
        sys.exit(1)

    inference_dirs = [
        obj.path
        for obj in os.scandir(root_dir)
        if "_inference" in obj.name and obj.is_dir()
    ]

    analysis_paths = []
    instance_paths = []
    chromatin_paths = []
    positions = []

    for dir in inference_dirs:

        name_stub = re.search(
            r"[A-H]([1-9]|[0][1-9]|[1][0-2])_s(\d{2}|\d{1})",
            str(dir)
        ).group()

        name_stub = str(name_stub)

        an_paths = glob.glob(f"{dir}/*analysis.xlsx")
        inst_paths = glob.glob(f"{dir}/*instance_movie.tif")

        if not cycb_only:
            chromatin_paths += [
                path
                for path in glob.glob(f"{root_dir}/*{chromatin_channel}.tif")
                if str(name_stub + '_') in path
            ]

        analysis_paths += an_paths
        instance_paths += inst_paths
        positions.append(name_stub)

    print(f"Channels: {channels}")

    # Run appropriate pipeline
    if not cycb_only:

        try:
            assert (
                len(analysis_paths) ==
                len(instance_paths) ==
                len(chromatin_paths)
            )

        except AssertionError:
            print("Files to analyze not organized properly")
            print("Analysis paths", len(analysis_paths))
            print("Instance paths", len(instance_paths))
            print(f"Chromatin ({chromatin_channel}) paths", len(chromatin_paths))
            sys.exit(1)

        print(f"Running chromatin-aware analysis (chromatin channel: {chromatin_channel})")

        cycb_chromatin_batch_analyze(
            positions,
            analysis_paths,
            instance_paths,
            chromatin_paths,
            frame_interval_minutes=4.0,
            version=version,
            channels=channels
        )

    else:

        try:
            assert len(analysis_paths) == len(instance_paths)

        except AssertionError:
            print("Files to analyze not organized properly")
            print("Analysis paths", len(analysis_paths))
            print("Instance paths", len(instance_paths))
            sys.exit(1)

        print("Running CycB-only analysis (no chromatin segmentation)")

        cycb_batch_analyze_nochrom(
            positions,
            analysis_paths,
            instance_paths,
            frame_interval_minutes=4.0,
            version=version,
            channels=channels
        )

    """
    Example with custom configuration for different experimental conditions
    from segment_chromatin import ChromatinSegConfig
    config = ChromatinSegConfig(
        min_chromatin_area=8,  # Higher threshold for cleaner data
        intensity_diff_ratio_threshold=0.3,  # More sensitive anaphase detection
        frame_interval_minutes=2.0  # Different acquisition rate
    )
    cycb_chromatin_batch_analyze(
        positions, analysis_paths, instance_paths, chromatin_paths,
        frame_interval_minutes=2.0, config=config
    )
    """
