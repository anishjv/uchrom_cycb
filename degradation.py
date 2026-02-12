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
from changept import validate_cyclin_b_trace
import sys
import h5py


def retrieve_traces(
    analysis_df: pd.DataFrame,
    wl: str,
    frame_interval: int,
    remove_end_mitosis: Optional[bool] = False,
) -> tuple[list[npt.NDArray], list[npt.NDArray], list, list, list[npt.NDArray], list]:
    """
    Retrieve corrected intensity and semantic traces for a channel, enforcing selection rules and experiment length constraints.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        analysis_df: pd.DataFrame, analysis.xlsx output from `https://github.com/ajitpj/cellapp-analysis`
        wl: str, channel name to process
        frame_interval: int, time between successive frames; sets t_char = 20 // frame_interval
        exp_length: int, total number of frames in the experiment; used for end-of-trace filtering
        remove_end_mitosis: bool, if True exclude traces that end mitotic at frame exp_length-1; otherwise allow a single end-plateau
    OUTPUTS:
        intensity_traces: list[npt.NDArray], corrected intensities (intensity - bkg) * intensity_corr
        semantic_traces: list[npt.NDArray], unpadded semantic traces
        ids: list, particle identifiers retained
        first_tps: list, indices of first mitotic call (left base of the detected peak)
        frame_traces: list[npt.NDArray], actual frame numbers for each trace
        last_mitotic_tps: list, indices of last mitotic call for each trace
    """

    ids = []
    intensity_traces = []
    semantic_traces = []
    frame_traces = []
    dead_traces = []
    t_char = 20 // frame_interval

    for id in analysis_df["particle"].unique():
        intensity = analysis_df.query(f"particle=={id}")[f"{wl}"].to_numpy()
        semantic = analysis_df.query(f"particle=={id}")["semantic_smoothed"].to_numpy()
        bkg = analysis_df.query(f"particle=={id}")[f"{wl}_bkg_corr"].to_numpy()
        shading = analysis_df.query(f"particle=={id}")[f"{wl}_int_corr"].to_numpy()
        offset = analysis_df.query(f"particle=={id}")["offset"].to_numpy()
        frames = analysis_df.query(f"particle=={id}")["frame"].to_numpy()
        dead = analysis_df.query(f"particle=={id}")["dead_flag"].to_numpy()

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

        corr_intensity = ((intensity - bkg) * shading) - offset
        intensity_traces.append(corr_intensity)
        semantic_traces.append(semantic)
        frame_traces.append(frames)
        dead_traces.append(dead)
        ids.append(id)

    return intensity_traces, semantic_traces, frame_traces, dead_traces, ids


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
    n_low = n_options[np.searchsorted(cumulative_prob, 0.025)]
    n_high = n_options[np.searchsorted(cumulative_prob, 0.975)]
    
    return int(n_best), (int(n_low), int(n_high)), probs


def cycb_chromatin_batch_analyze(
    positions: list,
    analysis_paths: list,
    instance_paths: list,
    chromatin_paths: list,
    frame_interval_minutes: float = 4.0,
    config: Optional[object] = None,
    version: Optional[str] = None
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
    OUTPUTS:
        None (saves Excel files to disk)
    """

    if config is None:
        config = ChromatinSegConfig()


    for name_stub, analysis_path, instance_path, chromatin_path in zip(
        positions, analysis_paths, instance_paths, chromatin_paths
    ):


        save_dir = os.path.dirname(analysis_path)
        if not os.path.isdir(save_dir):
            save_dir = os.getcwd()
        vis_dir = os.path.join(save_dir, "visualizations")
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
        intensity_traces, semantic_traces, frame_traces, dead_traces, ids = (
            retrieve_traces(analysis_df, "GFP", int(frame_interval_minutes))
        )

        # Initialize dataframe with empty lists for each column
        degradation_data = pd.DataFrame(
            {
                "cell_id": [],
                "frame": [],
                "cycb_intensity": [],
                "semantic_smoothed": [],
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

        # Store cell-level summary data
        cell_summary_data = []
        # Process each cell and extend the dataframe directly
        for i, cell_id in enumerate(ids):

            # Frequecy of dead calls in mitosis
            num_dead_flags = np.sum(semantic_traces[i]*dead_traces[i])
            #semantic traces are boolean, np.sum is the number of timepoints in mitosis
            #dead_freq = num_dead_flags / np.sum(semantic_traces[i]) 

            # Validate that intensity trace is a Cyclin B trace based on three criterion
            peaks, range, hysteresis = validate_cyclin_b_trace(intensity_traces[i])
                
            # Get chromatin segmentation data for this cell
            data_tuple = unaligned_chromatin(
                cell_id, analysis_df, instance, chromatin, config=config
            )

            if data_tuple is None:
                continue #skip the cell

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


            # New: for predicting chromosome number from chromosome areas
            u_num_trace = []
            u_num_low_trace = []
            u_num_high_trace = []
            for meas in u_area_trace:
                meas *= 0.3387**2 #conversion to square microns
                u_num, (u_num_low, u_num_high), _ = predict_integer_chromosomes(
                    meas,
                    123.83,
                    0.043,
                    13.57,
                    12.62,
                    0.0091
                )
                u_num_trace.append(u_num)
                u_num_low_trace.append(u_num_low)
                u_num_high_trace.append(u_num_high)

            # Get the actual frames for this cell
            frames = frame_traces[i]
            # Create data for this cell
            cell_data = {
                "cell_id": [cell_id] * len(frames),
                "frame": frames,
                "cycb_intensity": intensity_traces[i],
                "semantic_smoothed": semantic_traces[i],
                "u_area": u_area_trace,
                "u_area_intensity": u_area_int_trace,
                "t_area": t_area_trace,
                "t_area_intensity": t_area_int_trace,
                "mtphs_plate_width": width_trace,
                "u_chrom_num": u_num_trace,
                "u_chrom_num_low": u_num_low_trace,
                "u_chrom_num_high": u_num_high_trace,
            }

            # Extend the dataframe with this cell's data
            degradation_data = pd.concat(
                [degradation_data, pd.DataFrame(cell_data)], ignore_index=True
            )

            # Store cell-level summary data
            cell_summary_data.append(
                {
                    "cell_id": cell_id,
                    "track_len": len(frames),
                    "track_start": frames[0],
                    "track_end": frames[-1],
                    "plate_removal_freq": removal_freq,
                    "num_dead_flags": num_dead_flags,
                    "peaks_criterion": peaks,
                    "range_criterion": range,
                    "hysteresis_criterion": hysteresis
                }
            )

            file_path = os.path.join(vis_dir, f'cell_{cell_id}.h5') #for each cell
            with h5py.File(file_path, "w") as f:
                for i, stack in enumerate(visualization_stacks):
                    grp = f.create_group(f"cell_{i}") # for each stack type (should really have better names)
                    for j, img in enumerate(stack):   # for each image in each stack
                        grp.create_dataset(f'{j}', data=img, compression="gzip")


        # Create cell summary dataframe
        cell_summary_df = pd.DataFrame(cell_summary_data)

        # Create analysis info dataframe
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

        # Create config info dataframe separately
        if hasattr(config, "__dict__"):
            config_info = pd.DataFrame([config.__dict__])
        else:
            config_info = pd.DataFrame()

        # Combine the dataframes
        if not config_info.empty:
            analysis_config_df = pd.concat([analysis_info, config_info], axis=1)
        else:
            analysis_config_df = analysis_info

        if version:
            name_stub += f'_{version}'
        save_path = os.path.join(save_dir, f"{name_stub}_cycb_chromatin.xlsx")
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            # Save main data in long format (similar to cellaap_analysis.py)
            # Each row represents a (cell_id, frame) combination with all attributes as columns
            degradation_data.to_excel(
                writer, sheet_name="degradation_data", index=False
            )

            # Save cell-level summary data
            cell_summary_df.to_excel(writer, sheet_name="cell_summary", index=False)

            # Save combined analysis and config metadata
            analysis_config_df.to_excel(
                writer, sheet_name="analysis_config", index=False
            )



if __name__ == "__main__":

    version = '0.3'
    root_dir = Path("/nfs/turbo/umms-ajitj/anishjv/cyclinb_analysis/20250621-cycb-noc")
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
            r"[A-H]([1-9]|[0][1-9]|[1][0-2])_s(\d{2}|\d{1})", str(dir)
        ).group()
        name_stub = str(name_stub) #temporary (removed + '-')
        an_paths = glob.glob(f"{dir}/*analysis.xlsx")
        inst_paths = glob.glob(f"{dir}/*instance_movie.tif")
        chrom_paths = [
            path
            for path in glob.glob(f"{root_dir}/*Texas Red.tif")
            if str(name_stub + '_') in path
        ]
        analysis_paths += an_paths
        instance_paths += inst_paths
        chromatin_paths += chrom_paths
        positions.append(name_stub)

    try:
        assert len(analysis_paths) == len(instance_paths) == len(chromatin_paths)
    except AssertionError:
        print("Files to analyze not organized properly")
        print("Analysis paths", len(analysis_paths))
        print("Instance paths", len(instance_paths))
        print("Chromatin paths", len(chromatin_paths))
        sys.exit(1)
    
    # Use default configuration
    cycb_chromatin_batch_analyze(
        positions, analysis_paths, instance_paths, chromatin_paths,
        frame_interval_minutes=4.0, version=version
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
