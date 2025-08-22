import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile as tiff
import scipy.ndimage as ndi
from scipy.signal import find_peaks

from typing import Optional
import skimage
from skimage.morphology import (
    binary_closing,
)
from skimage.filters import gaussian
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import glob
import re
from segment_chromatin import (
    unaligned_chromatin, ChromatinSegConfig
)  # type:ignore


def retrieve_traces(
    analysis_df: pd.DataFrame,
    wl: str,
    frame_interval: int,
    remove_end_mitosis: Optional[bool] = False,
) -> tuple[list[npt.NDArray], list[npt.NDArray], list, list, list[npt.NDArray]]:
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
    """

    ids = []
    intensity_traces = []
    semantic_traces = []
    first_tps = []
    frame_traces = []
    t_char = 20 // frame_interval

    for id in analysis_df["particle"].unique():
        intensity = analysis_df.query(f"particle=={id}")[f"{wl}"].to_numpy()
        semantic = analysis_df.query(f"particle=={id}")["semantic_smoothed"].to_numpy()
        bkg = analysis_df.query(f"particle=={id}")[f"{wl}_bkg_corr"].to_numpy()
        intensity_corr = analysis_df.query(f"particle=={id}")[f"{wl}_int_corr"].to_numpy()
        frames = analysis_df.query(f"particle=={id}")["frame"].to_numpy()

        # Always remove traces that start in mitosis
        if semantic[0] == 1:
            continue
        # 1) If remove_end_mitosis is True and the trace ends in mitosis at the final frame, skip
        if remove_end_mitosis and semantic[-1] == 1:
            continue

        # Peak detection array: pad to allow end-plateau to be counted as a peak
        semantic_for_detection = ( np.append(semantic, np.zeros(3)))

        # Detect mitosis events; require exactly one event to satisfy
        _, props = find_peaks(semantic_for_detection, width=t_char)
        if props["widths"].size != 1:
            continue

        first_mitosis = props["left_bases"][0]
        corr_intensity = (intensity - bkg) * intensity_corr
        intensity_traces.append(corr_intensity)
        semantic_traces.append(semantic)
        frame_traces.append(frames)
        first_tps.append(first_mitosis)
        ids.append(id)

    return intensity_traces, semantic_traces, ids, first_tps, frame_traces


def watershed_split(
    binary_image: npt.NDArray, sigma: Optional[float] = 3.5
) -> npt.NDArray:
    """
    Splits and labels touching objects using the watershed algorithm
    ------------------------------------------------------------------
    INPUTS:
    	binary_img: image where 1s correspond to object regions and 0s correspond to background
    	sigma: standard deviation to use for gaussian kernal smoothing
    OUTPUTS:
    	labels: image of same size as binary_img but labeled
    """

    # distance transform
    distance = ndi.distance_transform_edt(
        binary_closing(binary_image, skimage.morphology.disk(9))
    )
    blurred_distance = gaussian(distance, sigma=sigma)

    # finding peaks in the distance transform
    coords = peak_local_max(blurred_distance, labels=binary_image)

    # creating markers and segmenting
    mask = np.zeros(binary_image.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    labels = watershed(-blurred_distance, markers, mask=binary_image)

    return labels


def longest_negative_sequence(arr):
    """
    Finds the longest contiguous run of negative values in a 1D array.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        arr: array-like, sequence of numeric values (will be converted to a NumPy array)
    OUTPUTS:
        If a negative run exists:
            start_idx: int, start index (inclusive) of the longest negative run
            end_idx: int, end index (inclusive) of the longest negative run
        If no negative run exists:
            0, None, None
    """

    arr = np.asarray(arr)
    # Boolean array: True where arr < 0
    is_negative = arr < 0

    # Identify the changes: start and end of sequences
    diff = np.diff(is_negative.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Edge cases: sequence starts at index 0
    if is_negative[0]:
        starts = np.r_[0, starts]
    # Sequence ends at last index
    if is_negative[-1]:
        ends = np.r_[ends, len(arr)]

    # Compute sequence lengths
    lengths = ends - starts

    if len(lengths) == 0:
        return 0, None, None  # No negative sequence found

    # Find the longest sequence
    max_idx = np.argmax(lengths)
    start_idx = starts[max_idx]
    end_idx = ends[max_idx]  # exclusive

    return start_idx, end_idx - 1  # inclusive end index


def cycb_chromatin_batch_analyze(
    positions: list,
    analysis_paths: list,
    instance_paths: list,
    chromatin_paths: list,
    frame_interval_minutes: float = 4.0,
    config: Optional[object] = None,
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
        intensity, semantic, ids, first_tps, frame_traces = retrieve_traces(
            analysis_df, "GFP", int(frame_interval_minutes)
        )

        # Initialize dataframe with empty lists for each column
        degradation_data = pd.DataFrame({
            'cell_id': [],
            'frame': [],
            'cycb_intensity': [],
            'semantic': [],
            'u_chromatin_area': [],
            'u_chromatin_intensity': [],
            't_chromatin_intensity': [],
            'num_u_chromosomes': [],
            't_chromatin_area': []
        })
        
        # Store cell-level summary data
        cell_summary_data = []
        
        # Process each cell and extend the dataframe directly
        for i, cell_id in enumerate(ids):
            # Get chromatin segmentation data for this cell
            data_tuple = unaligned_chromatin(
                cell_id, analysis_df, instance, chromatin,
                frame_interval_minutes=frame_interval_minutes,
                config=config
            )

            if data_tuple is None:
                continue
            
            # Get the actual frames for this cell
            frames = frame_traces[i]
            
            # Create data for this cell
            cell_data = {
                'cell_id': [cell_id] * len(frames),
                'frame': frames,
                'cycb_intensity': intensity[i],
                'semantic': semantic[i],
                'u_chromatin_area': data_tuple[0],
                'u_chromatin_intensity': data_tuple[1],
                't_chromatin_intensity': data_tuple[2],
                'num_u_chromosomes': data_tuple[3],
                't_chromatin_area': data_tuple[4]
            }
            
            # Extend the dataframe with this cell's data
            degradation_data = pd.concat([degradation_data, pd.DataFrame(cell_data)], ignore_index=True)
            
            # Store cell-level summary data
            cell_summary_data.append({
                'cell_id': cell_id,
                'first_mitosis': first_tps[i],
                'first_anaphase': data_tuple[5],
                'n_frames': len(frames),
                'frame_start': frames[0],
                'frame_end': frames[-1]
            })
        
        print(f"Created long-format dataframe with {len(degradation_data)} rows")
        print(f"Data format: {len(ids)} cells × {degradation_data['frame'].nunique()} unique frames")
        print(f"Frame range: {degradation_data['frame'].min()} to {degradation_data['frame'].max()}")
        print(f"Excel file contains: degradation_data, cell_summary, and analysis_config sheets")

        # Create cell summary dataframe
        cell_summary_df = pd.DataFrame(cell_summary_data)
        
        # Create analysis info dataframe
        analysis_info = pd.DataFrame({
            "instance_path": [instance_path],
            "analysis_path": [analysis_path],
            "chromatin_path": [chromatin_path],
            "frame_interval_minutes": [frame_interval_minutes],
            "n_cells": [len(ids)],
            "total_frames": [len(degradation_data)]
        })
        
        # Create config info dataframe separately
        if hasattr(config, '__dict__'):
            config_info = pd.DataFrame([config.__dict__])
        else:
            config_info = pd.DataFrame()
        
        # Combine the dataframes
        if not config_info.empty:
            analysis_config_df = pd.concat([analysis_info, config_info], axis=1)
        else:
            analysis_config_df = analysis_info

        save_dir = os.path.dirname(analysis_path)
        if not os.path.isdir(save_dir):
            save_dir = os.getcwd()
        save_path = os.path.join(save_dir, f"{name_stub}_cycb_chromatin.xlsx")

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            # Save main data in long format (similar to cellaap_analysis.py)
            # Each row represents a (cell_id, frame) combination with all attributes as columns
            degradation_data.to_excel(writer, sheet_name="degradation_data", index=False)
            
            # Save cell-level summary data
            cell_summary_df.to_excel(writer, sheet_name="cell_summary", index=False)
            
            # Save combined analysis and config metadata
            analysis_config_df.to_excel(writer, sheet_name="analysis_config", index=False)


if __name__ == "__main__":

    root_dir = Path("/Users/whoisv/Desktop/")
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
        an_paths = glob.glob(f"{dir}/*analysis.xlsx")
        inst_paths = glob.glob(f"{dir}/*instance_movie.tif")
        chrom_paths = [
            path
            for path in glob.glob(f"{root_dir}/*Texas Red.tif")
            if str(name_stub) in path
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
    
    # Use default configuration
    cycb_chromatin_batch_analyze(
        positions, analysis_paths, instance_paths, chromatin_paths,
        frame_interval_minutes=4.0
    )
    
    '''
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
    '''