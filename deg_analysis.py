from cmath import phase
import pandas as pd
import numpy as np
from typing import Optional
from skimage.restoration import denoise_tv_chambolle
from changept import *
import re

def smooth_cycb_chromatin(
    chromatin_df: pd.DataFrame,
    weight: Optional[int] = 12,
):
    """
    Smooths cyclin B and chromatin traces; computes derivative of signal
    ----------------------------------------------------------------------------------
    INPUTS:
        chromatin_df: pandas dataframe containing relevant information
        weight: weight to be used in TV denoising
    OUTPUTS:
        traces: list containing cyclinb traces
        smooth_traces: list containing smoothed cyclinb traces
        derivatives: list containing computed derivative traces
        uchromatin_traces: list containing euchromatin area traces
        achromatin_traces: list containing heterochromatin area traces
        semantic_traces: list containing semantic labels per cell
    """

    traces = []
    smooth_traces = []
    derivatives = []
    uchromatin_traces = []
    achromatin_traces = []
    semantic_traces = []
    id_traces = []
    frame_traces = []

    for id in chromatin_df["cell_id"].unique():
        trace = chromatin_df.query(f"cell_id=={id}")["cycb_intensity"].to_numpy()
        uchromatin = chromatin_df.query(f"cell_id=={id}")["u_chromatin_area"].to_numpy()
        tchromatin = chromatin_df.query(f"cell_id=={id}")["t_chromatin_area"].to_numpy()
        achromatin = tchromatin - uchromatin
        semantic = chromatin_df.query(f"cell_id=={id}")["semantic"].to_numpy()
        frames = chromatin_df.query(f'cell_id=={id}')["frame"].to_numpy()
        ids = chromatin_df.query(f'cell_id=={id}')["cell_id"].to_numpy()

        smooth_trace = denoise_tv_chambolle(trace, weight=weight)
        first_deriv = np.gradient(smooth_trace)

        traces.append(trace)
        smooth_traces.append(smooth_trace)
        derivatives.append(first_deriv)
        uchromatin_traces.append(uchromatin)
        achromatin_traces.append(achromatin)
        semantic_traces.append(semantic)
        id_traces.append(ids)
        frame_traces.append(frames)

    return (
        traces,
        smooth_traces,
        derivatives,
        uchromatin_traces,
        achromatin_traces,
        semantic_traces,
        id_traces,
        frame_traces
    )


def unpack_cycb_chromatin(
    traces: list,
    derivatives: list,
    semantics: list,
    uchromatin_traces: list,
    achromatin_traces: list,
    changepts: list,
    id_traces: list,
    frame_traces: list,
    remove_end_mitosis: Optional[bool] = True,
):
    """
    Unpacks per-timepoint features around mitosis for downstream comparison and modeling
    ------------------------------------------------------
    INPUTS:
        traces: list, smoothed Cyclin B intensity traces per cell
        derivatives: list, first derivatives of smoothed Cyclin B traces per cell
        semantics: list, per-timepoint semantic labels (1 indicates mitosis) per cell
        uchromatin_traces: list, euchromatin area traces per cell
        achromatin_traces: list, heterochromatin area traces per cell
        changepts: list, detected changepoint indices per cell (np.nan if unavailable)
        id_traces: list, cell ID traces per cell
        frame_traces: list, frame number traces per cell
        remove_end_mitosis: Optional[bool], skip cells ending the movie in mitosis when True
    OUTPUTS:
        unpacked_smooth_cycb: list[float], per-timepoint Cyclin B values within mitosis window
        unpacked_dcycb_dt: list[float], per-timepoint degradation rates (negative derivative)
        unpacked_uchromatin_area: list[float], per-timepoint euchromatin area within mitosis
        unpacked_achromatin_area: list[float], per-timepoint heterochromatin area within mitosis
        unpacked_pos_in_mitosis: list[float], normalized position in mitosis [0,1]
        phase_flag: list[str], label 'slow' before changepoint and 'fast' after
        min_after_changept: list[float], minutes relative to changepoint (scaled as 4*(t - cp))
        unpacked_tracking_ids: list[int], per-timepoint cell IDs within mitosis window
        unpacked_frames: list[int], per-timepoint frame numbers within mitosis window
    """

    unpacked_smooth_cycb = []
    unpacked_dcycb_dt = []
    unpacked_uchromatin_area = []
    unpacked_achromatin_area = []
    unpacked_pos_in_mitosis = []
    phase_flag = []
    min_after_changept = []
    unpacked_tracking_ids = []
    unpacked_frames = []

    nan_registry = np.isnan(changepts)
    for j, cell_trace in enumerate(traces):
        semantic = semantics[j]
        changept = changepts[j]
        deg_rate = -1 * derivatives[j]
        uchromatin = uchromatin_traces[j]
        achromatin = achromatin_traces[j]
        tracking_ids = id_traces[j]
        frames = frame_traces[j]

        # if changepoint detection failed ignore cell
        if nan_registry[j]:
            continue
        else:
            pass

        # if cell ended movie in mitosis, optionally ignore cell
        if remove_end_mitosis and semantic[-1] == 1:
            continue
        else:
            pass

        mitotic_indices = np.where(semantic == 1)[0]
        cycb_in_mitosis = cell_trace[semantic == 1]
        max_cycb = np.argmax(cycb_in_mitosis)
        low_bound = mitotic_indices[max_cycb]
        t_min = min(mitotic_indices)
        t_max = max(mitotic_indices)

        for t, val in enumerate(cell_trace):
            if low_bound < t and t_max >= t:
                flag = "slow" if t <= changept else "fast"
                min_after_changept.append(4 * (t - changept))
                unpacked_smooth_cycb.append(val)
                unpacked_dcycb_dt.append(deg_rate[t])
                unpacked_uchromatin_area.append(uchromatin[t])
                unpacked_achromatin_area.append(achromatin[t])
                unpacked_pos_in_mitosis.append((t - t_min) / (t_max - t_min))
                phase_flag.append(flag)
                unpacked_tracking_ids.append(tracking_ids[t])
                unpacked_frames.append(frames[t])

            else:
                pass

    return (
        unpacked_smooth_cycb,
        unpacked_dcycb_dt,
        unpacked_uchromatin_area,
        unpacked_achromatin_area,
        unpacked_pos_in_mitosis,
        phase_flag,
        min_after_changept,
        unpacked_tracking_ids,
        unpacked_frames,
    )


def create_aggregate_df(paths: list[str], remove_end_mitosis: Optional[bool] = True):
    """
    Creates an aggregated DataFrame from multiple Excel files of chromatin analysis outputs
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        paths: list[str], file paths to Excel files containing chromatin analysis data
        remove_end_mitosis: Optional[bool], skip cells ending the movie in mitosis when True
    OUTPUTS:
        df: pd.DataFrame, aggregated rows with columns ['cycb', 'deg_rate', 'uchromatin', 'achromatin', 'pos_in_mitosis', 'phase_flag', 'min_after_changept', 'tracking_id', 'frame', 'date-well']
    """

    usc_cont = []
    udd_cont = []
    uca_cont = []
    aca_cont = []
    upm_cont = []
    date_well_cont = []
    pf_cont = []
    mac_cont = []
    tracking_id_cont = []
    frame_cont = []
    pos_cont = []

    for path in paths:
        chromatin_df = pd.read_excel(path)
        _, smooth_traces, derivatives, uchromatin, achromatin, semantics, id_traces, frame_traces = (
            smooth_cycb_chromatin(chromatin_df, weight=5)
        )
        changepts = [
            peaks_changept(cycb, deg_rate, sem)
            for cycb, deg_rate, sem in zip(smooth_traces, derivatives, semantics)
        ]

        (
            unpacked_smooth_cycb,
            unpacked_dcycb_dt,
            unpacked_chromatin_area,
            unpacked_achromatin_area,
            unpacked_pos_in_mitosis,
            phase_flag,
            min_after_changept,
            unpacked_tracking_ids,
            unpacked_frames,
        ) = unpack_cycb_chromatin(
            smooth_traces,
            derivatives,
            semantics,
            uchromatin,
            achromatin,
            changepts,
            id_traces,
            frame_traces,
            remove_end_mitosis,
        )
        usc_cont += unpacked_smooth_cycb
        udd_cont += unpacked_dcycb_dt
        uca_cont += unpacked_chromatin_area
        aca_cont += unpacked_achromatin_area
        upm_cont += unpacked_pos_in_mitosis
        pf_cont += phase_flag
        mac_cont += min_after_changept
        tracking_id_cont += unpacked_tracking_ids
        frame_cont += unpacked_frames

        well = re.search(r"[A-H]([1-9]|0[1-9]|1[0-2])_s(\d{1,2})", str(path)).group()
        date = re.search(
            r"20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])", str(path)
        ).group()
        date_well = date + "-" + well[0]
        date_well_cont += [date_well] * len(unpacked_smooth_cycb)

        pos = well.split('_')[-1]
        pos_cont += [pos] * len(unpacked_smooth_cycb)

    df = pd.DataFrame(
        {
            "cycb": usc_cont,
            "deg_rate": udd_cont,
            "uchromatin": uca_cont,
            "achromatin": aca_cont,
            "pos_in_mitosis": upm_cont,
            "phase_flag": pf_cont,
            "min_after_changept": mac_cont,
            "tracking_id": tracking_id_cont,
            "frame": frame_cont,
            "date-well": date_well_cont,
            "position": pos_cont
        }
    )

    return df

def save_chromatin_crops(
    cell: np.ndarray,
    tophat_cell: np.ndarray,
    cell_mask: np.ndarray,
    labeled_chromosomes: np.ndarray,
    removal_mask: np.ndarray,
) -> np.ndarray:
    """
    Create a stack of chromatin crops for a single cell across metaphase timepoints.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    cell: np.ndarray, raw grayscale crop of the cell
    tophat_cell: np.ndarray, preprocessed cell image for metaphase plate visualization
    cell_mask: np.ndarray, binary mask of the cell
    labeled_chromosomes: np.ndarray, labeled chromosome regions
    removal_mask: np.ndarray, mask for removed regions
    OUTPUTS:
    crop_stack: np.ndarray, shape (3, H, W, 3) containing:
        [0] raw cell (converted to RGB)
        [1] metaphase plate visualization
        [2] colored overlay of unaligned chromosomes + cell mask
    """

    # --- Convert grayscale raw cell to RGB ---
    cell_rgb = np.stack([cell]*3, axis=-1)  # shape: (H, W, 3)

    # --- Create metaphase plate visualization ---
    from segment_chromatin import get_largest_signal_regions
    labeled_regions, max_lbl = get_largest_signal_regions(tophat_cell, cell)
    region_mask = labeled_regions == max_lbl

    metaphase_plate_crop = np.stack([tophat_cell]*3, axis=-1)
    metaphase_plate_crop[region_mask, 2] = 255    # Blue overlay for region
    metaphase_plate_crop[removal_mask, 0] = 255   # Red overlay for removal

    # --- Create colored overlay for unaligned chromosomes ---
    unaligned_chromosomes_crop = np.zeros((*labeled_chromosomes.shape, 3), dtype=np.uint8)
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
        [0, 128, 0],    # Dark Green
        [128, 128, 0],  # Olive
    ]

    unique_labels = np.unique(labeled_chromosomes[labeled_chromosomes > 0])
    for i, lbl in enumerate(unique_labels):
        mask = labeled_chromosomes == lbl
        color_idx = i % len(colors)
        unaligned_chromosomes_crop[mask] = colors[color_idx]

    # Overlay the cell mask in white
    unaligned_chromosomes_crop[cell_mask] = [255, 255, 255]

    # --- Stack all three crops ---
    crop_stack = np.stack([
        cell_rgb,
        metaphase_plate_crop,
        unaligned_chromosomes_crop
    ], axis=0)  # shape: (3, H, W, 3)

    return crop_stack
