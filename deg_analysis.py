from cmath import phase
import pandas as pd
import numpy as np
from typing import Optional, List
from skimage.restoration import denoise_tv_chambolle
from changept import *
import re
import h5py
from matplotlib.colors import to_rgba

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

    from segment_chromatin import get_largest_signal_regions
    H, W = cell.shape
    raw_img = cell.copy()  # raw grayscale base

    # --- Helper function for blending ---
    def blend_overlay(base_rgba, overlay_rgba):
        """Alpha blend overlay_rgba onto base_rgba."""
        alpha = overlay_rgba[..., 3:4]
        base_rgba[..., :3] = (1 - alpha) * base_rgba[..., :3] + alpha * overlay_rgba[..., :3]
        base_rgba[..., 3] = np.clip(base_rgba[..., 3] + alpha[..., 0], 0, 1)
        return base_rgba

    # --- Metaphase overlay ---
    labeled_regions, max_lbl = get_largest_signal_regions(tophat_cell, cell)
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

    # --- Unaligned chromosomes overlay ---
    unaligned_overlay = np.zeros((H, W, 4), dtype=float)
    colors = [
        [1, 0, 0, 0.2], [0, 1, 0, 0.2], [0, 0, 1, 0.2], [1, 1, 0, 0.2],
        [1, 0, 1, 0.2], [0, 1, 1, 0.2], [1, 0.5, 0, 0.2], [0.5, 0, 1, 0.2],
        [0, 0.5, 0, 0.2], [0.5, 0.5, 0, 0.2]
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


def visualize_chromatin_hd5(file_path: str, max_cells: Optional[int] = 10):
    """
    Open an HDF5 file containing chromatin crop stacks and visualize them.
    ----------------------------------------------------------------------
    INPUTS:
        file_path : str, Path to the HDF5 file.
        max_cells : int, optional
    """
    
    with h5py.File(file_path, "r") as f:
        cell_groups = list(f.keys())
        n_cells = min(len(cell_groups), max_cells)
        print(f"Found {len(cell_groups)} crop stacks, displaying {n_cells}.")

        # --- Infer target shape ---
        heights, widths = [], []
        for col in range(n_cells):
            grp = f[cell_groups[col]]
            raw_img = grp['0'][()]
            heights.append(raw_img.shape[0])
            widths.append(raw_img.shape[1])
        target_shape = (max(heights), max(widths))

        fig, axes = plt.subplots(3, n_cells, figsize=(3*n_cells, 9), squeeze=False)

        for col in range(n_cells):
            grp = f[cell_groups[col]]
            raw_img = grp['0'][()].astype(float)/255   # dataset 0
            metaphase_overlay = grp['1'][()]
            unaligned_overlay = grp['2'][()]

            # --- Pad images to target_shape ---
            def pad_to_shape(img, target_shape, value=0):
                H, W = img.shape[:2]
                pad_h = target_shape[0] - H
                pad_w = target_shape[1] - W
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                if img.ndim == 2:  # grayscale
                    return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=value)
                else:  # RGBA
                    return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), constant_values=value)

            raw_img = pad_to_shape(raw_img, target_shape, value=0)
            metaphase_overlay = pad_to_shape(metaphase_overlay, target_shape, value=0)
            unaligned_overlay = pad_to_shape(unaligned_overlay, target_shape, value=0)

            # --- Scale overlay RGB to darken ---
            metaphase_overlay_disp = metaphase_overlay.copy()
            metaphase_overlay_disp[..., :3] *= 3
            unaligned_overlay_disp = unaligned_overlay.copy()
            unaligned_overlay_disp[..., :3] *= 3

            # Row 0: raw grayscale
            axes[0, col].imshow(raw_img, cmap="gray")
            axes[0, col].axis("off")


            # Row 2: metaphase overlay over raw
            axes[1, col].imshow(raw_img, cmap="gray")
            axes[1, col].imshow(metaphase_overlay_disp)
            axes[1, col].axis("off")


            # Row 1: unaligned overlay over raw
            axes[2, col].imshow(raw_img, cmap="gray")
            axes[2, col].imshow(unaligned_overlay_disp)
            axes[2, col].axis("off")

        # Row labels
        row_labels = ["Raw", "Unaligned Chromosomes"]
        for i, label in enumerate(row_labels):
            axes[i, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=40, va="center")

        plt.tight_layout()
        plt.show()

        return fig