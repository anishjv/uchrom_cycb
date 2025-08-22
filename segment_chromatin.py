import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import zoom

from typing import Optional

import skimage
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    disk,
)
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.restoration import unsupervised_wiener, richardson_lucy
import warnings
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ChromatinSegConfig:
    """Configuration for chromatin segmentation parameters."""
    top_hat_radius: int = 11
    mask_dilation_radius: int = 3
    psf_sigma: float = 2.0
    psf_size: int = 19
    min_chromatin_area: int = 4
    intensity_diff_ratio_threshold: float = 0.5
    eccentricity_threshold: float = 0.8
    euler_threshold: float = -5
    lookahead_minutes: float = 20.0
    frame_interval_minutes: float = 4.0
    border_dilation_radius: int = 9
    top_hat_radius_remove_border: int = 11


def adjust_zoom_factor(
    chromatin_shape: tuple[int, int, int], instance_shape: tuple[int, int, int]
) -> float:
    """
    Adjust zoom factor based on input shapes.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        chromatin_shape: tuple
        instance_shape: tuple
    OUTPUTS:
        zoom_factor: float
    """
    try:
        assert (
            chromatin_shape[1] / instance_shape[1]
            == chromatin_shape[2] / instance_shape[2]
        )
        return chromatin_shape[2] / instance_shape[2]
    except AssertionError:
        raise ValueError("Chromatin and Instance must be square arrays")


def remove_border_objs(img: npt.NDArray):
    """
    Removes objects touching the border of a given image
    -------------------------------------------------------
    INPUTS:
        img: np.ndarray
    OUTPUTS:
        cleared_img: np.ndarray of same shape as img

    """

    thresh = threshold_otsu(img)
    binary = img > thresh
    labels = label(binary)
    cleared_labels = clear_border(labels)

    border_objs = np.logical_xor(cleared_labels > 0, binary)
    border_objs = binary_dilation(border_objs, disk(9))

    bkg_pixels = img <= thresh
    bkg = np.median(img[bkg_pixels > 0])

    cleared_img = np.copy(img)
    cleared_img[border_objs > 0] = bkg

    return cleared_img


def bbox(img: npt.NDArray) -> tuple[int]:
    """
    Returns the minimum bounding box of a boolean array containing one region of True values
    ------------------------------------------------------------------------------------------
    INPUTS:
        img: npt.NDArray
    OUTPUTS:
        rmin: float, lower row index
        rmax: float, upper row index
        cmin: float, lower column index
        cmax: float, upper column index
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def prepare_cell_image(
    chromatin: np.ndarray,
    frame: int,
    bbox_coords: tuple[int, int, int, int],
    top_hat_radius: int = 11,
) -> np.ndarray:
    """
    Prepares the cell image by cropping and applying top-hat filtering and gaussian smoothing.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        chromatin: np.ndarray
        frame: int
        bbox_coords: tuple (rmin, rmax, cmin, cmax)
    OUTPUTS:
        cell: np.ndarray
        nobkg_cell: np.ndarray
    """
    rmin, rmax, cmin, cmax = bbox_coords
    cell = chromatin[frame, rmin:rmax, cmin:cmax]
    cell = remove_border_objs(cell)

    nobkg_cell = skimage.morphology.white_tophat(cell, disk(top_hat_radius))

    return cell, nobkg_cell


def compute_cell_images(
    instance: np.ndarray,
    chromatin: np.ndarray,
    frame: int,
    label_id: int,
    zoom_factor: float,
    dilate_radius: int = 3,
    top_hat_radius: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build zoomed mask, bbox and return (cell, nobkg_cell) for a given frame/label.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        instance: np.ndarray
        chromatin: np.ndarray
        frame: int
        label_id: int
        zoom_factor: float
    OUTPUTS:
        cell: np.ndarray
        nobkg_cell: np.ndarray
    """
    mask = instance[frame] == label_id
    if zoom_factor != 1:
        mask = zoom(mask, zoom_factor, order=0)
    zoom_mask = binary_dilation(mask, disk(dilate_radius))
    bbox_coords = bbox(zoom_mask)
    cell, nobkg_cell = prepare_cell_image(
        chromatin, frame, bbox_coords, top_hat_radius=top_hat_radius
    )
    rmin, rmax, cmin, cmax = bbox_coords
    cropped_mask = zoom_mask[rmin:rmax, cmin:cmax]
    return cell, nobkg_cell, cropped_mask


def get_largest_signal_regions(
    nobkg_cell, cell: np.ndarray, num_regions: int = 1
) -> tuple[list[regionprops], np.ndarray]:
    """
    Segments the cell and returns the brightest regions.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray
        num_regions: int
    OUTPUTS:
        sorted_regions: List[regionprops]
        labeled: np.ndarray
    """
    thresh = threshold_otsu(nobkg_cell)
    thresh_cell = nobkg_cell > thresh
    labeled, num_labels = label(thresh_cell, return_num=True, connectivity=1)
    labels = np.linspace(1, num_labels, num_labels).astype(int)
    region_intensities = [np.nansum(cell[labeled == (lbl)]) for lbl in labels]
    if len(region_intensities) == 0:
        return labeled, None, None, 1

    max_intensity = max(region_intensities)
    max_intensity_lbl = labels[region_intensities.index(max_intensity)]

    if len(region_intensities) > 1:
        second_max_intensity = sorted(region_intensities)[-2]
        nxt_max_intensity_lbl = labels[region_intensities.index(second_max_intensity)]

        intensity_diff_ratio = (max_intensity - second_max_intensity) / max_intensity

    else:
        nxt_max_intensity_lbl = None
        intensity_diff_ratio = 1

    return labeled, max_intensity_lbl, nxt_max_intensity_lbl, intensity_diff_ratio


def remove_regions(labels: list[int], labeled: np.ndarray) -> np.ndarray:
    """
    Removes specified labeled regions by dilating their masks.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        regions: List[regionprops]
        labeled: np.ndarray
    OUTPUTS:
        removal_mask: np.ndarray
    """
    removal_mask = np.zeros_like(labeled, dtype=bool)
    for lbl in labels:
        removal_mask[labeled == lbl] = 1
    return binary_dilation(removal_mask, disk(9))


def remove_metaphase_if_eccentric(
    lbl: int, 
    labeled: np.ndarray,
    eccentricity_threshold: float = 0.8,
    euler_threshold: float = -5,
) -> np.ndarray:
    """
    Removes the metaphase plate only if its eccentricity exceeds threshold.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        region: regionprops
        labeled: np.ndarray
        eccentricity_threshold: float, threshold for eccentricity
        euler_threshold: float, threshold for Euler number
    OUTPUTS:
        removal_mask: np.ndarray
    """
    region_mask = np.zeros_like(labeled, dtype=bool)
    region_mask[labeled == lbl] = 1
    props = regionprops(label(region_mask.astype(int)))[0]
    eccentricity = props.eccentricity
    euler_number = props.euler_number

    if eccentricity > eccentricity_threshold and euler_number > euler_threshold:
        print("metaphase; removing plate")
        return binary_dilation(region_mask, disk(9))
    else:
        print("metaphase; NOT removing plate")
        return np.zeros_like(labeled, dtype=bool)


def determine_removal_mask(
    nobkg_cell: np.ndarray, 
    cell: np.ndarray, 
    idx: int, 
    semantics: list[int],
    frame_interval_minutes: float = 4.0,
    lookahead_minutes: float = 20.0,
    intensity_diff_ratio_threshold: float = 0.5,
    eccentricity_threshold: float = 0.8,
    euler_threshold: float = -5,
) -> tuple[Optional[np.ndarray], Optional[bool]]:
    """
    Compute removal mask for metaphase/anaphase and flag whether anaphase-like condition is met.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        nobkg_cell: np.ndarray
        cell: np.ndarray
        idx: int, index within the semantics timeline
        semantics: list[int], per-frame semantic labels
        frame_interval_minutes: float, time between frames in minutes
        lookahead_minutes: float, time window to look ahead for end-of-mitosis
        intensity_diff_ratio_threshold: float, threshold for intensity difference ratio
        eccentricity_threshold: float, threshold for eccentricity
        euler_threshold: float, threshold for Euler number
    OUTPUTS:
        removal_mask: np.ndarray or None if the cell was lost
        is_anaphase: bool or None if the cell was lost
    """
    labeled_regions, max_lbl, second_lbl, intensity_diff_ratio = (
        get_largest_signal_regions(nobkg_cell, cell, num_regions=2)
    )

    if max_lbl is None:
        return None, None

    if second_lbl:
        # Look-ahead window based on frame interval
        check_range = int(lookahead_minutes // frame_interval_minutes)
        to_check = idx + check_range if (idx + check_range) < len(semantics) else -1
        near_end_of_mitosis = any(s == 0 for s in semantics[idx:to_check])

        if intensity_diff_ratio < intensity_diff_ratio_threshold and near_end_of_mitosis:
            removal_mask = remove_regions([max_lbl, second_lbl], labeled_regions)
            return removal_mask, True
        else:
            removal_mask = remove_metaphase_if_eccentric(
                max_lbl, labeled_regions,
                eccentricity_threshold=eccentricity_threshold,
                euler_threshold=euler_threshold
            )
            return removal_mask, False
    else:
        removal_mask = remove_metaphase_if_eccentric(
            max_lbl, labeled_regions,
            eccentricity_threshold=eccentricity_threshold,
            euler_threshold=euler_threshold
        )
        return removal_mask, False


def segment_unaligned_chromosomes(
    cell: np.ndarray, removal_mask: np.ndarray, min_area: int
) -> tuple[int, int, int]:
    """
    Segments and measures properties of unaligned chromosomes.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray
        removal_mask: np.ndarray
        min_area: int
    OUTPUTS:
        total_area: int
        total_intensity: int
        object_count: int
    """
    binary_mask = segment_mask_unaligned(cell, removal_mask, psf_sigma=2, psf_size=19)

    labeled = label(binary_mask, connectivity=1)
    num_labels = labeled.max()
    labels = np.arange(1, num_labels + 1, dtype=int)

    areas, intensities = [], []
    for lbl in labels:
        area = np.nansum(labeled == lbl)
        if area >= min_area:
            intensity = np.nansum(cell[labeled == lbl])
            areas.append(area)
            intensities.append(intensity)

    return np.nansum(areas), np.nansum(intensities), len(areas)


def measure_whole_cell(cell: np.ndarray) -> tuple[int, float, float]:
    """
    Measures whole cell area, total intensity, and average intensity.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray
    OUTPUTS:
        area: int
        intensity: float
        avg_intensity: float
    """
    thresh = threshold_otsu(cell)
    labeled = clear_border(label(cell > thresh))
    mask = labeled > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(cell[mask])

    return np.nansum(mask), np.nansum(cell[mask]), mean


def measure_whole_cell_with_instance(cell: np.ndarray, instance_mask: np.ndarray) -> tuple[int, float]:
    """
    Measure whole-cell metrics using the (zoomed, cropped) instance mask instead of thresholding.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray
        instance_mask: np.ndarray of bools aligned to cell
    OUTPUTS:
        area: int
        intensity: float
    """
    mask = instance_mask.astype(bool)
    return np.nansum(mask), np.nansum(cell[mask])


def segment_mask_unaligned(
    cell: np.ndarray, removal_mask: np.ndarray, psf_sigma: float = 2, psf_size: int = 19
) -> np.ndarray:
    """
    Build a binary mask of unaligned chromosomes by deconvolving, removing structures,
    thresholding, and clearing borders.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray
        removal_mask: np.ndarray
        psf_sigma: float, Gaussian sigma for PSF smoothing
        psf_size: int, size of square PSF kernel (odd preferred)
    OUTPUTS:
        binary_mask: np.ndarray of bools
    """
    perfect_psf = np.zeros((psf_size, psf_size))
    center = psf_size // 2
    perfect_psf[center, center] = 1
    psf = gaussian(perfect_psf, psf_sigma)

    deconv_cell = unsupervised_wiener(cell, psf, clip=False)[0]
    cell_minus_struct = np.copy(deconv_cell)
    cell_minus_struct[removal_mask] = 0

    thresh = threshold_otsu(cell_minus_struct)
    labeled, _ = label(cell_minus_struct > thresh, return_num=True, connectivity=1)
    binary_mask = clear_border(labeled) > 0
    return binary_mask


def unaligned_chromatin(
    identity: int,
    analysis_df: pd.DataFrame,
    instance: np.ndarray,
    chromatin: np.ndarray,
    min_chromatin_area: Optional[int] = 4,
    frame_interval_minutes: float = 4.0,
    config: Optional[ChromatinSegConfig] = None,
) -> tuple[list[int], list[int], list[float], list[int], list[int], int]:
    """
    Given an image capturing histone fluoresence, returns the area emitting of signal emitting regions minus the
    area of the largest signal emitting region (corresponds with unaligned chromosomes in metaphase)
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        identity: int
        analysis_df: pd.DataFrame
        instance: np.ndarray
        chromatin: np.ndarray
        min_chromatin_area: Optional[int]
        frame_interval_minutes: float, time between frames in minutes
        config: Optional[ChromatinSegConfig], configuration parameters
    OUTPUTS:
        area_signal: List[int]
        intensity_signal: List[int]
        whole_cell_intensity: List[float]
        num_signals: List[int]
        whole_cell_area: List[int]
        first_anaphase: int
    """
    if config is None:
        config = ChromatinSegConfig()
    
    zoom_factor = adjust_zoom_factor(chromatin.shape, instance.shape)
    frames_data = analysis_df.query(f"particle == {identity}")
    semantics = frames_data["semantic_smoothed"].tolist()

    results = []
    anaphase_indices = []
    first_anaphase = None
    print(f"Working on cell {identity}")

    for idx, row in frames_data.iterrows():
        f, l, semantic = (
            int(row["frame"]),
            int(row["label"]),
            int(row["semantic_smoothed"]),
        )

        cell, nobkg_cell, cropped_mask = compute_cell_images(
            instance, chromatin, f, l, zoom_factor,
            dilate_radius=config.mask_dilation_radius,
            top_hat_radius=config.top_hat_radius
        )

        if semantic == 1:
            removal_mask, is_anaphase = determine_removal_mask(
                nobkg_cell, cell, idx, semantics,
                frame_interval_minutes=frame_interval_minutes,
                lookahead_minutes=config.lookahead_minutes,
                intensity_diff_ratio_threshold=config.intensity_diff_ratio_threshold,
                eccentricity_threshold=config.eccentricity_threshold,
                euler_threshold=config.euler_threshold
            )

            if removal_mask is None:
                print("Lost track of cell! Moving on to next cell")
                return None

            if is_anaphase:
                anaphase_indices.append(idx)

            if len(anaphase_indices) > 0:
                consecutives = np.split(
                    anaphase_indices, np.where(np.diff(anaphase_indices) != 1)[0] + 1
                )
                for sublist in consecutives:
                    if len(sublist) > 1:
                        first_anaphase = sublist[0]
                        break
                    else:
                        first_anaphase = anaphase_indices[0]
            else:
                # anaphase may not be detected due to finite temporal resolution
                first_anaphase = None

            area_sig, int_sig, num_sig = segment_unaligned_chromosomes(
                cell, removal_mask, config.min_chromatin_area
            )
        else:
            area_sig, int_sig, num_sig = 0, 0, 0

        # Prefer instance-mask based measurement to avoid thresholding biases
        whole_area, whole_intensity = measure_whole_cell_with_instance(
            cell, cropped_mask
        )

        results.append(
            (
                area_sig,
                int_sig,
                whole_intensity,
                num_sig,
                whole_area,
            )
        )

    (
        area_signal,
        intensity_signal,
        whole_cell_intensity,
        num_signals,
        whole_cell_area,
    ) = zip(*results)

    return (
        list(area_signal),
        list(intensity_signal),
        list(whole_cell_intensity),
        list(num_signals),
        list(whole_cell_area),
        first_anaphase,
    )


def display_save(
    rois: npt.NDArray,
    path: str,
    identity: int,
    cmap="gray",
    step=2,
    well_pos: Optional[str] = None,
):
    """
    Function for creating montages
    ------------------------------
    INPUTS
        rois: npt.NDArray, array with shape = (t, x, y) where t is the number of timepoints
        path: str
        identity: int
        cmap: str
        step: int
        well_pos: str

    """
    data_montage = skimage.util.montage(rois[0::step], padding_width=4, fill=np.nan)
    fig, ax = plt.subplots(figsize=(40, 40))
    ax.imshow(data_montage, cmap=cmap)
    title = f"Cell {identity} from {well_pos}" if well_pos else f"Cell {identity}"
    ax.set_title(title)
    ax.set_axis_off()

    if path != None:
        fig.savefig(path, dpi=300)


def extract_montages(
    identity: int,
    analysis_df: pd.DataFrame,
    instance: npt.NDArray,
    chromatin: npt.NDArray,
    cmap: Optional[str] = None,
    well_pos: Optional[str] = None,
    save_path: Optional[str] = None,
    mode: Optional[str] = "seg",
    frame_interval_minutes: float = 4.0,
    config: Optional[ChromatinSegConfig] = None,
):
    """
    Function for saving montages of ROIs
    ------------------------------------
    INPUTS:
        identity: int,
        analysis_df: pd.DataFrame.
        instance: npt.NDArray,
        chromatin: npt.NDArray,
        well_pos: str,
        save_path:str,
        mode: str,
        frame_interval_minutes: float, time between frames in minutes
        config: Optional[ChromatinSegConfig], configuration parameters
    TODO:
        color masks by area extracted
    """
    if config is None:
        config = ChromatinSegConfig()

    zoom_factor = adjust_zoom_factor(chromatin.shape, instance.shape)

    rois = []
    frames = analysis_df.query(f"particle=={identity}")["frame"].tolist()
    markers = analysis_df.query(f"particle=={identity}")["label"].tolist()
    semantics = analysis_df.query(f"particle=={identity}")["semantic_smoothed"].tolist()

    for idx, (f, l, classifier) in enumerate(zip(frames, markers, semantics)):
        # expand mask and capture indices
        # trim cell and mask for efficiency
        cell, nobkg_cell, cropped_mask = compute_cell_images(
            instance, chromatin, f, l, zoom_factor,
            dilate_radius=config.mask_dilation_radius,
            top_hat_radius=config.top_hat_radius
        )

        if mode == "cell":
            rois.append(cell)
            continue

        if classifier == 1:
            removal_mask, is_anaphase = determine_removal_mask(
                nobkg_cell, cell, idx, semantics,
                frame_interval_minutes=frame_interval_minutes,
                lookahead_minutes=config.lookahead_minutes,
                intensity_diff_ratio_threshold=config.intensity_diff_ratio_threshold,
                eccentricity_threshold=config.eccentricity_threshold,
                euler_threshold=config.euler_threshold
            )
            if removal_mask is None:
                rois.append(cell)
                continue

            binary_mask = segment_mask_unaligned(
                cell, removal_mask, psf_sigma=config.psf_sigma, psf_size=config.psf_size
            )
            borders = binary_dilation(binary_mask) ^ binary_erosion(binary_mask)
            with_borders = np.copy(cell)
            with_borders[borders] = 0

            rois.append(with_borders)
        else:
            rois.append(cell)

    max_rows = max([roi.shape[0] for roi in rois])
    max_cols = max([roi.shape[1] for roi in rois])

    rois_new = []
    for roi in rois:
        template = np.zeros((max_rows, max_cols))
        template[: roi.shape[0], : roi.shape[1]] = roi
        rois_new.append(template)

    rois = np.asarray(rois_new)

    if not well_pos:
        display_save(rois, save_path, identity, cmap=cmap, step=2)
    else:
        display_save(rois, save_path, identity, step=2, cmap=cmap, well_pos=well_pos)

    return rois
