import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import zoom

from typing import Optional

import skimage
from skimage.morphology import disk, binary_dilation, remove_small_objects
from skimage.filters import threshold_otsu, gaussian, threshold_li
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.restoration import richardson_lucy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.special import j1
import cv2
from extractRect import findRotMaxRect
import os
from pathlib import Path
import glob
import re
import tifffile as tiff
import random
from PIL import Image
from deg_analysis import save_chromatin_crops


@dataclass
class ChromatinSegConfig:
    """Configuration for chromatin segmentation parameters."""

    top_hat_radius: int = 11
    psf_size: int = 19
    gaussian_sigma: float = 0
    min_chromatin_area: int = 20
    eccentricity_threshold: float = 0.7 #largest region must have eccentricty greater than threshold to be considered for metaphase plate detection
    euler_threshold: float = -2 # can be no more than three holes in the metaphase plate


def airy_disk_psf(NA, wavelength_nm, pixel_size_um, psf_size=51, oversample=1):
    """
    Generate a 2D Airy disk PSF (diffraction-limited) for a widefield microscope.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        NA: float, numerical aperture of the objective
        wavelength_nm: float, emission wavelength in nanometers
        pixel_size_um: float, camera pixel size in microns
        psf_size: int, size of the PSF array (default=51)
        oversample: int, number of sub-pixels per image pixel (default=4)
    OUTPUTS:
        psf: np.ndarray, 2D array normalized to sum to 1
    """

    if psf_size % 2 == 0:
        psf_size -= 1

    wavelength_um = wavelength_nm / 1000
    k = 2 * np.pi * NA / wavelength_um

    dx = pixel_size_um / oversample
    half = psf_size // 2
    x = np.arange(-half, half + 1) * dx
    y = np.arange(-half, half + 1) * dx
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    kr = k * R
    kr[kr == 0] = 1e-10  # avoid divide by zero
    airy = (2 * j1(kr) / kr) ** 2

    psf = airy
    return psf


def adjust_zoom_factor(
    chromatin_shape: tuple[int, int, int], instance_shape: tuple[int, int, int]
) -> float:
    """
    Adjust zoom factor based on input shapes.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        chromatin_shape: tuple[int, int, int], shape of chromatin array
        instance_shape: tuple[int, int, int], shape of instance array
    OUTPUTS:
        zoom_factor: float, calculated zoom factor
    """
    try:
        assert (
            chromatin_shape[1] / instance_shape[1]
            == chromatin_shape[2] / instance_shape[2]
        )
        return chromatin_shape[2] / instance_shape[2]
    except AssertionError:
        raise ValueError("Chromatin and Instance must be square arrays")


def bbox(img: npt.NDArray, expansion_factor: Optional[float] = 0) -> tuple[int]:
    """
    Returns the minimum bounding box of a boolean array containing one region of True values.
    Optionally expands the bounding box by a specified factor.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        img: npt.NDArray, boolean array with one region of True values
        expansion_factor: float, factor to expand the bounding box (default=0.0, no expansion)
                         e.g., 0.1 expands by 10% on each side
    OUTPUTS:
        rmin: int, lower row index (expanded)
        rmax: int, upper row index (expanded)
        cmin: int, lower column index (expanded)
        cmax: int, upper column index (expanded)
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Calculate expansion amounts
    if expansion_factor > 0:
        height = rmax - rmin
        width = cmax - cmin

        # Calculate expansion in pixels
        row_expansion = int(height * expansion_factor)
        col_expansion = int(width * expansion_factor)

        # Expand the bounding box
        rmin = max(0, rmin - row_expansion)
        rmax = min(img.shape[0], rmax + row_expansion)
        cmin = max(0, cmin - col_expansion)
        cmax = min(img.shape[1], cmax + col_expansion)

    return rmin, rmax, cmin, cmax


def compute_cell_images(
    instance: np.ndarray,
    chromatin: np.ndarray,
    frame: int,
    label_id: int,
    zoom_factor: float,
    config: Optional[ChromatinSegConfig] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build zoomed mask, bbox and return (cell, tophat_cell, mask, deconv_cell) for a given frame/label.
    Processing pipeline: gaussian smoothing -> deconvolution -> top-hat filtering
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        instance: np.ndarray, instance segmentation masks
        chromatin: np.ndarray, chromatin fluorescence images
        frame: int, frame index
        label_id: int, cell label ID
        zoom_factor: float, zoom factor for mask resizing
        config: Optional[ChromatinSegConfig], configuration parameters (default=None)
    OUTPUTS:
        cell: np.ndarray, cropped chromatin image
        tophat_cell: np.ndarray, top-hat filtered cell image (for segmentation)
        mask: np.ndarray, cell mask
        deconv_cell: np.ndarray, deconvolved cell image (for segmentation only, not intensity measurements)
    """
    if config is None:
        config = ChromatinSegConfig()

    mask = instance[frame] == label_id
    if zoom_factor != 1:
        mask = zoom(mask, zoom_factor, order=0)
    bbox_coords = bbox(mask, expansion_factor=0.25)
    mask_cropped = mask[
        bbox_coords[0] : bbox_coords[1], bbox_coords[2] : bbox_coords[3]
    ]

    # Get the cropped cell image
    rmin, rmax, cmin, cmax = bbox_coords
    cell = chromatin[frame, rmin:rmax, cmin:cmax]

    # Processing pipeline for segmentation:
    # 1. Gaussian smoothing to reduce noise
    smoothed_cell = gaussian(cell, sigma=config.gaussian_sigma)

    # 2. Deconvolution to improve resolution
    min_im_dim = min(smoothed_cell.shape[0], smoothed_cell.shape[1])
    psf = airy_disk_psf(
        NA=0.45,
        wavelength_nm=624,
        pixel_size_um=0.3387,
        psf_size=min(config.psf_size, min_im_dim),
    )
    deconv_cell = richardson_lucy(smoothed_cell, psf, num_iter=5)

    # 3. Top-hat filtering on deconvolved image for better structure enhancement
    tophat_cell = skimage.morphology.white_tophat(
        deconv_cell, disk(config.top_hat_radius)
    )

    return cell, tophat_cell, mask_cropped, deconv_cell


def get_largest_signal_regions(
    tophat_cell: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, Optional[int]]:
    """
    Segments the cell and returns the brightest region.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        tophat_cell: np.ndarray, top-hat filtered cell image
        cell: np.ndarray, cropped chromatin image
    OUTPUTS:
        labeled: np.ndarray, labeled image with background = 0
        max_intensity_lbl: int or None, label of the brightest region
    """
    thresh = threshold_otsu(tophat_cell)
    thresh_cell = tophat_cell > thresh
    labeled, num_labels = label(thresh_cell, return_num=True, connectivity=1)

    if num_labels == 0:
        return labeled, None

    labels = np.linspace(1, num_labels, num_labels).astype(int)
    region_intensities = [np.nansum(cell[labeled == (lbl)]) for lbl in labels]

    max_intensity = max(region_intensities)
    max_intensity_lbl = labels[region_intensities.index(max_intensity)]

    return labeled, max_intensity_lbl


def calculate_region_orientation(region_mask: np.ndarray, cell: np.ndarray) -> float:
    """
    Calculate the orientation angle of a region using its principal axes.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        region_mask: np.ndarray, boolean mask of the region
        cell: np.ndarray, intensity image
    OUTPUTS:
        angle_degrees: float, orientation angle in degrees (0-180)
    """
    # Get region properties
    props = regionprops(label(region_mask.astype(int)), intensity_image=cell)[0]
    mu20 = props.weighted_moments_central[2, 0]
    mu02 = props.weighted_moments_central[0, 2]
    mu11 = props.weighted_moments_central[1, 1]

    # Get the orientation (returns angle in radians, -π/2 to π/2)
    orientation_weighted_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    # Convert to degrees
    angle_degrees = np.degrees(orientation_weighted_rad)

    # Transform from regionprops coordinate system to extractRect coordinate system:
    # regionprops: counter-clockwise from π/2 (vertical), range -90° to 90°
    # extractRect: clockwise from 0 (horizontal), range 0° to 180°
    # Transformation: flip direction and shift reference
    transformed_angle = (-angle_degrees + 90) % 180

    return transformed_angle


def remove_metaphase_plate(
    lbl: int,
    labeled: np.ndarray,
    cell: np.ndarray,
    config: Optional[ChromatinSegConfig] = None,
) -> np.ndarray:
    """
    Removes the metaphase plate by finding the largest inscribed rectangle
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        lbl: int, label of the region to check
        labeled: np.ndarray, labeled image
        cell: np.ndarray, intensity image
        config: Optional[ChromatinSegConfig], configuration parameters (default=None)
    OUTPUTS:
        removal_mask: np.ndarray, boolean mask of regions to remove
    """
    if config is None:
        config = ChromatinSegConfig()

    region_mask = np.zeros_like(labeled, dtype=bool)
    region_mask[labeled == lbl] = 1

    # Convert to format expected by findRotMaxRect (0=foreground, 1=background)
    data_for_rect = np.ones_like(region_mask, dtype=int)
    data_for_rect[region_mask] = 0

    # First test: Check Euler number on region_mask
    props = regionprops(label(region_mask.astype(int)))[0]
    euler_number = props.euler_number
    eccentricity = props.eccentricity

    if euler_number <= config.euler_threshold:
        return np.zeros_like(labeled, dtype=bool)

    if eccentricity < config.eccentricity_threshold:
        return np.zeros_like(labeled, dtype=bool)

    try:
        # Calculate the orientation of the region mask
        region_orientation = calculate_region_orientation(region_mask, cell)

        rect_coords_ori, angle_optimal, _ = findRotMaxRect(
            data_for_rect,
            flag_opt=True,
            nbre_angle=20,
            flag_parallel=False,
            flag_out="rotation",
            flag_enlarge_img=False,
            limit_image_size=1000,
            initial_angle=region_orientation,
        )

    except Exception as e:
        print(e)
        return np.zeros_like(labeled, dtype=bool)

    rect_coords_ori = np.asarray(rect_coords_ori)
    rect_transformed = np.zeros_like(rect_coords_ori)
    rect_transformed[:, 0] = rect_coords_ori[:, 1]  # swap (x, y) to (y, x)
    rect_transformed[:, 1] = rect_coords_ori[:, 0]  # swap (x, y) to (y, x)
    rect_transformed = np.asarray(rect_transformed, dtype=np.int64)

    rect_mask = np.zeros_like(labeled, dtype="uint8")
    cv2.fillPoly(rect_mask, [rect_transformed], 255).astype("bool")

    # create custom structuring element
    new_shape = (rect_mask.shape[0] // 1, rect_mask.shape[0] // 1)
    struct = np.asarray(Image.fromarray(rect_mask).resize(new_shape))
    bbox_coords = bbox(struct)
    struct = struct[
        bbox_coords[0] : bbox_coords[1], bbox_coords[2] : bbox_coords[3]
    ]  # for efficiency
    rect_mask = binary_dilation(
        rect_mask, struct
    )  # dilate to approach size of circumscribed (not inscribed) rectangle

    return rect_mask


def determine_removal_mask(
    tophat_cell: np.ndarray,
    cell: np.ndarray,
    config: Optional[ChromatinSegConfig] = None,
) -> Optional[np.ndarray]:
    """
    Compute removal mask for metaphase plate based on eccentricity and Euler number.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        tophat_cell: np.ndarray, top-hat filtered cell image
        cell: np.ndarray, cropped chromatin image
        config: Optional[ChromatinSegConfig], configuration parameters (default=None)
    OUTPUTS:
        removal_mask: np.ndarray or None if the cell was lost
    """
    if config is None:
        config = ChromatinSegConfig()

    labeled_regions, max_lbl = get_largest_signal_regions(tophat_cell, cell)

    if max_lbl is None:
        return None

    removal_mask = remove_metaphase_plate(max_lbl, labeled_regions, cell, config)
    return removal_mask


def segment_mask_unaligned(
    removal_mask: np.ndarray,
    tophat_cell: np.ndarray,
    cell_mask: np.ndarray,
    config: Optional[ChromatinSegConfig] = None,
) -> np.ndarray:
    """
    Build a labeled image of unaligned chromosomes by removing structures from deconvolved image,
    thresholding, and clearing borders. Only keeps objects that are within the cell mask.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        removal_mask: np.ndarray, mask of regions to remove
        tophat_cell: np.ndarray, deconvolved and whitetophat-ed cell image (for segmentation only)
        cell_mask: np.ndarray, boolean mask defining the cell boundary
        config: Optional[ChromatinSegConfig], configuration parameters (default=None)
    OUTPUTS:
        labeled: np.ndarray, labeled image with background = 0
    """

    if config is None:
        config = ChromatinSegConfig()

    # Remove metaphase plate from deconvolved image
    cell_minus_struct = np.copy(tophat_cell)
    cell_minus_struct[removal_mask] = 0

    # Threshold the image
    thresh = threshold_li(cell_minus_struct[cell_minus_struct > 0])
    labeled = label(cell_minus_struct > thresh, connectivity=1)
    labeled = remove_small_objects(labeled, min_size=config.min_chromatin_area)

    # Only keep objects that are within the cell mask
    # Set any labeled regions outside the cell mask to 0
    labeled[~cell_mask] = 0

    # Clear border objects (this will remove objects touching the image border)
    labeled = clear_border(labeled)

    total_chromatin_mask = tophat_cell > thresh
    total_chromatin_mask[~cell_mask] = 0

    return labeled, total_chromatin_mask


def segment_unaligned_chromosomes(
    cell: np.ndarray,
    tophat_cell: np.ndarray,
    removal_mask: np.ndarray,
    cell_mask: np.ndarray,
    config: Optional[ChromatinSegConfig] = None,
):
    """
    Segments and measures properties of unaligned chromosomes and total chromatin.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        cell: np.ndarray, cropped chromatin image (for intensity measurements)
        tophat_cell: np.ndarray, top-hat filtered cell image (for segmentation)
        removal_mask: np.ndarray, mask of regions to remove
        cell_mask: np.ndarray, boolean mask defining the cell boundary
        config: Optional[ChromatinSegConfig], configuration parameters (default=None)
    OUTPUTS:
        total_area: int, total area of unaligned chromosomes
        total_intensity: int, total intensity of unaligned chromosomes
        object_count: int, number of unaligned chromosome objects
        total_chromatin_area: int, total area of chromatin above threshold
        total_chromatin_intensity: float, total intensity of chromatin above threshold
    """

    if config is None:
        config = ChromatinSegConfig()

    labeled, total_chromatin_mask = segment_mask_unaligned(
        removal_mask, tophat_cell, cell_mask
    )

    # Compute total chromatin measurements from the mask
    total_chromatin_area = np.nansum(total_chromatin_mask)
    total_chromatin_intensity = np.nansum(cell[total_chromatin_mask])

    # Get the actual labels present in the image (after clear_border may have removed some)
    labels = np.unique(labeled[labeled > 0])

    if labels.size == 0:
        measurements = (0, 0, 0, total_chromatin_area, total_chromatin_intensity)
        return measurements, labeled

    areas, intensities = [], []
    for lbl in labels:
        area = np.nansum(labeled == lbl)
        if area >= config.min_chromatin_area:
            # Use cell for intensity measurements (not deconvolved)
            intensity = np.nansum(cell[labeled == lbl])
            areas.append(area)
            intensities.append(intensity)

    measurements = (
        np.nansum(areas),
        np.nansum(intensities),
        len(areas),
        total_chromatin_area,
        total_chromatin_intensity,
    )

    return measurements, labeled


def find_contiguous_ranges(frame_list: list[int]) -> list[tuple[int, int]]:
    """
    Find contiguous ranges in a list of frame numbers.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        frame_list: list[int], sorted list of frame numbers
    OUTPUTS:
        ranges: list[tuple[int, int]], list of (start, end) tuples for contiguous ranges
    """
    if not frame_list:
        return []

    ranges = []
    start = frame_list[0]
    end = frame_list[0]

    for i in range(1, len(frame_list)):
        if frame_list[i] == end + 1:
            end = frame_list[i]
        else:
            ranges.append((start, end))
            start = frame_list[i]
            end = frame_list[i]

    ranges.append((start, end))
    return ranges


def unaligned_chromatin(
    identity: int,
    analysis_df: pd.DataFrame,
    instance: np.ndarray,
    chromatin: np.ndarray,
    config: Optional[ChromatinSegConfig] = None,
) -> tuple[list[int], list[int], list[float], list[int], list[int], int, int, int] | None:
    """
    Given an image capturing histone fluorescence, returns the area of signal emitting regions minus the
    area of the largest signal emitting region (corresponds with unaligned chromosomes in metaphase).
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        identity: int, cell identifier
        analysis_df: pd.DataFrame, analysis dataframe with cell tracking information
        instance: np.ndarray, instance segmentation masks
        chromatin: np.ndarray, chromatin fluorescence images
        config: Optional[ChromatinSegConfig], configuration parameters (default=None)
    OUTPUTS:
        area_signal: list[int], areas of unaligned chromosome regions
        intensity_signal: list[int], intensities of unaligned chromosome regions
        total_chromatin_intensity: list[float], total chromatin intensities
        num_signals: list[int], number of unaligned chromosome objects
        total_chromatin_area: list[int], total chromatin areas
        num_removal_regions: int, number of contiguous regions where metaphase plate was removed
        first_removal_frame: int, first frame where metaphase plate was removed (or -1 if none)
        last_removal_frame: int, last frame where metaphase plate was removed (or -1 if none)
    """

    if config is None:
        config = ChromatinSegConfig()

    zoom_factor = adjust_zoom_factor(chromatin.shape, instance.shape)
    frames_data = analysis_df.query(f"particle == {identity}")

    results = []
    successful_removal_frames = (
        []
    )  # Track frames where metaphase plate was successfully removed
    print(f"Working on cell {identity}")

    visualization_stacks = []

    for _, row in frames_data.iterrows():
        f, l, semantic = (
            int(row["frame"]),
            int(row["label"]),
            int(row["semantic_smoothed"]),
        )

        cell, tophat_cell, cell_mask, deconv_cell = compute_cell_images(
            instance, chromatin, f, l, zoom_factor, config
        )

        if semantic == 1:
            removal_mask = determine_removal_mask(tophat_cell, cell, config)

            if removal_mask is None: #this happens only if no chromatin was found
                print("Lost track of cell! Moving on to next cell")
                return None

            # Check if metaphase plate was successfully removed (non-zero removal mask)
            if np.any(removal_mask):
                successful_removal_frames.append(f)

            measurements, labeled_chromosmes = segment_unaligned_chromosomes(
                cell, tophat_cell, removal_mask, cell_mask, config
            )
            (
                area_sig,
                int_sig,
                num_sig,
                total_chromatin_area,
                total_chromatin_intensity,
            ) = measurements


            crop_stack = save_chromatin_crops(
                                            cell, 
                                            tophat_cell, 
                                            cell_mask, 
                                            labeled_chromosmes, 
                                            removal_mask
                                            )
            visualization_stacks.append(crop_stack)

        else:
            area_sig, int_sig, num_sig = 0, 0, 0
            total_chromatin_area, total_chromatin_intensity = 0, 0.0

        results.append(
            (
                area_sig,
                int_sig,
                total_chromatin_intensity,
                num_sig,
                total_chromatin_area,
            )
        )
    
    (
        area_signal,
        intensity_signal,
        total_chromatin_intensity,
        num_signals,
        total_chromatin_area,
    ) = zip(*results)

    # Calculate metaphase plate removal metrics
    num_removal_regions = 0
    first_removal_frame = -1
    last_removal_frame = -1

    if successful_removal_frames:
        successful_removal_frames.sort()
        ranges = find_contiguous_ranges(successful_removal_frames)
        num_removal_regions = len(ranges)
        first_removal_frame = successful_removal_frames[0]
        last_removal_frame = successful_removal_frames[-1]
        print(
            f"Cell {identity}: Metaphase plates were removed from time point {first_removal_frame} to {last_removal_frame} in {num_removal_regions} contigous regions"
        )
    else:
        print(f"Cell {identity}: No metaphase plates were removed")

    return (
        list(area_signal),
        list(intensity_signal),
        list(total_chromatin_intensity),
        list(num_signals),
        list(total_chromatin_area),
        num_removal_regions,
        first_removal_frame,
        last_removal_frame,
        visualization_stacks
    )


def visualize_chromatin_processing(
    identity: int,
    analysis_df: pd.DataFrame,
    instance: np.ndarray,
    chromatin: np.ndarray,
    n_rows: int = 10,
    config: Optional[ChromatinSegConfig] = None,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 16),
    dpi: int = 150,
) -> None:
    """
    Visualize the chromatin processing pipeline for a single cell across metaphase timepoints.
    Creates a plot with N rows and 5 columns showing the processing steps.
    Only displays timepoints where semantic == 1 (metaphase).
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        identity: int, cell identifier
        analysis_df: pd.DataFrame, analysis dataframe with cell tracking information
        instance: np.ndarray, instance segmentation masks
        chromatin: np.ndarray, chromatin fluorescence images
        n_rows: int, number of metaphase timepoints to display (default=10)
        config: Optional[ChromatinSegConfig], configuration parameters (default=None)
        save_path: Optional[str], path to save the plot (if None, displays plot)
        figsize: tuple[int, int], figure size in inches (width, height)
        dpi: int, dots per inch for saved figure
    OUTPUTS:
        None (displays or saves plot)
    """
    if config is None:
        config = ChromatinSegConfig()

    # Get cell data and filter for metaphase only
    frames_data = analysis_df.query(
        f"particle == {identity} and semantic_smoothed == 1"
    )
    if len(frames_data) == 0:
        print(f"No mitotic timepoints found for cell {identity}")
        return

    # Select evenly spaced metaphase timepoints
    total_metaphase_frames = len(frames_data)
    if n_rows > total_metaphase_frames:
        n_rows = total_metaphase_frames
        print(
            f"Warning: Only {total_metaphase_frames} mitotic frames available, showing all frames"
        )

    # Calculate frame indices to display
    if n_rows == total_metaphase_frames:
        frame_indices = list(range(total_metaphase_frames))
    else:
        frame_indices = [
            int(i * (total_metaphase_frames - 1) / (n_rows - 1)) for i in range(n_rows)
        ]

    # Create figure
    fig, axes = plt.subplots(n_rows, 5, figsize=figsize, dpi=dpi)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Column titles
    col_titles = [
        "Original Cell",
        "Deconvolved",
        "Top-hat Filtered",
        "Metaphase Plate Mask",
        "Unaligned Chromosomes",
    ]

    # Set column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    zoom_factor = adjust_zoom_factor(chromatin.shape, instance.shape)

    print("frame_indices", frame_indices)
    for row, frame_idx in enumerate(frame_indices):
        row_data = frames_data.iloc[frame_idx]
        f, l = (
            int(row_data["frame"]),
            int(row_data["label"]),
        )

        # Get processed images
        cell, tophat_cell, cell_mask, deconv_cell = compute_cell_images(
            instance, chromatin, f, l, zoom_factor, config
        )

        # Column 1: Original cell
        axes[row, 0].imshow(cell, cmap="gray")
        axes[row, 0].set_ylabel(f"Frame {f}", fontsize=10, fontweight="bold")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # Column 2: Deconvolved
        axes[row, 1].imshow(deconv_cell, cmap="gray")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        # Column 3: Top-hat filtered
        axes[row, 2].imshow(tophat_cell, cmap="gray")
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

        # Column 4: Metaphase plate mask
        removal_mask = determine_removal_mask(tophat_cell, cell, config)
        if removal_mask is not None:
            axes[row, 3].imshow(tophat_cell, cmap="gray")

            # Get the original region mask for comparison
            labeled_regions, max_lbl = get_largest_signal_regions(tophat_cell, cell)
            if max_lbl is not None:
                region_mask = labeled_regions == max_lbl

                # Create colored overlays
                # Blue for original region mask
                colored_region = np.zeros((*region_mask.shape, 4))
                colored_region[region_mask] = [0, 0, 1, 0.2]  # Blue with alpha

                # Red for hybrid rectangle (removal mask)
                colored_removal = np.zeros((*removal_mask.shape, 4))
                colored_removal[removal_mask] = [1, 0, 0, 0.2]  # Red with alpha

                # Overlay both
                axes[row, 3].imshow(colored_region)
                axes[row, 3].imshow(colored_removal)
            else:
                # Fallback to just showing removal mask
                colored_removal = np.zeros((*removal_mask.shape, 4))
                colored_removal[removal_mask] = [1, 0, 0, 0.2]  # Red with alpha
                axes[row, 3].imshow(colored_removal)
        else:
            axes[row, 3].text(
                0.5,
                0.5,
                "No metaphase\nplate detected",
                ha="center",
                va="center",
                transform=axes[row, 3].transAxes,
                fontsize=10,
                color="red",
            )
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])

        # Column 5: Unaligned chromosomes
        if removal_mask is not None:
            labeled_chromosomes, _ = segment_mask_unaligned(
                removal_mask, tophat_cell, cell_mask, config
            )

            # Create a colored overlay for different chromosome regions
            colored_chromosomes = np.zeros((*labeled_chromosomes.shape, 4))

            # Define a color palette for different chromosomes
            colors = [
                [1, 0, 0, 0.2],  # Red
                [0, 1, 0, 0.2],  # Green
                [0, 0, 1, 0.2],  # Blue
                [1, 1, 0, 0.2],  # Yellow
                [1, 0, 1, 0.2],  # Magenta
                [0, 1, 1, 0.2],  # Cyan
                [1, 0.5, 0, 0.2],  # Orange
                [0.5, 0, 1, 0.2],  # Purple
                [0, 0.5, 0, 0.2],  # Dark Green
                [0.5, 0.5, 0, 0.2],  # Olive
            ]

            unique_labels = np.unique(labeled_chromosomes[labeled_chromosomes > 0])
            for i, lbl in enumerate(unique_labels):
                mask = labeled_chromosomes == lbl
                color_idx = i % len(
                    colors
                )  # Cycle through colors if more than 10 chromosomes
                colored_chromosomes[mask] = colors[color_idx]

            colored_cellmask = np.zeros((*cell_mask.shape, 4))
            colored_cellmask[cell_mask] = [1, 1, 1, 0.2]  # White with alpha

            axes[row, 4].imshow(tophat_cell, cmap="gray")
            axes[row, 4].imshow(colored_chromosomes)
            axes[row, 4].imshow(colored_cellmask)
            axes[row, 4].set_xticks([])
            axes[row, 4].set_yticks([])

        else:
            axes[row, 4].text(
                0.5,
                0.5,
                "Cell lost\ntracking",
                ha="center",
                va="center",
                transform=axes[row, 4].transAxes,
                fontsize=10,
                color="red",
            )
            axes[row, 4].set_xticks([])
            axes[row, 4].set_yticks([])

    # Add overall title
    fig.suptitle(
        f"Chromatin Processing Pipeline - Cell {identity})",
        fontsize=16,
        fontweight="bold",
    )

    # Adjust layout
    plt.tight_layout()

    # Save or display
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":

    root_dir = Path("/Users/whoisv/Desktop/metphs_example/") #path to folder containing cell_app inference directory and chromatin image
    inference_dirs = [
        obj.path
        for obj in os.scandir(root_dir)
        if "_inference" in obj.name and obj.is_dir()
    ]

    # Find all available data files
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
        exit(1)

    # Example: visualize processing for a random cell in the first position
    if len(analysis_paths) > 0:
        # Load data for the first position
        analysis_df = pd.read_excel(analysis_paths[0])
        instance = tiff.imread(instance_paths[0])
        chromatin = tiff.imread(chromatin_paths[0])

        # Clean the data
        analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        analysis_df.dropna(inplace=True)

        # Get available cells
        available_cells = analysis_df["particle"].unique()

        if len(available_cells) > 0:
            # Find cells with metaphase timepoints
            cells_with_metaphase = []
            for cell_id in available_cells:
                metaphase_frames = analysis_df.query(
                    f"particle == {cell_id} and semantic_smoothed == 1"
                )
                if len(metaphase_frames) > 0:
                    cells_with_metaphase.append(cell_id)

            if len(cells_with_metaphase) > 0:
                # Randomly select a cell
                cell_id = random.choice(cells_with_metaphase)
                metaphase_frames = analysis_df.query(
                    f"particle == {cell_id} and semantic_smoothed == 1"
                )

                print(f"Randomly selected cell {cell_id} from position {positions[0]}")
                print(f"Found {len(metaphase_frames)} metaphase timepoints")

                # Create save path
                save_dir = os.path.dirname(analysis_paths[0])
                save_path = os.path.join(
                    save_dir, f"{positions[0]}_cell_{cell_id}_processing.png"
                )

                config = ChromatinSegConfig() #this may be edited

                # Run visualization
                visualize_chromatin_processing(
                    cell_id,
                    analysis_df,
                    instance,
                    chromatin,
                    n_rows=10,  # Show 10 metaphase timepoints
                    config = config,
                    save_path=save_path,
                    figsize=(12, 16),
                    dpi=150,
                )
            else:
                print("No cells with metaphase timepoints found")
        else:
            print("No cells found in the analysis data")
    else:
        print("No data files found")
