import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import zoom

from typing import Optional, Tuple

import skimage
from skimage.exposure import rescale_intensity
from skimage.morphology import disk, remove_small_objects, binary_erosion
from skimage.filters import threshold_otsu, gaussian, threshold_li
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.restoration import richardson_lucy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.special import j1
import cv2
import os
from pathlib import Path
import glob
import re
import tifffile as tiff

import sys
sys.path.append('/Users/whoisv/')
from uchrom_cycb.deg_analysis import save_chromatin_crops
from uchrom_cycb.changept import validate_cyclin_b_trace
from uchrom_cycb.extractRect import findRotMaxRect

@dataclass
class ChromatinSegConfig:
    """Configuration for chromatin segmentation parameters."""

    top_hat_radius: int = 15
    psf_size: int = 19
    gaussian_sigma: float = 0
    min_chromatin_area: int = 20
    eccentricity_threshold: float = 0.7 #largest region must have eccentricty greater than threshold to be considered for metaphase plate detection
    euler_threshold: float = -2 # can be no more than three holes in the metaphase plate
    truncate_z: Optional[float] = None


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

    psf = airy / np.sum(airy)
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
    cell = rescale_intensity(cell, out_range=(0, 1))
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

    return cell.astype('float32'), tophat_cell.astype('float32'), mask_cropped, deconv_cell.astype('float32')

def determine_removal_mask(
    tophat_cell: np.ndarray,
    cell: np.ndarray,
    cell_mask: np.ndarray,
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

    labeled_regions, max_lbl = get_largest_signal_regions(tophat_cell, cell, cell_mask)

    if max_lbl is None:
        return None

    removal_mask = remove_metaphase_plate(max_lbl, labeled_regions, cell, config)
    props = regionprops(label(removal_mask))
    width = props[0].axis_minor_length if len(props)>0 else 0.0

    return removal_mask, width


def get_largest_signal_regions(
    tophat_cell: np.ndarray, cell: np.ndarray, cell_mask: np.ndarray
) -> tuple[np.ndarray, Optional[int]]:
    """
    Segments the cell and returns the brightest region.
    -----------------------------------------------------------
    INPUTS:
        tophat_cell: np.ndarray, top-hat filtered cell image
        cell: np.ndarray, cropped chromatin image
    OUTPUTS:
        labeled: np.ndarray, labeled image with background = 0
        max_intensity_lbl: int or None, label of the brightest region
    """
    cell_in_mask = np.copy(tophat_cell)
    cell_in_mask[~cell_mask] = 0

    thresh = threshold_otsu(cell_in_mask[cell_in_mask > 0])
    thresh_cell = cell_in_mask > thresh
    labeled, num_labels = label(thresh_cell, return_num=True)

    if num_labels == 0:
        return labeled, None

    labels = np.linspace(1, num_labels, num_labels).astype(int)
    region_intensities = [np.nansum(cell[labeled == (lbl)]) for lbl in labels]

    max_intensity = max(region_intensities)
    max_intensity_lbl = labels[region_intensities.index(max_intensity)]

    return labeled, max_intensity_lbl


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

    # 2. Geometric Scaling Logic (The "sqrt(2)" Rule)
    # Convert to float for precision during scaling
    rect_coords_ori = np.asarray(rect_coords_ori)
    rect_transformed = np.zeros_like(rect_coords_ori)
    rect_transformed[:, 0] = rect_coords_ori[:, 1]  # swap (x, y) to (y, x)
    rect_transformed[:, 1] = rect_coords_ori[:, 0]  # swap (x, y) to (y, x)
    rect_transformed = np.asarray(rect_transformed, dtype=np.float32)

    
    # Calculate the centroid of the LIR
    centroid = np.mean(rect_transformed, axis=0)
    
    # This transforms the LIR into the Bounding Box of the inferred ellipse
    truncate_z = config.truncate_z
    scale_factor = 1.56*np.sqrt(2)
    scaled_pts = centroid + (rect_transformed - centroid) * scale_factor
    
    # 3. Create the mask of the scaled rectangle
    # OpenCV fillPoly expects (x, y) in int32
    scaled_pts_int = scaled_pts.astype(np.int32)
    
    scaled_rect_mask = np.zeros_like(labeled, dtype=np.uint8)
    cv2.fillPoly(scaled_rect_mask, [scaled_pts_int], 1)
    final_plate_mask = scaled_rect_mask.astype(bool)

    return final_plate_mask


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

def segment_mask_unaligned(
    removal_mask: np.ndarray,
    tophat_cell: np.ndarray,
    cell_mask: np.ndarray,
    config: Optional[ChromatinSegConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:

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

    # 1. Apply cell mask immediately using a view to save memory
    # We only care about pixels where cell_mask is True
    cell_pixels = tophat_cell[cell_mask]
    
    if cell_pixels.size == 0:
        return np.zeros_like(tophat_cell, dtype=int), np.zeros_like(tophat_cell, dtype=bool)

    # 2. Initial validation threshold
    global_thresh = threshold_li(cell_pixels[cell_pixels > 0])
    validation_mask = (tophat_cell > global_thresh) & cell_mask
    
    # Check if chromatin exists outside the removal mask
    # Logical AND NOT is faster than copying and zeroing out
    chromatin_outside = validation_mask & ~removal_mask
    
    if np.sum(chromatin_outside) < 10:
        total_chromatin_mask = validation_mask
        # No unaligned chromosomes to label in this branch based on logic
        labeled = np.zeros_like(tophat_cell, dtype=int)
    else:
        
        # 3. Targeted thresholding: exclude the metaphase plate area from the calculation
        # to find dimmer unaligned fragments
        target_indices = cell_mask & ~removal_mask
        target_pixels = tophat_cell[target_indices]
        
        # Recalculate threshold specifically for the "unaligned" zone
        unaligned_thresh = threshold_li(target_pixels[target_pixels > 0])
        
        # Create masks
        thresh_cell = (tophat_cell > unaligned_thresh) & target_indices
        total_chromatin_mask = (tophat_cell > unaligned_thresh) & cell_mask
        
        # 4. Clean up labeled objects
        labeled = label(thresh_cell)
        if config.min_chromatin_area > 0:
            labeled = remove_small_objects(labeled, min_size=config.min_chromatin_area)

    # 5. Consistent Post-processing
    # Apply border clearing to both outputs to ensure consistency
    border_mask = binary_erosion(cell_mask, disk(1))
    labeled = clear_border(labeled, mask=border_mask)
    total_chromatin_mask = clear_border(total_chromatin_mask.astype(bool), mask=border_mask)

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
    frames_data = (
        analysis_df[analysis_df["particle"] == identity]
        .sort_values("frame")
    )

    # Preallocate accumulators
    u_area_trace = []
    u_area_int_trace = []
    u_num_trace = []
    t_area_trace = []
    t_area_int_trace = []
    width_trace = []
    visualization_stacks = []
    successful_removal_frames = []

    for _, row in frames_data.iterrows():
        f = int(row["frame"])
        l = int(row["label"])
        semantic = int(row["semantic_smoothed"])

        # Default values (non-mitotic frame)
        u_area = 0
        u_area_int = 0
        u_num = 0
        t_area = 0
        t_area_int = 0.0
        width = 0

        if semantic == 1:
            cell, tophat_cell, cell_mask, _ = compute_cell_images(
                instance, chromatin, f, l, zoom_factor, config
            )

            removal_mask, width = determine_removal_mask(
                tophat_cell, cell, cell_mask, config
            )

            if removal_mask is None:
                print(f"Lost track of cell {identity}, moving on")
                return None

            successful_removal_frames.append(int(np.any(removal_mask)))

            (
                u_area,
                u_area_int,
                u_num,
                t_area,
                t_area_int,
            ), labeled_chromosomes = segment_unaligned_chromosomes(
                cell, tophat_cell, removal_mask, cell_mask, config
            )

            crop_stack = save_chromatin_crops(
                cell,
                tophat_cell,
                cell_mask,
                labeled_chromosomes,
                removal_mask,
            )
            visualization_stacks.append(crop_stack)

        # Append results directly (no zip later)
        u_area_trace.append(u_area)
        u_area_int_trace.append(u_area_int)
        u_num_trace.append(u_num)
        t_area_trace.append(t_area)
        t_area_int_trace.append(t_area_int)
        width_trace.append(width)

    removal_freq = (
        np.mean(successful_removal_frames)
        if successful_removal_frames
        else 0.0
    )

    return (
        u_area_trace,
        u_area_int_trace,
        u_num_trace,
        t_area_trace,
        t_area_int_trace,
        width_trace,
        visualization_stacks,
        removal_freq,
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

    minors = []
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
        #axes[row, 0].set_xticks([])
        #axes[row, 0].set_yticks([])

        # Column 2: Deconvolved
        axes[row, 1].imshow(deconv_cell, cmap="gray")
        #axes[row, 1].set_xticks([])
        #axes[row, 1].set_yticks([])

        # Column 3: Top-hat filtered
        axes[row, 2].imshow(tophat_cell, cmap="gray")
        #axes[row, 2].set_xticks([])
        #axes[row, 2].set_yticks([])

        # Column 4: Metaphase plate mask
        removal_mask, width = determine_removal_mask(tophat_cell, cell, cell_mask, config)
        if removal_mask is not None:
            minors.append(width)

            axes[row, 3].imshow(tophat_cell, cmap="gray")

            # Get the original region mask for comparison
            labeled_regions, max_lbl = get_largest_signal_regions(tophat_cell, cell, cell_mask)
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
    avg_plt_width = np.nanmean(minors)
    fig.suptitle(
        f"Chromatin Processing Pipeline - Cell {identity}), 'Average plate width {avg_plt_width}",
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


    if len(analysis_paths) == 0:
        print("No data files found")
        sys.exit()

    # Load data for the first position
    analysis_df = pd.read_excel(analysis_paths[0])
    instance = tiff.imread(instance_paths[0])
    chromatin = tiff.imread(chromatin_paths[0])

    # Clean the data
    analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    analysis_df.dropna(inplace=True)

    # Get available cells
    available_cells = analysis_df["particle"].unique()

    if len(available_cells) == 0:
        print('No cells to analyze')
        sys.exit()

    valid_cells = []
    for cell_id in available_cells:

        intensity = analysis_df.query(f"particle=={cell_id}")["GFP"].to_numpy()
        shading = analysis_df.query(f"particle=={cell_id}")["GFP_int_corr"].to_numpy()
        offset = analysis_df.query(f"particle=={cell_id}")["offset"].to_numpy()
        bkg = analysis_df.query(f"particle=={cell_id}")["GFP_bkg_corr"].to_numpy()
        dead = analysis_df.query(f"particle=={cell_id}")["dead_flag"].to_numpy()
        semantic_smoothed = analysis_df.query(f"particle=={cell_id}")["semantic_smoothed"].to_numpy()
        dead_score = np.sum(semantic_smoothed * dead)
        corr_intensity = ((intensity - bkg) * shading) - offset

        _, range_crit, _ = validate_cyclin_b_trace(corr_intensity)

        metaphase_frames = analysis_df.query(
            f"particle == {cell_id} and semantic_smoothed == 1"
        )

        valid = range_crit and dead_score <= 5 and np.sum(semantic_smoothed) >=5

        if valid:
            valid_cells.append(cell_id)
    

    print(f'Will visualize {len(valid_cells)} cells')
    for cell_id in valid_cells:

        metaphase_frames = analysis_df.query(
            f"particle == {cell_id} and semantic_smoothed == 1"
        )

        print(f"Working on cell {cell_id} from position {positions[0]}")
        print(f"Found {len(metaphase_frames)} metaphase timepoints")

        # Create save path
        save_dir = os.path.dirname(analysis_paths[0])
        save_path = os.path.join(
            save_dir, f"{positions[0]}_cell_{cell_id}_processing.png"
        )

        config = ChromatinSegConfig(top_hat_radius = 15, truncate_z = 10e5) #this may be edited

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