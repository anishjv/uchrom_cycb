import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import zoom

from typing import Tuple, List, Optional

import skimage
from skimage.morphology import (
    binary_dilation,
    disk,
    square,
    remove_small_holes,
    remove_small_objects
)
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.restoration import unsupervised_wiener, richardson_lucy
import warnings
import tempfile
import tifffile as tiff
import os 
import subprocess


def adjust_zoom_factor(chromatin_shape: tuple[int, int, int], instance_shape: tuple[int, int, int]) -> float:
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
        assert chromatin_shape[1] / instance_shape[1] == chromatin_shape[2] / instance_shape[2]
        return chromatin_shape[2] / instance_shape[2]
    except AssertionError:
        raise ValueError("Chromatin and Instance must be square arrays")


def prepare_cell_image(chromatin: np.ndarray, frame: int, bbox_coords: tuple[int, int, int, int]) -> np.ndarray:
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
    nobkg_cell = skimage.morphology.white_tophat(cell, disk(5))
    return cell, gaussian(nobkg_cell, sigma=1.5)


def get_largest_signal_regions(nobkg_cell, cell: np.ndarray, num_regions: int = 1) -> tuple[list[regionprops], np.ndarray]:
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
    thresh = threshold_otsu(nobkg_cell) #
    labeled, num_labels = label(nobkg_cell > thresh, return_num=True, connectivity=1)
    labels = np.linspace(1, num_labels, num_labels).astype(int)
    region_intensities = [np.nansum(cell[labeled == (lbl)]) for lbl in labels]
    max_intensity = max(region_intensities)
    max_intensity_lbl = labels[region_intensities.index(max_intensity)]
    
    if len(region_intensities) > 1:
        second_max_intensity = sorted(region_intensities)[-2]
        nxt_max_intensity_lbl = labels[
                    region_intensities.index(second_max_intensity)
                ]
        
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


def remove_metaphase_if_eccentric(lbl:int, labeled: np.ndarray) -> np.ndarray:
    """
    Removes the metaphase plate only if its eccentricity exceeds threshold.
    ---------------------------------------------------------------------------------------------------------------
    INPUTS:
        region: regionprops
        labeled: np.ndarray
    OUTPUTS:
        removal_mask: np.ndarray
    """
    region_mask = np.zeros_like(labeled, dtype=bool)
    region_mask[labeled == lbl] = 1
    eccentricity = regionprops(label(region_mask.astype(int)))[0].eccentricity
    if eccentricity > 0.7:
        print('metaphase; removing plate')
        return binary_dilation(region_mask, disk(9))
    else:
        print('metaphase; NOT removing plate')
        return np.zeros_like(labeled, dtype=bool)


def segment_unaligned_chromosomes(cell: np.ndarray, removal_mask: np.ndarray, min_area: int) -> tuple[int, int, int]:
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
    perfect_psf = np.zeros((19, 19))
    perfect_psf[9, 9] = 1
    psf = gaussian(perfect_psf, 2)
    deconv_cell = unsupervised_wiener(cell, psf, clip=False)[0]
    cell_minus_struct = np.copy(deconv_cell)
    cell_minus_struct[removal_mask] = 0
    
    thresh = threshold_otsu(cell_minus_struct)
    labeled, num_labels = label(cell_minus_struct > thresh, return_num=True, connectivity=1)
    labels = np.linspace(1, num_labels, num_labels).astype(int)
    labeled = clear_border(labeled)
    
    areas, intensities = [], []
    for lbl in labels:
        area = np.nansum(labeled[labeled == lbl])
        if area >= min_area:
            intensity = np.nansum(cell[labeled==lbl])
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
    
    return np.nansum(mask),np.nansum(cell[mask]), mean


def unaligned_chromatin(
    identity: int,
    analysis_df: pd.DataFrame,
    instance: np.ndarray,
    chromatin: np.ndarray,
    min_chromatin_area: Optional[int] = 4
) -> tuple[list[int], list[int], list[float], list[int], list[float], list[int], int]:
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
    OUTPUTS:
        area_signal: List[int]
        intensity_signal: List[int]
        whole_cell_intensity: List[float]
        num_signals: List[int]
        whole_cell_avg_intensity: List[float]
        whole_cell_area: List[int]
        first_anaphase: int
    """
    zoom_factor = adjust_zoom_factor(chromatin.shape, instance.shape)
    frames_data = analysis_df.query(f"particle == {identity}")
    semantics = frames_data["semantic_smoothed"].tolist()

    results = []
    anaphase_indices = []
    print(f'Working on cell {identity}')
    
    for idx, row in frames_data.iterrows():
        f, l, semantic = int(row["frame"]), int(row["label"]), int(row["semantic_smoothed"])

        mask = instance[f] == l
        if zoom_factor != 1:
            mask = zoom(mask, zoom_factor, order=0)
        mask = binary_dilation(mask, disk(3))

        bbox_coords = bbox(mask)
        cell, nobkg_cell = prepare_cell_image(chromatin, f, bbox_coords)

        if semantic == 1:
            labeled_regions, max_lbl, second_lbl, intensity_diff_ratio = get_largest_signal_regions(nobkg_cell, cell, num_regions=2)

            if second_lbl:
                to_check = idx+9 if (idx+9) < len(semantics) else -1
                near_end_of_mitosis = any(s == 0 for s in semantics[idx:to_check]) 
                
                if intensity_diff_ratio < (1 / 3) and near_end_of_mitosis:
                    print('anaphase; removing blobs')
                    removal_mask = remove_regions([max_lbl, second_lbl], labeled_regions)
                    anaphase_indices.append(idx)
                else:
                    removal_mask = remove_metaphase_if_eccentric(max_lbl, labeled_regions)
            else:
                removal_mask = remove_metaphase_if_eccentric(max_lbl, labeled_regions)

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

            area_sig, int_sig, num_sig = segment_unaligned_chromosomes(cell, removal_mask, min_chromatin_area)
        else:
            area_sig, int_sig, num_sig = 0, 0, 0

        whole_area, whole_intensity, whole_avg_intensity = measure_whole_cell(cell)

        results.append((area_sig, int_sig, whole_intensity, num_sig, whole_avg_intensity, whole_area))
        
    area_signal, intensity_signal, whole_cell_intensity, num_signals, whole_cell_avg_intensity, whole_cell_area = zip(*results)

    return (
        list(area_signal),
        list(intensity_signal),
        list(whole_cell_intensity),
        list(num_signals),
        list(whole_cell_avg_intensity),
        list(whole_cell_area),
        first_anaphase,
    )

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


"""
Relevant to morphology based identification unaligned chromosomes (above)
---------------------------------------------------------------------
Relevant to ilastik based identificaiton unaligned chromosomes (below)
"""

def remove_border_objs(img:npt.NDArray):
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
    
    border_objs = np.logical_xor(cleared_labels>0, binary)
    border_objs = binary_dilation(border_objs, disk(9))
    
    bkg_pixels = img <= thresh
    bkg = np.median(img[bkg_pixels>0])
    
    cleared_img = np.copy(img)
    cleared_img[border_objs>0] = bkg
    
    return cleared_img

def run_ilastik_arrays(
    image_arrays: List[np.ndarray],
    ilastik_project: str,
    export_source: str = "Simple Segmentation",
    ilastik_executable: str = "/sw/pkgs/arc/ilastik/1.4.0/run_ilastik.sh"
) -> List[np.ndarray]:
    """
    Run Ilastik in headless mode on a list of NumPy arrays.

    Parameters:
        image_arrays (List[np.ndarray]): List of 2D grayscale images as NumPy arrays.
        ilastik_project (str): Path to the Ilastik .ilp project file.
        export_source (str): Ilastik export source ("Simple Segmentation", etc.).
        ilastik_executable (str): Path to Ilastik's run_ilastik.sh executable.

    Returns:
        List[np.ndarray]: List of output arrays from Ilastik (one per input).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save all input arrays as TIFFs
        input_paths = []
        for i, arr in enumerate(image_arrays):
            path = os.path.join(tmpdir, f"input_{i}.tiff")
            tiff.imsave(path, arr.astype(np.uint16))
            input_paths.append(path)

        # Define output path template using {nickname}
        output_template = os.path.join(tmpdir, "{nickname}_seg.tiff")

        # Construct and run Ilastik command
        cmd = [
            ilastik_executable,
            "--headless",
            f"--project={ilastik_project}",
            f"--export_source={export_source}",
            f"--output_filename_format={output_template}",
            "--output_format=tiff",
            "--raw_data"
        ] + input_paths

        subprocess.run(cmd, check=True)

        # Load all output images back as NumPy arrays
        output_arrays = []
        for path in input_paths:
            base = os.path.splitext(os.path.basename(path))[0]  # e.g., input_0
            output_file = os.path.join(tmpdir, f"{base}_seg.tiff")
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"Missing output: {output_file}")
            output_arrays.append(tiff.imread(output_file))

        return output_arrays
    

def ilastik_unaligned_chromatin(
    identity: int,
    analysis_df: pd.DataFrame,
    instance: np.ndarray,
    chromatin: np.ndarray,
    ilastic_project_path: str,
    min_chromatin_area: Optional[int] = 16
) -> tuple[list[int], list[int], list[float], list[int], list[float], list[int], int]:


    zoom_factor = adjust_zoom_factor(chromatin.shape, instance.shape)
    frames_data = analysis_df.query(f"particle == {identity}")
    print(f'Working on cell {identity}')
    
    cells_fillers = []
    for _, row in frames_data.iterrows():
        f, l, semantic = int(row["frame"]), int(row["label"]), int(row["semantic_smoothed"])

        if semantic == 1:
            mask = instance[f] == l
            if zoom_factor != 1:
                mask = zoom(mask, zoom_factor, order=0)
            mask = binary_dilation(mask, square(10))

            bbox_coords = bbox(mask)
            rmin, rmax, cmin, cmax = bbox_coords
            cell = chromatin[f, rmin:rmax, cmin:cmax]
            cell = remove_border_objs(cell)

            perfect_psf = np.zeros((19, 19))
            perfect_psf[9, 9] = 1
            psf = gaussian(perfect_psf, 2)
            cell = richardson_lucy(cell/cell.max(), psf, 20)
            cells_fillers.append(cell)

        else:
            cells_fillers.append(None)

    cells = [obj for obj in cells_fillers if isinstance(obj, np.ndarray)]
    segmentations = run_ilastik_arrays(cells, ilastic_project_path)

    manual_count = 0
    results = []
    for obj in cells_fillers:

        if isinstance(obj, np.ndarray):
            print(obj.shape)
            segmentation = segmentations[manual_count]
            print(segmentation.shape)

            unaligned_chromatin = np.copy(segmentation)
            unaligned_chromatin[unaligned_chromatin != 3] = 0
            unaligned_chromatin = unaligned_chromatin > 0
            unaligned_chromatin = remove_small_holes(unaligned_chromatin, area_threshold=5) #should update based on resolution
            unaligned_chromatin = remove_small_objects(unaligned_chromatin, min_size = min_chromatin_area)
            print(unaligned_chromatin.shape)

            metaphase_plate = np.copy(segmentation)
            metaphase_plate[metaphase_plate != 2] = 0
            metaphase_plate = metaphase_plate > 0
            metaphase_plate = remove_small_holes(metaphase_plate, area_threshold=15)

            area_sig = np.nansum(unaligned_chromatin)
            intensity_sig = np.nansum(
                obj[unaligned_chromatin>0]
            )

            metaphase_sig = np.nansum(metaphase_plate)
            metaphase_intensity_sig = np.nansum(
                obj[metaphase_plate>0]
            )
            
            _, num_sig= label(unaligned_chromatin, return_num=True)

            manual_count += 1

        else:
            area_sig, intensity_sig, num_sig, metaphase_sig, metaphase_intensity_sig = 0, 0, 0, 0, 0

        results.append((area_sig, intensity_sig, num_sig, metaphase_sig, metaphase_intensity_sig))
        
    area_sigs, intensity_sigs, num_sigs, metaphase_sigs, metaphase_intensity_sigs = zip(*results)


    return (
    list(area_sigs),
    list(intensity_sigs),
    list(num_sigs),
    list(metaphase_sigs),
    list(metaphase_intensity_sigs),
    )

    