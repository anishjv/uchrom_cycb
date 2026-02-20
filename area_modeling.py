# Utilities for estimating the number of unaligned chromosomes from unaligned chromatin area measurements

import numpy.typing as npt
from typing import Optional
import numpy as np
import skimage
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, binary_dilation, disk
from skimage.measure import label, regionprops
from skimage.draw import polygon
from skimage.restoration import richardson_lucy
from skimage.transform import rescale
from skimage.exposure import rescale_intensity
from typing import Optional


import sys
sys.path.append('/Users/whoisv/')
sys.path.append('/Users/whoisv/uchrom_cycb/')
from uchrom_cycb.segment_chromatin import determine_removal_mask, airy_disk_psf, segment_unaligned_chromosomes, ChromatinSegConfig
from uchrom_cycb.degradation import *


def psuedo_cellapp_mask(
    img:npt.NDArray, 
    config:Optional[ChromatinSegConfig] = None
    ) -> npt.NDArray:

    '''
    Creates a square cell mask based on the location 
    and orientation of the cell's largest histone region
    ------------------------------------------------------
    INPUTS:
        img: np.ndarray, img to create a mask for
        config: ChromatinSegConfig, dictionary of standard configurations
    OUTPUTS:
        bbox_mask: square mask surroundng cell's histone signal
    '''

    if config is None:
        config = ChromatinSegConfig()

    #naive segmentation
    thresh = threshold_otsu(img)
    seg = img > thresh

    #remove objects touching the edges
    seg_cleaned = clear_border(seg)
    seg_cleaned = remove_small_objects(seg_cleaned, config.min_chromatin_area)
    seg_cleaned_labeled, num_labels = label(seg_cleaned, return_num=True)

    labels = np.linspace(1, num_labels, num_labels).astype(int)
    areas = [np.nansum(seg_cleaned[seg_cleaned_labeled == (lbl)]) for lbl in labels]

    max_area = max(areas)
    max_area_lbl = labels[areas.index(max_area)]
    seg_max = seg_cleaned_labeled == max_area_lbl

    # region properties of largest object
    props = regionprops(seg_max.astype(int))[0]

    theta = props.orientation        # radians
    cy, cx = props.centroid
    maj = (props.major_axis_length / 2)*1.25
    minr = maj

    # unit vectors along major and minor axes (row, col)
    u_major = np.array([ np.cos(theta),  np.sin(theta)])
    u_minor = np.array([-np.sin(theta),  np.cos(theta)])

    # rectangle corners (row, col)
    corners = np.array([
        [cy, cx] +  maj*u_major +  minr*u_minor,
        [cy, cx] +  maj*u_major -  minr*u_minor,
        [cy, cx] -  maj*u_major -  minr*u_minor,
        [cy, cx] -  maj*u_major +  minr*u_minor,
    ])

    # rasterize rectangle into mask
    rr, cc = polygon(corners[:, 0], corners[:, 1], seg_max.shape)
    bbox_mask = np.zeros_like(seg_max, dtype=bool)
    bbox_mask[rr, cc] = True
    bbox_mask = binary_dilation(bbox_mask, disk(3))

    return bbox_mask

def degrade_to_ixn(
    hq_stack: npt.NDArray, 
    current_ps:float=0.05, 
    target_ps:float=0.3387
    ) -> npt.NDArray:

    '''
    Degrades Z-stacks taken on the Joglekar Lab's confocal microscope 
    to appear as if they were taken with a 0.45 NA, 20X objective. 
    Note: Z-stacks must have depth greater than focal depth of images 
    taken with the target (low resolution) objective.
    -------------------------------------------------------------------
    INPUTS:
        hq_stack: np.array, 3D Z-stack taken with a high resolution objective
        current_ps: float, XY pixel size in microns of input stack 
        target_ps: float, XY pixel size in microns of target stack 
    OUPUTS:
        matched_image: np.array, 2D degraded image
    '''


    # 1. Apply weighted projection (returns float32)
    # n=1.33 is correct for immersion, but check if your 20X is a "Dry" lens (n=1.0)
    lq_stack = sinc_squared_weighting(hq_stack, 0.17, 0.45, 0.624, n=1.33)
    lq_image = np.max(lq_stack, axis=0)
    
    # 2. Explicitly scale intensities to [0, 1] for the deconvolution math
    # This replaces the need for the .astype('uint16') hack
    lq_image = rescale_intensity(lq_image, out_range=(0, 1))
    lq_smooth_image = gaussian(lq_image, sigma = 7) #sigma computed from rayleigh criterion: sigma ~ [ 0.61 (0.624)/(0.46) ]/2.355 um
    
    # 3. Calculate scaling factor
    scale = current_ps / target_ps 
    # 4. Apply Anti-Aliasing and Rescale
    matched_image = rescale(
        lq_smooth_image, 
        scale, 
        anti_aliasing=True, 
        preserve_range=True, # Keeps the 0-1 range we just set
        order=1 
    )
    
    return matched_image.astype('float32')


def sinc_squared_weighting(
    stack: npt.NDArray,
    z_microns_per_slice: float,
    na: float,
    wavelength_um:float,
    n: Optional[float]=1.33
    ) -> npt.NDArray:

    '''
    Weights a Z-stack by [sin(x)/x]^2 to model 
    the focal volume of a given NA objective
    ---------------------------------------------
    INPUTS:
        stack: np.array, stack to be weighted
        z_microns_per_slice: float, Z pixel size in microns of stack
        na: float, NA of objective whose focal depth is to be modeled
        wavelength_um: float, emission wavelength at which image was acquired
        n: float, refractive index of the used imaging media
    OUTPUTS:
        weighted_stack: np.array, weighted stack
    '''

    nz = stack.shape[0]
    center_idx = nz // 2
    z = (np.arange(nz) - center_idx) * z_microns_per_slice
    
    # The optical coordinate 'u'
    # u = (2 * pi / lambda) * z * (NA^2 / n)
    u = (2 * np.pi * na**2 * z) / (n * wavelength_um)
    
    # I(z) = [sin(u/4) / (u/4)]^2
    # We use np.sinc, but note: np.sinc(x) is sin(pi*x)/(pi*x)
    arg = u / (4 * np.pi)
    weights = np.sinc(arg)**2
    
    weighted_stack = stack.astype(np.float32) * weights[:, np.newaxis, np.newaxis]

    return weighted_stack

def compute_cell_images(
    cell: npt.NDArray,
    config: Optional[ChromatinSegConfig] = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

    '''
    Analog of compute_cell_images() in segment_chromatin.py
    -------------------------------------------------------
    INPUTS:
        cell: np.array, image to be processed
        config: ChromatinSegConfig, dictionary of standard configurations
    OUTPUTS:
        cell: np.array, original (input) image
        tophat_cell: np.array, image after deconvolution and tophat filtering
        deconv_cell: np.array, image after deconvolution (no tophat filtering)
    '''

    if config is None:
        config = ChromatinSegConfig()

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

    return cell, tophat_cell, deconv_cell


def read_chrom_num(
    seg: npt.NDArray, 
    cell_mask: npt.NDArray, 
    mtphs_plate: npt.NDArray, 
    min_chrom_vol: Optional[int] = 500, 
    max_plate_overlap: Optional[float] = 0.25,
    min_cell_overlap: Optional[float] = 0.05,
    config: Optional[ChromatinSegConfig] = None
    ) -> tuple[int, npt.NDArray, list[float]]:

    '''
    Reads in the number of unaligned chromosomes  
    from a FAST-CHIMP segmentation
    --------------------------------------------
    INPUTS:
        seg: np.array, FAST-CHIMP segmentation
        cell_mask: np.array, segmentation mask from psuedo_cellapp_mask()
        mtphs_plate: np.array, computed metaphase plate (see segment_chromatin.py)
        min_chrom_volume: Optional[int]: minimum volume for a found object to be considered a chromosome
        max_plate_overlap: Optional[float]: maximum overlap with metaphase plate for a chromsome to be considered unaligned
        min_cell_overlap: Optional[float]: minimum overlap with cell mask for chromosome to be considered in the imaging volume
        config: ChromatinSegConfig: dictionary of common configuations
    OUTPUTS:
    '''

    if config is None:
        config = ChromatinSegConfig()

    #assumes all inputs are square 
    seg = seg.astype('int32')
    scale = seg.shape[1] / mtphs_plate.shape[1]

    cell_mask = rescale(
                    cell_mask, 
                    scale, 
                    anti_aliasing=False, 
                    preserve_range=True,
                    order=0 
                ).astype('bool')
    cell_mask = np.repeat(cell_mask[np.newaxis, ...], seg.shape[0], axis=0)

    truncate_z = config.truncate_z
    if truncate_z is not None and truncate_z < 0.31875*seg.shape[0]:
        frames_to_keep = int(round(truncate_z / 0.31875))
        frames_to_trunc = (cell_mask.shape[0] - frames_to_keep) // 2
        cell_mask[:frames_to_trunc] = 0
        cell_mask[-frames_to_trunc:] = 0

    mtphs_plate = rescale(
                mtphs_plate, 
                scale, 
                anti_aliasing=False, 
                preserve_range=True,
                order=0
            ).astype('bool')
    mtphs_plate = np.repeat(mtphs_plate[np.newaxis, ...], seg.shape[0], axis=0)

    seg = clear_border(seg, buffer_size=10)

    props = regionprops(seg)
    chroms = np.zeros_like(seg)
    areas = []
    for prop in props:

        coords = prop.coords

        # Fraction inside metaphase plate
        mp_overlap = np.sum(
            mtphs_plate[
                coords[:, 0],
                coords[:, 1],
                coords[:, 2]
            ] > 0
        ) / prop.area

        # Fraction inside cell mask
        cell_overlap = np.sum(
            cell_mask[
                coords[:, 0],
                coords[:, 1],
                coords[:, 2]
            ] > 0
        ) / prop.area

        # Keep if mostly inside cell and mostly outside metaphase plate
        if (
            mp_overlap < max_plate_overlap and
            cell_overlap >= min_cell_overlap
        ):

            chroms[
                coords[:, 0],
                coords[:, 1],
                coords[:, 2]
            ] = prop.label

            areas.append(prop.area)

    chroms = remove_small_objects(chroms, min_size=min_chrom_vol)

    return len(np.unique(chroms)) - 1, chroms, areas


def run_pipeline(
    seg_3d: npt.NDArray, 
    cell_3d: npt.NDArray, 
    config: Optional[ChromatinSegConfig] = None
    ):

    '''
    Given a high resolution Z stack (H2B-visualizing) 
    and FAST-CHIMP segmenation, computes the unaligned chromatin 
    area that you would measure with a lower resolution objective and 
    reads the number of unaligned chrmosomes in the FAST-CHIMP segmenation.
    ----------------------------------------------------------------------
    INPUTS:
        seg_3d: np.array, FAST-CHIMP segmentation
        cell_3d: p.array, 3D Z-stack taken with a high resolution objective
        config: ChromatinSegConfig, dictionary of common configurations
    OUPUTS:
        for_vals: list, contains following quantities:
                            1. unaligned chromatin area
                            2. number of unlaigned chromosomes
                            3. FAST-CHIMP area of each unaligned chromosome
                            4. metaphase plate width
        for_ims: list, contains following images:
                            1. degraded 3D Z-stack
                            2. segmented unaligned chromatin
                            3. metaphase plate
                            4. cell mask
                            5. FAST-CHIMP unaligned chromosomes

    '''

    if config is None:
        config = ChromatinSegConfig()


    cell = degrade_to_ixn(cell_3d)
    cell, tophat_cell, deconv_cell = compute_cell_images(cell) 
    mask = psuedo_cellapp_mask(tophat_cell, config)
    metphs_plt, width = determine_removal_mask(tophat_cell, cell, mask, config)

    measurements, uchroms = segment_unaligned_chromosomes(
        cell, tophat_cell, metphs_plt, mask, config
    )

    chrom_num, fast_uchroms, chrom_area = read_chrom_num(seg_3d, mask, metphs_plt, config=config) 

    for_vals = [
        measurements[0],
        chrom_num,
        chrom_area,
        width
        ]

    for_ims = [
        cell,
        uchroms,
        metphs_plt,
        mask,
        fast_uchroms,
    ]

    return for_vals, for_ims