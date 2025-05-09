import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.signal import find_peaks

from typing import Tuple, List, Optional

import skimage
from skimage.morphology import (
    binary_closing,
    white_tophat,
    binary_dilation,
    binary_erosion,

)
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, watershed
from skimage.feature import peak_local_max
import glob
import re
from sklearn.linear_model import LinearRegression
from segment_chromatin import unaligned_chromatin, bbox, ilastik_unaligned_chromatin #type:ignore


def retrieve_traces(
    analysis_df: pd.DataFrame,
    wl: str,
    frame_interval: int,
    clean: Optional[bool] = True,
) -> list[npt.NDArray]:
    """
    Retrieves traces for a given wavelength from the "analysis.xlsx" cellapp-analysis output spreadsheet
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        analysis_df: pd.DataFrame, analysis.xlsx output from https://github.com/ajitpj/cellapp-analysis
        wl: str
        clean_spurious: bool
    OUTPUTS:
        traces: list[npt.NDArray]
    """

    traces = []
    ids = []
    for id in analysis_df["particle"].unique():
        trace = analysis_df.query(f"particle=={id}")[[f"{wl}", "semantic_smoothed"]]
        bkg = analysis_df.query(f"particle=={id}")[f"{wl}_bkg_corr"]
        intensity = analysis_df.query(f"particle=={id}")[f"{wl}_int_corr"]

        trace = trace.to_numpy()

        if clean:
            t_char = 30 // frame_interval
            padded_semantic = np.append(trace[:, 1], np.zeros(3))
            _, props = find_peaks(padded_semantic, plateau_size=1)
            peak_widths = [
                width
                for i, width in enumerate(props["plateau_sizes"])
                if width >= t_char
            ]
            if len(peak_widths) == 1 and (np.sum(props["plateau_sizes"]) - peak_widths[0]) < t_char:
                if padded_semantic[0] != 1:
                    trace[:, 0] = (trace[:, 0] - bkg) * intensity
                    traces.append(trace)
                    ids.append(id)
        else:
            traces.append(trace)
            ids.append(id)

    return traces, ids



def qual_deg(traces: npt.NDArray, frame_interval: int) -> tuple[npt.NDArray]:
    """
    Returns the signal from some arbitrary fluorophore begining after a cell-aap mitosis call
    ------------------------------------------------------------------------------------------
    INPUTS:
        traces: npt.NDarray, output of retrieve_traces()
        frame_interval: int, time between successive frames
    OUTPUTS:
        intensity_container: npt.NDArray
        semantic_container: npt.NDArray
    """

    intensity_traces = []
    semantic_traces = []
    first_tp = []
    t_char = 30 // frame_interval
    for trace in traces:
        padded_semantic = np.append(trace[:, 1], np.zeros(3))
        peaks, props = find_peaks(padded_semantic, plateau_size=t_char)
        if props["plateau_sizes"][0] % 2:
            first_mitosis = peaks[0] - (props["plateau_sizes"][0] // 2)
        else:
            first_mitosis = peaks[0] - (props["plateau_sizes"][0] // 2 - 1)
        intensity_trace = trace[:, 0]
        semantic_trace = trace[:, 1]
        intensity_traces.append(intensity_trace)
        semantic_traces.append(semantic_trace)
        first_tp.append(first_mitosis)

    return intensity_traces, semantic_traces, first_tp



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


def extract_montages(
    identity: int,
    analysis_df: pd.DataFrame,
    instance: npt.NDArray,
    chromatin: npt.NDArray,
    cmap: Optional[str] = None,
    well_pos: Optional[str] = None,
    save_path: Optional[str] = None,
    mode: Optional[str] = "seg",
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
        save_path:str
    TODO:
        color masks by area extracted
    """

    if chromatin.shape != instance.shape:
        try:
            assert (
                chromatin.shape[1] / instance.shape[1]
                == chromatin.shape[2] / instance.shape[2]
            )
            zoom_factor = chromatin.shape[1] / instance.shape[1]
        except AssertionError:
            print("Chromatin and Instance must be square arrays")
            return

    rois = []
    frames = analysis_df.query(f"particle=={identity}")["frame"].tolist()
    markers = analysis_df.query(f"particle=={identity}")["label"].tolist()
    semantic = analysis_df.query(f"particle=={identity}")["semantic"].tolist()

    for index, zipped in enumerate(zip(frames, markers, semantic)):

        f, l, classifier = zipped
        # expand mask and capture indices
        mask = instance[f, :, :] == l
        if zoom_factor != 1:
            mask = ndi.zoom(mask, zoom_factor, order=0)
        zoom_mask = binary_dilation(mask, skimage.morphology.disk(3))
        rmin, rmax, cmin, cmax = bbox(zoom_mask)

        # trim cell and mask for efficiency
        cell = chromatin[f, rmin:rmax, cmin:cmax]
        nobkg_cell = white_tophat(cell, skimage.morphology.disk(5))
        nobkg_cell = gaussian(nobkg_cell, sigma=1.5)
        zoom_mask = zoom_mask[rmin:rmax, cmin:cmax]

        if mode != "cell":
            if classifier == 1:
                # find first aggresive threshold
                thresh = threshold_otsu(nobkg_cell)

                # threshold and label bkg subtracted cell
                thresh_cell, num_labels = label(
                    nobkg_cell > thresh, return_num=True, connectivity=1
                )
                labels = np.linspace(1, num_labels, num_labels).astype(int)
                # find the label corresponding to the maximum cummulative intensity image
                region_intensities = [
                    np.sum(cell[thresh_cell == (lbl)]) for lbl in labels
                ]
                max_intensity = max(region_intensities)
                max_intensity_lbl = labels[region_intensities.index(max_intensity)]
                if len(region_intensities) > 1:
                    second_max_intensity = sorted(region_intensities)[-2]
                    nxt_max_intensity_lbl = labels[
                        region_intensities.index(second_max_intensity)
                    ]
                    intensity_diff = (
                        max_intensity - second_max_intensity
                    ) / max_intensity

                # if we are two or one timepoint away from cytokensis
                index_to_check = (
                    index + 8 if len(semantic) > index + 8 else len(semantic) - 1
                )
                if (
                    semantic[index_to_check] == 1
                    and intensity_diff < (7 / 8)
                    and len(region_intensities) > 1
                ):
                    anaphase_blobs_mask = np.zeros_like(thresh_cell)
                    anaphase_blobs_mask[thresh_cell == max_intensity_lbl] = 1
                    anaphase_blobs_mask[thresh_cell == nxt_max_intensity_lbl] = 1
                    anaphase_blobs_mask = binary_dilation(
                        anaphase_blobs_mask, skimage.morphology.disk(9)
                    )
                    to_remove_mask = anaphase_blobs_mask
                    print("anaphase; removing blobs")

                else:
                    # grab the metaphase plate
                    metphs_plate_mask = np.copy(thresh_cell)
                    metphs_plate_mask[thresh_cell != max_intensity_lbl] = 0
                    # dilate the metaphase plate
                    metphs_plate_mask = binary_dilation(
                        metphs_plate_mask, skimage.morphology.disk(9)
                    )
                    if np.sum(metphs_plate_mask) != 0:
                        eccen = regionprops(label(metphs_plate_mask))[0]["eccentricity"]
                        if eccen < 0.7:
                            metphs_plate_mask = np.zeros_like(metphs_plate_mask)
                            print("metaphase; not removing metaphase plate")
                        else:
                            print("metaphase; removing")
                    to_remove_mask = metphs_plate_mask
            else:
                to_remove_mask = np.zeros_like(cell)

            # deconvolve and create cell image with metaphase plate removed
            perfect_psf = np.zeros((25, 25))
            perfect_psf[12, 12] = 1
            psf = gaussian(perfect_psf, 2)
            deconv_cell = skimage.restoration.unsupervised_wiener(
                cell, psf, clip=False
            )[0]
            cell_minus_struct = np.copy(deconv_cell)
            cell_minus_struct[to_remove_mask] = 0

            # threshold cell with metaphase plate removed
            thresh2 = threshold_otsu(cell_minus_struct)
            thresh_cell2, num_labels2 = label(
                cell_minus_struct > thresh2, return_num=True, connectivity=1
            )
            thresh_cell2 = clear_border(thresh_cell2)
            labels2 = np.linspace(1, num_labels2, num_labels2).astype(int)

            borders = binary_dilation(thresh_cell2) ^ binary_erosion(thresh_cell2)
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


def fit_three_lines(x, y):
    best_score = float('inf')
    best_breaks = None
    best_models = None

    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Try all valid pairs of breakpoints: i < j
    for i in range(2, len(x) - 4):
        for j in range(i + 2, len(x) - 2):
            x1, y1 = x[:i], y[:i]
            x2, y2 = x[i:j], y[i:j]
            x3, y3 = x[j:], y[j:]

            x1r, x2r, x3r = x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1)

            model1 = LinearRegression().fit(x1r, y1)
            model2 = LinearRegression().fit(x2r, y2)
            model3 = LinearRegression().fit(x3r, y3)

            y1_pred = model1.predict(x1r)
            y2_pred = model2.predict(x2r)
            y3_pred = model3.predict(x3r)

            ssr = np.sum((y1 - y1_pred) ** 2) + np.sum((y2 - y2_pred) ** 2) + np.sum((y3 - y3_pred) ** 2)

            if ssr < best_score:
                best_score = ssr
                best_breaks = (i, j)
                best_models = (model1, model2, model3)

    return best_breaks, best_models
    

def fit_two_lines(x, y):
    
    best_break = None
    best_score = float('inf')
    best_models = (None, None)

    for i in range(2, len(x) - 2):
        x1, y1 = x[:i].reshape(-1, 1), y[:i]
        x2, y2 = x[i:].reshape(-1, 1), y[i:]

        model1 = LinearRegression().fit(x1, y1)
        model2 = LinearRegression().fit(x2, y2)

        y1_pred = model1.predict(x1)
        y2_pred = model2.predict(x2)

        ssr = np.sum((y1 - y1_pred) ** 2) + np.sum((y2 - y2_pred) ** 2)

        if ssr < best_score:
            best_score = ssr
            best_break = i
            best_models = (model1, model2)
     

    return best_break, best_models


def fit_cycb_regimes(x, y):
    
    best_breaks, best_models = fit_two_lines(x,y)
    
    # If x/y was short (len<2) error will occur
    try:
        eps = np.abs(best_models[0].coef_ - best_models[1].coef_)
    except AttributeError:
        return np.nan, (np.nan,)
    
    #If abs(slope) of first fit > abs(slope) of second fit => check for three regimes
    if np.abs(best_models[0].coef_) > abs(best_models[1].coef_):
        best_breaks, best_models = fit_three_lines(x,y)
        return best_breaks, best_models 
    
    #Must check for 3 regimes first, as
    #the slope of the two misfitted lines may have been arbitrarily similar
    #If slope of two fits is similar => we say that no slow regime occured (fit one line)
    if eps > np.abs(max([fit.coef_ for fit in best_models]))/5:
        pass
    else:
        x = x.reshape(-1, 1)
        best_models = (LinearRegression().fit(x,y) ,)
        best_breaks = np.nan
        
    
    return best_breaks, best_models


def cycb_chromatin_batch_analyze(positions:list, analysis_paths:list, instance_paths:list, chromatin_paths:list, piecewise_fit:bool
) -> tuple[pd.DataFrame]:

    for name_stub, analysis_path, instance_path, chromatin_path in zip(positions, analysis_paths, instance_paths, chromatin_paths):
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
        traces, ids = retrieve_traces(analysis_df, "GFP", 4)
        intensity, semantic, first_tps = qual_deg(traces, 4)

        if piecewise_fit:
            brk_pts = []
            fit_info = []
            for trace in intensity:
                front_end_chopped = 0
                mit_trace = trace[classification.iloc[i] ==1]
                front_end_chopped += (np.nonzero(semantic[i])[0][0] + 1)

                #forcing glob_min_index to be greater than glob_max_index
                glob_max_index = np.where(mit_trace == max(mit_trace))[0][0]
                glob_min_index = np.where(mit_trace == min(mit_trace[glob_max_index:]))[0][0]
                
                mit_neg_trace = mit_trace[glob_max_index:glob_min_index]
                front_end_chopped += glob_max_index + 1
                
                x = np.linspace(0, mit_neg_trace.shape[0]-1, mit_neg_trace.shape[0])
                brk_pt, fits = fit_cycb_regimes(x, mit_neg_trace)
                try:
                    slope_int = [(fit.coef_, fit.intercept_) for fit in fits]
                    brk_pts.append(brk_pt+front_end_chopped)
                except AttributeError:
                    slope_int = np.nan
                    brk_pts.append(np.nan)
                
                fit_info.append(slope_int)
        else:
            pass

        un_chromatin = []
        un_intensity = []
        tot_intensity = []
        un_number = []
        tot_avg_intensity = []
        tot_area = []
        first_anaphase = []
        unaligned_chromosomes = [
            unaligned_chromatin(identity, analysis_df, instance, chromatin)
            for i, identity in enumerate(ids)
        ]
        for i, data_tuple in enumerate(unaligned_chromosomes):
            un_chromatin.append(data_tuple[0])
            un_intensity.append(data_tuple[1])
            tot_intensity.append(data_tuple[2])
            un_number.append(data_tuple[3])
            tot_avg_intensity.append(data_tuple[4])
            tot_area.append(data_tuple[5])
            first_anaphase.append(data_tuple[6])

        cycb = pd.DataFrame(intensity)
        classification = pd.DataFrame(semantic)
        un_chromatin_area = pd.DataFrame(un_chromatin)
        un_chromatin_intensity = pd.DataFrame(un_intensity)
        un_chromatin_number = pd.DataFrame(un_number)
        total_intensity = pd.DataFrame(tot_intensity)
        total_avg_intensity = pd.DataFrame(tot_avg_intensity)
        total_area = pd.DataFrame(tot_area)

        d_temp = {
            "ids": ids,
            "first_mitosis": first_tps,
            "first_anaphase": first_anaphase,
            "raw_data_paths": [instance_path, analysis_path, chromatin_path],
        }
        other_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d_temp.items()]))

        save_dir = os.path.dirname(analysis_path)
        if not os.path.isdir(save_dir):
            save_dir = os.getcwd()
        save_path = os.path.join(save_dir, f"{name_stub}_cycb_chromatin.xlsx")

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            cycb.to_excel(writer, sheet_name="cycb")
            classification.to_excel(writer, sheet_name="classification")
            un_chromatin_area.to_excel(writer, sheet_name="unaligned chromatin area")
            un_chromatin_intensity.to_excel(
                writer, sheet_name="unaligned chromatin intensity"
            )
            total_intensity.to_excel(writer, sheet_name="total chromatin intensity")
            un_chromatin_number.to_excel(
                writer, sheet_name="number of unaligned chromosomes (approx.)"
            )
            total_avg_intensity.to_excel(
                writer, sheet_name="average chromatin intensity"
            )
            total_avg_intensity.to_excel(writer, sheet_name="total chromatin area")
            other_data.to_excel(writer, sheet_name="analysis_info")



def cycb_chromatin_batch_analyze_ilastik(positions:list, 
                                         analysis_paths:list, 
                                         instance_paths:list, 
                                         chromatin_paths:list, 
                                         ilastik_project_path: str,
                                         piecewise_fit:Optional[bool] = False
                                        ) -> tuple[pd.DataFrame]:

    for name_stub, analysis_path, instance_path, chromatin_path in zip(positions, analysis_paths, instance_paths, chromatin_paths):
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
        traces, ids = retrieve_traces(analysis_df, "GFP", 4)
        intensity, semantic, first_tps = qual_deg(traces, 4)

        if piecewise_fit:
            brk_pts = []
            fit_info = []
            for trace in intensity:
                front_end_chopped = 0
                mit_trace = trace[classification.iloc[i] ==1]
                front_end_chopped += (np.nonzero(semantic[i])[0][0] + 1)

                #forcing glob_min_index to be greater than glob_max_index
                glob_max_index = np.where(mit_trace == max(mit_trace))[0][0]
                glob_min_index = np.where(mit_trace == min(mit_trace[glob_max_index:]))[0][0]
                
                mit_neg_trace = mit_trace[glob_max_index:glob_min_index]
                front_end_chopped += glob_max_index + 1
                
                x = np.linspace(0, mit_neg_trace.shape[0]-1, mit_neg_trace.shape[0])
                brk_pt, fits = fit_cycb_regimes(x, mit_neg_trace)
                try:
                    slope_int = [(fit.coef_, fit.intercept_) for fit in fits]
                    brk_pts.append(brk_pt+front_end_chopped)
                except AttributeError:
                    slope_int = np.nan
                    brk_pts.append(np.nan)
                
                fit_info.append(slope_int)
        else:
            pass

        un_chromatin = []
        un_intensity = []
        un_number = []
        metphs = []
        metphs_intensity = []
        unaligned_chromosomes = [
            ilastik_unaligned_chromatin(identity, analysis_df, instance, chromatin, ilastik_project_path)
            for i, identity in enumerate(ids)
        ]
        for i, data_tuple in enumerate(unaligned_chromosomes):
            un_chromatin.append(data_tuple[0])
            un_intensity.append(data_tuple[1])
            un_number.append(data_tuple[2])
            metphs.append(data_tuple[3])
            metphs_intensity.append(data_tuple[4])

        cycb = pd.DataFrame(intensity)
        classification = pd.DataFrame(semantic)
        un_chromatin_area = pd.DataFrame(un_chromatin)
        un_chromatin_intensity = pd.DataFrame(un_intensity)
        un_chromatin_number = pd.DataFrame(un_number)
        metphs_area = pd.DataFrame(metphs)
        metphs_intensity = pd.DataFrame(metphs_intensity)

        d_temp = {
            "ids": ids,
            "first_mitosis": first_tps,
            "raw_data_paths": [instance_path, analysis_path, chromatin_path],
        }
        other_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d_temp.items()]))

        save_dir = os.path.dirname(analysis_path)
        if not os.path.isdir(save_dir):
            save_dir = os.getcwd()
        save_path = os.path.join(save_dir, f"{name_stub}_cycb_chromatin_ilastic.xlsx")

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            cycb.to_excel(writer, sheet_name="cycb")
            classification.to_excel(writer, sheet_name="classification")
            un_chromatin_area.to_excel(writer, sheet_name="unaligned chromatin area")
            un_chromatin_intensity.to_excel(
                writer, sheet_name="unaligned chromatin intensity"
            )
            metphs_area.to_excel(writer, sheet_name="metaphase plate area")
            un_chromatin_number.to_excel(
                writer, sheet_name="number unaligned chromosomes"
            )
            metphs_intensity.to_excel(
                writer, sheet_name="metaphase plate inten."
            )
            other_data.to_excel(writer, sheet_name="analysis_info")



if __name__ == "__main__":

    root_dir = Path('input/root/dir')
    inference_dirs = [obj.path for obj in os.scandir(root_dir) if '_inference' in obj.name and obj.is_dir()]
    analysis_paths = []
    instance_paths = []
    chromatin_paths = []
    positions = []
    for dir in inference_dirs:
        name_stub = re.search(r"[A-H]([1-9]|[0][1-9]|[1][0-2])_s(\d{2}|\d{1})", str(dir)).group()
        an_paths = glob.glob(f'{dir}/*analysis.xlsx')
        inst_paths = glob.glob(f'{dir}/*instance_movie.tif')
        chrom_paths = [path for path in glob.glob(f'{root_dir}/*Texas Red.tif') if str(name_stub) in path]
        analysis_paths += an_paths
        instance_paths += inst_paths
        chromatin_paths += chrom_paths
        positions.append(name_stub)

    try:
        assert len(analysis_paths)  == len(instance_paths) == len(chromatin_paths)
    except AssertionError:
        print('Files to analyze not organized properly')
        print('Analysis paths', len(analysis_paths))
        print('Instance paths', len(instance_paths))
        print('Chromatin paths', len(chromatin_paths))


    cycb_chromatin_batch_analyze(
        positions, analysis_paths, instance_paths, chromatin_paths
    )
