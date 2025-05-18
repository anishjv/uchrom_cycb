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
)
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, watershed
from skimage.feature import peak_local_max
import glob
import re
from sklearn.linear_model import LinearRegression
from segment_chromatin import (
    unaligned_chromatin,
    bbox,
    ilastik_unaligned_chromatin,
)  # type:ignore
from piecewise_fit import (
    bayes_fit_cycb_regimes,
    ssfun_1,
    ssfun_2,
    model_1,
    model_2,
)  # type:ignore


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
            if (
                len(peak_widths) == 1
                and (np.sum(props["plateau_sizes"]) - peak_widths[0]) < t_char
            ):
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


def longest_negative_sequence(arr):
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
    piecewise_fit: Optional[bool] = False,
) -> tuple[pd.DataFrame]:

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
        traces, ids = retrieve_traces(analysis_df, "GFP", 4)
        intensity, semantic, first_tps = qual_deg(traces, 4)

        if piecewise_fit:
            fit_info = []
            for i, trace in enumerate(intensity):
                front_end_chopped = 0
                mit_trace = trace[semantic[i] == 1]
                front_end_chopped += np.nonzero(semantic[i])[0][0] + 1

                # forcing glob_min_index to be greater than glob_max_index
                glob_max_index = np.where(mit_trace == max(mit_trace))[0][0]
                glob_min_index = np.where(mit_trace == min(mit_trace[glob_max_index:]))[
                    0
                ][0]

                mit_neg_trace = mit_trace[glob_max_index:glob_min_index]
                front_end_chopped += glob_max_index + 1

                x = np.linspace(1, mit_neg_trace.shape[0], mit_neg_trace.shape[0])
                if x.shape[0] > 0:
                    params = bayes_fit_cycb_regimes(
                        x, mit_neg_trace, [ssfun_1, ssfun_2], [model_1, model_2]
                    )
                else:
                    params = None

                if params == None:
                    pass
                elif len(params) == 3:
                    params[2] += front_end_chopped
                elif len(params) == 7:
                    params = [
                        param if i < 4 else param + front_end_chopped
                        for i, param in enumerate(params)
                    ]

                fit_info.append(params)

        else:
            pass

        un_chromatin = []
        un_intensity = []
        tot_intensity = []
        un_number = []
        tot_avg_intensity = []
        tot_area = []
        first_anaphase = []
        traced_rois = []
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
            '''
            if len(data_tuple[7]) > 0:
                traced_rois.append(np.asarray(data_tuple[7], dtype="object"))
            '''

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

        if len(traced_rois) > 0:
            arr_save_path = os.path.join(save_dir, f"{name_stub}_rois.npy")
            traced_rois = np.asarray(traced_rois, dtype="object")
            np.save(arr_save_path, traced_rois)

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


def cycb_chromatin_batch_analyze_ilastik(
    positions: list,
    analysis_paths: list,
    instance_paths: list,
    chromatin_paths: list,
    ilastik_project_path: str,
    piecewise_fit: Optional[bool] = False,
) -> tuple[pd.DataFrame]:

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
        traces, ids = retrieve_traces(analysis_df, "GFP", 4)
        intensity, semantic, first_tps = qual_deg(traces, 4)

        if piecewise_fit:
            fit_info = []
            for i, trace in enumerate(intensity):
                front_end_chopped = 0
                mit_trace = trace[semantic[i] == 1]
                front_end_chopped += np.nonzero(semantic[i])[0][0] + 1

                # forcing glob_min_index to be greater than glob_max_index
                glob_max_index = np.where(mit_trace == max(mit_trace))[0][0]
                glob_min_index = np.where(mit_trace == min(mit_trace[glob_max_index:]))[
                    0
                ][0]

                mit_neg_trace = mit_trace[glob_max_index:glob_min_index]
                front_end_chopped += glob_max_index + 1

                x = np.linspace(1, mit_neg_trace.shape[0], mit_neg_trace.shape[0])
                if x.shape[0] > 0:
                    params = bayes_fit_cycb_regimes(
                        x, mit_neg_trace, [ssfun_1, ssfun_2], [model_1, model_2]
                    )
                else:
                    params = None

                if params == None:
                    pass
                elif len(params) == 3:
                    params[2] += front_end_chopped
                elif len(params) == 7:
                    params = [
                        param if i < 4 else param + front_end_chopped
                        for i, param in enumerate(params)
                    ]

                fit_info.append(params)

        else:
            pass

        un_chromatin = []
        un_intensity = []
        un_number = []
        metphs = []
        metphs_intensity = []
        unaligned_chromosomes = [
            ilastik_unaligned_chromatin(
                identity, analysis_df, instance, chromatin, ilastik_project_path
            )
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
            metphs_intensity.to_excel(writer, sheet_name="metaphase plate inten.")
            other_data.to_excel(writer, sheet_name="analysis_info")


if __name__ == "__main__":

    root_dir = Path("input/root/dir")
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

    cycb_chromatin_batch_analyze(
        positions, analysis_paths, instance_paths, chromatin_paths
    )
