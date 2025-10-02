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

    for id in chromatin_df["cell_id"].unique():
        trace = chromatin_df.query(f"cell_id=={id}")["cycb_intensity"].to_numpy()
        uchromatin = chromatin_df.query(f"cell_id=={id}")["u_chromatin_area"].to_numpy()
        tchromatin = chromatin_df.query(f"cell_id=={id}")["t_chromatin_area"].to_numpy()
        achromatin = tchromatin - uchromatin
        semantic = chromatin_df.query(f"cell_id=={id}")["semantic"].to_numpy()

        smooth_trace = denoise_tv_chambolle(trace, weight=weight)
        first_deriv = np.gradient(smooth_trace)

        traces.append(trace)
        smooth_traces.append(smooth_trace)
        derivatives.append(first_deriv)
        uchromatin_traces.append(uchromatin)
        achromatin_traces.append(achromatin)
        semantic_traces.append(semantic)

    return (
        traces,
        smooth_traces,
        derivatives,
        uchromatin_traces,
        achromatin_traces,
        semantic_traces,
    )


def unpack_cycb_chromatin(
    traces: list,
    derivatives: list,
    semantics: list,
    uchromatin_traces: list,
    achromatin_traces: list,
    changepts: list,
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
        remove_end_mitosis: Optional[bool], skip cells ending the movie in mitosis when True
    OUTPUTS:
        unpacked_smooth_cycb: list[float], per-timepoint Cyclin B values within mitosis window
        unpacked_dcycb_dt: list[float], per-timepoint degradation rates (negative derivative)
        unpacked_uchromatin_area: list[float], per-timepoint euchromatin area within mitosis
        unpacked_achromatin_area: list[float], per-timepoint heterochromatin area within mitosis
        unpacked_pos_in_mitosis: list[float], normalized position in mitosis [0,1]
        phase_flag: list[str], label 'slow' before changepoint and 'fast' after
        min_after_changept: list[float], minutes relative to changepoint (scaled as 4*(t - cp))
    """

    unpacked_smooth_cycb = []
    unpacked_dcycb_dt = []
    unpacked_uchromatin_area = []
    unpacked_achromatin_area = []
    unpacked_pos_in_mitosis = []
    phase_flag = []
    min_after_changept = []

    nan_registry = np.isnan(changepts)
    for j, cell_trace in enumerate(traces):
        semantic = semantics[j]
        changept = changepts[j]
        deg_rate = -1 * derivatives[j]
        uchromatin = uchromatin_traces[j]
        achromatin = achromatin_traces[j]

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
    )


def create_aggregate_df(paths: list[str], remove_end_mitosis: Optional[bool] = True):
    """
    Creates an aggregated DataFrame from multiple Excel files of chromatin analysis outputs
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        paths: list[str], file paths to Excel files containing chromatin analysis data
        remove_end_mitosis: Optional[bool], skip cells ending the movie in mitosis when True
    OUTPUTS:
        df: pd.DataFrame, aggregated rows with columns ['cycb', 'deg_rate', 'uchromatin', 'achromatin', 'pos_in_mitosis', 'phase_flag', 'min_after_changept', 'date-well']
    """

    usc_cont = []
    udd_cont = []
    uca_cont = []
    aca_cont = []
    upm_cont = []
    date_well_cont = []
    pf_cont = []
    mac_cont = []

    for path in paths:
        chromatin_df = pd.read_excel(path)
        _, smooth_traces, derivatives, uchromatin, achromatin, semantics = (
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
        ) = unpack_cycb_chromatin(
            smooth_traces,
            derivatives,
            semantics,
            uchromatin,
            achromatin,
            changepts,
            remove_end_mitosis,
        )
        usc_cont += unpacked_smooth_cycb
        udd_cont += unpacked_dcycb_dt
        uca_cont += unpacked_chromatin_area
        aca_cont += unpacked_achromatin_area
        upm_cont += unpacked_pos_in_mitosis
        pf_cont += phase_flag
        mac_cont += min_after_changept

        well = re.search(r"[A-H]([1-9]|0[1-9]|1[0-2])_s(\d{1,2})", str(path)).group()
        date = re.search(
            r"20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])", str(path)
        ).group()
        date_well = date + "-" + well[0]
        date_well_cont += [date_well] * len(unpacked_smooth_cycb)

    df = pd.DataFrame(
        {
            "cycb": usc_cont,
            "deg_rate": udd_cont,
            "uchromatin": uca_cont,
            "achromatin": aca_cont,
            "pos_in_mitosis": upm_cont,
            "phase_flag": pf_cont,
            "min_after_changept": mac_cont,
            "date-well": date_well_cont,
        }
    )

    return df
