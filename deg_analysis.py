import pandas as pd
import numpy as np
from typing import Optional
from skimage.restoration import denoise_tv_chambolle
from changept import *
import re


def smooth_cycb_chromatin(
    chromatin_df: pd.DataFrame,
    weight: Optional[int]=12,
):
    """
    Smooths and signal, chromatin traces; computes derivative of signal
    ----------------------------------------------------------------------------------
    INPUTS:
        chromatin_df: pandas dataframe containing relevant information
        weight: weight to be used in TV denoising
    OUPUTS:
        traces: list containing cyclinb traces
        smooth_traces: list containing smoothed cyclinb traces
        derivatives: list containing computed derivative traces
        uchromatin_traces: list containing chromatin traces
        smoothed_uchromatin_traces: list containing smoothed chromatin traces
    """

    traces = []
    smooth_traces = []
    derivatives = []
    uchromatin_traces = []
    smoothed_uchromatin_traces = []
    semantic_traces = []

    for id in chromatin_df['cell_id'].unique():
        trace = chromatin_df.query(f"cell_id=={id}")['cycb_intensity'].to_numpy()
        uchromatin = chromatin_df.query(f"cell_id=={id}")['u_chromatin_area'].to_numpy()
        semantic = chromatin_df.query(f"cell_id=={id}")['semantic'].to_numpy()
        
        smooth_trace = denoise_tv_chambolle(trace, weight=weight)
        first_deriv = np.gradient(smooth_trace)
        smooth_uchromatin = denoise_tv_chambolle(uchromatin, weight=weight)

        traces.append(trace)
        smooth_traces.append(smooth_trace)
        derivatives.append(first_deriv)
        uchromatin_traces.append(uchromatin)
        smoothed_uchromatin_traces.append(smooth_uchromatin)
        semantic_traces.append(semantic)

    return (
    traces, 
    smooth_traces, 
    derivatives, 
    uchromatin_traces, 
    smoothed_uchromatin_traces, 
    semantic_traces
    )


def unpack_cycb_chromatin(
    traces: list,
    derivatives: list,
    semantics: list,
    uchromatin_traces: list,
    changepts: list,
    remove_end_mitosis: Optional[bool]=True
):
    """
    Unpacks data to compare individual data points
    ------------------------------------------------------
    INPUTS:
        cycb: traces output of retrieve()
        dcycb_dt: dcycb output of retrieve()
        classi: classification output of retrieve()
        chromatin_area: chromatin output of retrieve()
        fit_info: fit_info output of retrieve()
    OUTPUTS:
        unpacked_smooth_cycb: unpacked version
        unpacked_dcycb_dt: ""
        unpacked_chromatin_area: ""
        unpacked_regime: ""
    """

    unpacked_smooth_cycb = []
    unpacked_dcycb_dt = []
    unpacked_chromatin_area = []
    unpacked_pos_in_mitosis = []

    nan_registry = np.isnan(changepts)
    for j, cell_trace in enumerate(traces):
        semantic = semantics[j]
        changept = changepts[j]
        deg_rate = -1*derivatives[j]
        uchromatin = uchromatin_traces[j]
    
        # if changepoint detection failed ignore cell
        if nan_registry[j]:
            continue
        else:
            pass
          
        # if cell ended movie in mitosis, optionally ignore cell
        if remove_end_mitosis and semantic[-1] == 1 :
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
            if low_bound <= t and changept >= t:
                unpacked_smooth_cycb.append(val)
                unpacked_dcycb_dt.append(deg_rate[t])
                unpacked_chromatin_area.append(uchromatin[t])
                unpacked_pos_in_mitosis.append(
                                                (t-t_min)/(t_max-t_min)
                                            )

            else:
                pass

    return (
        unpacked_smooth_cycb,
        unpacked_dcycb_dt,
        unpacked_chromatin_area,
        unpacked_pos_in_mitosis
    )



def create_aggregate_df(paths):
    """
    Create aggregated DataFrame from multiple Excel files containing chromatin analysis data.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	paths: list[str], list of file paths to Excel files containing chromatin analysis data
    OUTPUTS:
    	df: pd.DataFrame, aggregated data with columns ['cycb', 'deg_rate', 'uchromatin', 'pos_in_mitosis', 'date-well']
    """

    usc_cont = []
    udd_cont = []
    uca_cont = []
    upm_cont = []
    date_well_cont = []

    for path in paths:
        chromatin_df = pd.read_excel(path)
        _, smooth_traces, derivatives, uchromatin, _, semantics = smooth_cycb_chromatin(chromatin_df, weight = 5)
        changepts = [gauss_changept(cycb, deg_rate, sem) for cycb, deg_rate, sem in zip(smooth_traces, derivatives, semantics)] 

        unpacked_smooth_cycb, unpacked_dcycb_dt, unpacked_chromatin_area, unpacked_pos_in_mitosis = unpack_cycb_chromatin(
                                                                                        smooth_traces,
                                                                                        derivatives,
                                                                                        semantics,
                                                                                        uchromatin,
                                                                                        changepts,
                                                                                        )
        usc_cont += unpacked_smooth_cycb
        udd_cont += unpacked_dcycb_dt
        uca_cont += unpacked_chromatin_area
        upm_cont += unpacked_pos_in_mitosis
        
        well = re.search(r"[A-H]([1-9]|0[1-9]|1[0-2])_s(\d{1,2})", str(path)).group()        
        date = re.search(r"20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])", str(path)).group()
        date_well = date + '-' + well
        date_well_cont += [date_well] * len(unpacked_smooth_cycb)

    
    df = pd.DataFrame({
        'cycb': usc_cont,
        'deg_rate': udd_cont,
        'uchromatin': uca_cont,
        'pos_in_mitosis': upm_cont,
        'date-well': date_well_cont
    })

    return df



        

        





