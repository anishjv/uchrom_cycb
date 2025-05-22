import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import itertools
from scipy import stats
import seaborn as sns
import findiff
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression


def smooth_cycb_chromatin(
    cycb: pd.DataFrame,
    chromatin: pd.DataFrame,
    width: int,
    deriv_order: int,
):
    """
    Smooths and signal, chromatin traces; computes derivative of signal
    INPUTS:
        cycb: dataframe containing cyclinB signals
        chromatin: dataframe containing chromatin signals
        width: width of savitsky golay filter
        deriv_order: derivative to compute
    OUPUTS:
        smooth_cycb: list containing smoothed cylinb traces
        dcycb_dt: list containing computed derivative traces
        chromatin: list containing smoothed chromatin traces
    """

    smooth_cycb = []
    dcycb_dt = []
    smooth_chromatin = []
    for i in range(cycb.shape[0]):
        trace = cycb.iloc[i].to_numpy()
        trace[np.isnan(trace)] = 0
        chromatin_trace = chromatin.iloc[i].to_numpy()
        chromatin_trace[np.isnan(chromatin_trace)] = 0

        if trace.shape[0] > width:
            smooth_trace = savgol_filter(trace, width, 2)
            first_deriv = savgol_filter(trace, width, 2, deriv=deriv_order)
            smooth_chroma = savgol_filter(chromatin_trace, width, 2)
            smooth_chroma[smooth_chroma < 0] = (
                0  # artifact of savistky golay operating on non-continous array
            )
            smooth_cycb.append(smooth_trace)
            dcycb_dt.append(first_deriv)
            smooth_chromatin.append(smooth_chroma)
        else:
            smooth_cycb.append(None)
            dcycb_dt.append(None)
            smooth_chromatin.append(None)

    return smooth_cycb, dcycb_dt, smooth_chromatin


def retrive(positions: list[tuple], path_templ: str):
    """
    Retrieves information given positions and path template (lambda function)
    """

    traces = []
    classification = []
    dcycb = []
    chromatin = []
    fit_info = []

    for pos in positions:
        path = path_templ(pos)
        cycb = pd.read_excel(path, sheet_name="cycb", index_col=0)
        classi = pd.read_excel(path, sheet_name="classification", index_col=0)
        un_chromatin_area = pd.read_excel(
            path, sheet_name="unaligned chromatin area", index_col=0
        )
        fitting = pd.read_excel(path, sheet_name="fitting info", index_col=0)

        smooth_cycb, dcycb_dt, chromatin_area = smooth_cycb_chromatin(
            cycb, un_chromatin_area, 21, 1
        )
        for i, trace in enumerate(smooth_cycb):
            traces.append(trace)
            classification.append(np.asarray(classi.iloc[i]))
            dcycb.append(dcycb_dt[i])
            chromatin.append(chromatin_area[i])
            fit_format = fitting.iloc[i].dropna().to_numpy()
            fit_info.append(fit_format)

    return (traces, classification, dcycb, chromatin, fit_info)


def unpack_cycb_chromatin(
    smooth_cycb: list,
    dcycb_dt: list,
    classi: list,
    chromatin_area: list,
    fit_info: list[np.ndarray],
):
    """
    Unpacks data to compare individual data points
    INPUTS:
        smooth_cycb: traces output of retrieve()
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
    unpacked_regime = []

    for j, cell_trace in enumerate(smooth_cycb):

        if classi[j][-1] == 1:
            continue
        else:
            pass

        low_bound, high_bound = deg_interval(cell_trace, classi[j])

        for k, val in enumerate(cell_trace):
            if low_bound <= k and high_bound > k:
                unpacked_smooth_cycb.append(val)
                unpacked_dcycb_dt.append(-1 * dcycb_dt[j][k])
                unpacked_chromatin_area.append(chromatin_area[j][k])

                if fit_info:
                    fit = fit_info[j]

                    if fit.shape[0] > 3:
                        if k < fit[4] - 2:
                            regime = 0
                        elif fit[4] - 2 <= k and k < fit[5] - 2:
                            regime = 1
                        elif fit[5] - 2 <= k and k < fit[6] - 2:
                            regime = 0
                        else:
                            regime = 1
                    else:
                        if k < fit[2] - 2:
                            regime = 0
                        else:
                            regime = 1

                    unpacked_regime.append(regime)
                else:
                    pass

            else:
                pass

    return (
        unpacked_smooth_cycb,
        unpacked_dcycb_dt,
        unpacked_chromatin_area,
        unpacked_regime,
    )

def deg_interval(cycb: np.ndarray, classi:np.ndarray):
    """
    Computes the interval over which CyclinB is degrading
    INPUTS:
        cycb: CyclinB trace
        classi: Cell-APP classification trace
    OUTPUTS:
        low_bound: first timepoint
        high_bound: last timepoint
    """
        
    front_end_chopped = 0
    mit_trace = cycb[classi == 1]
    front_end_chopped += np.nonzero(classi)[0][0]

    # forcing glob_min_index to be greater than glob_max_index
    glob_max_index = np.where(mit_trace == max(mit_trace))[0][0]
    glob_min_index = np.where(mit_trace == min(mit_trace[glob_max_index:]))[0][0]

    low_bound = front_end_chopped + glob_max_index
    high_bound = (
        front_end_chopped + glob_min_index
    )  # these are exactly the indices considered for bayesian inference

    return low_bound, high_bound


def chromatin_vs_rate(cycb:np.ndarray, classi:np.ndarray, chromatin:np.ndarray, fit_info:list):
    """
    Retrives rate from Bayesian inference and unaligned chromatin during timeframe that aligns with rate
    INPUTS:
        cycb: CyclinB trace
        classi: Cell-APP classifications
        chromatin: unaligned chromatin trace
        fit_info: bayesian inference fitting info
    OUTPUTS:
        list of tuples with form: (rate, mean_chromatin, regime)
    """

    taus = []
    mean_chromatin = []

    low_bound, high_bound = deg_interval(cycb, classi)

    if fit_info.shape[0] > 3:
        taus += [fit_info[4], fit_info[5], fit_info[6]]
        taus = np.asarray(taus).astype('int')
        mean_chromatin.append( chromatin[low_bound:taus[0]].mean() )
        mean_chromatin.append( chromatin[taus[0]:taus[1]].mean() )
        mean_chromatin.append( chromatin[taus[1]:taus[2]].mean() )
        mean_chromatin.append( chromatin[taus[2]:high_bound].mean() )
        regimes = [0, 1, 0, 1]
        rates = [fit_info[0], fit_info[1], fit_info[2], fit_info[3]]


    else:
        taus.append(fit_info[2])
        taus = np.asarray(taus).astype('int')
        mean_chromatin.append( chromatin[low_bound:taus[0]].mean() )
        mean_chromatin.append( chromatin[taus[0]:high_bound].mean() )
        regimes = [0, 1]
        rates = [fit_info[0], fit_info[1]]

    #rates = set(fit_info) - set(taus)

    print(len(rates), len(mean_chromatin), len(regimes))

    return list( zip(rates, mean_chromatin, regimes) )


        

        

        

        

        

        





