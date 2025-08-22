import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from collections import namedtuple
from skimage.restoration import denoise_tv_chambolle


def model_1_post(data, q):
    """
    Post-processing model_1 with final flat segment.
    q = [alpha_1, beta_1, gamma_1, tau_1, tau_2]
    """
    alpha_1, beta_1, gamma_1, tau_1, tau_2 = q
    x = data.xdata[0]
    y = data.ydata[0]
    x = x.reshape((x.shape[0],))

    heaviside_1 = np.heaviside(x - tau_1, 1)
    heaviside_2 = np.heaviside(x - tau_2, 1)

    return (
        y[0]
        + alpha_1 * x
        + (beta_1 - alpha_1) * (x - tau_1) * heaviside_1
        + (gamma_1 - beta_1) * (x - tau_2) * heaviside_2
    )


def model_2_post(data, q):
    """
    Post-processing model_2 with final flat segment.
    q = [alpha_1, alpha_2, beta_1, beta_2, gamma_2, tau_1, tau_2, tau_3, tau_4]
    """
    alpha_1, alpha_2, beta_1, beta_2, gamma_2, tau_1, tau_2, tau_3, tau_4 = q
    x = data.xdata[0]
    y = data.ydata[0]
    x = x.reshape((x.shape[0],))

    heaviside_1 = np.heaviside(x - tau_1, 1)
    heaviside_2 = np.heaviside(x - tau_2, 1)
    heaviside_3 = np.heaviside(x - tau_3, 1)
    heaviside_4 = np.heaviside(x - tau_4, 1)

    return (
        y[0]
        + alpha_1 * x
        + (beta_1 - alpha_1) * (x - tau_1) * heaviside_1
        + (alpha_2 - beta_1) * (x - tau_2) * heaviside_2
        + (beta_2 - alpha_2) * (x - tau_3) * heaviside_3
        + (gamma_2 - beta_2) * (x - tau_4) * heaviside_4
    )


def smooth_cycb_chromatin(
    cycb: pd.DataFrame,
    chromatin: pd.DataFrame,
):
    """
    Smooths and signal, chromatin traces; computes derivative of signal
    INPUTS:
        cycb: dataframe containing cyclinB signals
        chromatin: dataframe containing chromatin signals
        width: width of savitsky golay filter
        deriv_order: derivative to compute
    OUPUTS:
        cycb: list containing  cyclinb traces
        dcycb_dt: list containing computed derivative traces
        chromatin: list containing smoothed chromatin traces
    """

    traces = []
    dcycb_dt = []
    smooth_chromatin = []
    for i in range(cycb.shape[0]):
        trace = cycb.iloc[i].to_numpy()
        trace[np.isnan(trace)] = 0
        chromatin_trace = chromatin.iloc[i].to_numpy()
        chromatin_trace[np.isnan(chromatin_trace)] = 0

        smooth_trace = denoise_tv_chambolle(trace, weight=12)
        first_deriv = np.gradient(smooth_trace)
        smooth_chroma = denoise_tv_chambolle(chromatin_trace, weight=12)

        dcycb_dt.append(first_deriv)
        smooth_chromatin.append(smooth_chroma)
        traces.append(trace)

    return traces, dcycb_dt, smooth_chromatin


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
            cycb, un_chromatin_area
        )
        for i, trace in enumerate(smooth_cycb):
            traces.append(cycb.iloc[i])
            classification.append(np.asarray(classi.iloc[i]))
            dcycb.append(dcycb_dt[i])
            chromatin.append(chromatin_area[i])
            fit_format = fitting.iloc[i].dropna().to_numpy()
            fit_info.append(fit_format)

    return (traces, classification, dcycb, chromatin, fit_info)


def unpack_cycb_chromatin(
    cycb: list,
    dcycb_dt: list,
    classi: list,
    chromatin_area: list,
    fit_info: list[np.ndarray],
):
    """
    Unpacks data to compare individual data points
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
    unpacked_regime = []

    for j, cell_trace in enumerate(cycb):

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

                    if fit.shape[0] > 5:
                        if k < fit[5]:
                            regime = 0
                        elif fit[5]  <= k and k < fit[6]:
                            regime = 1
                        elif fit[6] <= k and k < fit[7]:
                            regime = 0
                        elif fit[7] <= k and k < fit[8]:
                            regime = 1
                        else:
                            regime = -1
                    else:
                        if k < fit[3]:
                            regime = 0
                        elif fit[3] <= k and k < fit[4]:
                            regime = 1
                        else:
                            regime = -1

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

    if fit_info.shape[0] > 5:
        taus += [fit_info[5], fit_info[6], fit_info[7], fit_info[8]]
        taus = np.asarray(taus).astype('int')
        mean_chromatin.append( chromatin[low_bound:taus[0]].mean() )
        mean_chromatin.append( chromatin[taus[0]:taus[1]].mean() )
        mean_chromatin.append( chromatin[taus[1]:taus[2]].mean() )
        mean_chromatin.append( chromatin[taus[2]:taus[3]].mean() )
        regimes = [0, 1, 0, 1]
        rates = [fit_info[0], fit_info[2], fit_info[1], fit_info[3]] 
        durations = [ taus[0]-low_bound, taus[1]-taus[0], taus[2]-taus[1], taus[3]-taus[2] ]  #alpha_1, #beta_1, #alpha_2, #beta_2
        cycb_ranges = [ cycb[low_bound]-cycb[high_bound], cycb[taus[0]] - cycb[high_bound], cycb[taus[1]] - cycb[high_bound], cycb[taus[2]] - cycb[high_bound]]

    else:
        taus += [fit_info[3], fit_info[4]]
        taus = np.asarray(taus).astype('int')
        mean_chromatin.append( chromatin[low_bound:taus[0]].mean() )
        mean_chromatin.append( chromatin[taus[0]:taus[1]].mean() )
        regimes = [0, 1]
        rates = [fit_info[0], fit_info[1]]
        durations = [ taus[0]-low_bound, taus[1]-taus[0] ]
        cycb_ranges = [ cycb[low_bound]-cycb[high_bound], cycb[taus[0]] - cycb[high_bound] ]


    return list( zip(rates, mean_chromatin, regimes, durations, cycb_ranges) )


def two_col_plot_montage(
        col1_traces:list[np.ndarray], 
        col2_traces:list[np.ndarray], 
        classification:list[np.ndarray],
        fits:Optional[list]=None
        ):

    try:
        assert len(col1_traces) == len(col2_traces)
    except AssertionError:
        print("Traces for column 1 and column 2 must be of the same length")
        return 
    
    fig, ax = plt.subplots(len(col1_traces), 2, figsize=(10, 2*len(col1_traces)), sharey='col')
    
    for i, zipped in enumerate(zip(col1_traces, col2_traces, classification)):
        col1_trace, col2_trace, classi = zipped
        
        col1_trace = np.asarray(col1_trace)
        low, high = deg_interval(col1_trace, classi)

        col1_trace = col1_trace[low:high]
        col2_trace = col2_trace[low:high]
        x = np.linspace(1, col1_trace.shape[0], col1_trace.shape[0])

        ax[i, 0].plot(x, col1_trace, c = 'k')
        ax[i, 1].plot(x, col2_trace, c = 'k')

        if fits:
            fit = fits[i]

            data = namedtuple("data", ["xdata", "ydata"])
            d = data([x], [col1_trace])

            if np.array_equal(fit, np.zeros(3)):
                continue
            else:
                if fit.shape[0] > 5:
                    fit = [val if i not in [5, 6, 7, 8] else val-low for i, val in enumerate(fit)]
                    print(fit)
                    to_plot = model_2_post(d, fit)
                else:
                    fit = [val if i not in [3, 4] else val-low for i, val in enumerate(fit)]
                    print(fit)
                    to_plot = model_1_post(d, fit)
                
                ax[i, 0].plot(x, to_plot, c = 'r')
        else:
            pass

    return fig




        

        

        

        

        

        





