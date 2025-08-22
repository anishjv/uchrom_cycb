import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from typing import Callable
from pymcmcstat.MCMC import MCMC  # type:ignore
from pathlib import Path
import os
import pandas as pd
from scipy.signal import savgol_filter
from zipfile import BadZipFile
from deg_analysis import deg_interval
from typing import Optional
import glob


def fit_three_lines(
    x: npt.NDArray, y: npt.NDArray
) -> tuple[tuple[int], LinearRegression]:
    """
    Fits three lines to a curve using Linear Regression
    INPUTS:
        x: np.ndarray, domain over which to fit
        y: np.ndarray, values to fit
    OUTPUTS:
    """
    best_score = float("inf")
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

            ssr = (
                np.sum((y1 - y1_pred) ** 2)
                + np.sum((y2 - y2_pred) ** 2)
                + np.sum((y3 - y3_pred) ** 2)
            )

            if ssr < best_score:
                best_score = ssr
                best_breaks = (i, j)
                best_models = (model1, model2, model3)

    return best_breaks, best_models


def fit_two_lines(
    x: npt.NDArray, y: npt.NDArray
) -> tuple[tuple[int], LinearRegression]:
    """
    Fits two lines to a curve using Linear Regression
    INPUTS:
        x: np.ndarray, domain over which to fit
        y: np.ndarray, values to fit
    OUTPUTS:
    """

    best_break = None
    best_score = float("inf")
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


def fit_cycb_regimes(
    x: npt.NDArray, y: npt.NDArray
) -> tuple[tuple[int], LinearRegression]:
    """
    Wrapper function for 'fit_two_lines' and 'fit_three_lines'
    INPUTS:
        x: np.ndarray, domain over which to fit
        y: np.ndarray, values to fit
    OUTPUTS:
    """

    best_breaks, best_models = fit_two_lines(x, y)

    # If x/y was short (len<2) error will occur
    try:
        eps = np.abs(best_models[0].coef_ - best_models[1].coef_)
    except AttributeError:
        return np.nan, (np.nan,)

    # If abs(slope) of first fit > abs(slope) of second fit => check for three regimes
    if np.abs(best_models[0].coef_) > abs(best_models[1].coef_):
        best_breaks, best_models = fit_three_lines(x, y)
        return best_breaks, best_models

    # Must check for 3 regimes first, as
    # the slope of the two misfitted lines may have been arbitrarily similar
    # If slope of two fits is similar => we say that no slow regime occured (fit one line)
    if eps > np.abs(max([fit.coef_ for fit in best_models])) / 5:
        pass
    else:
        x = x.reshape(-1, 1)
        best_models = (LinearRegression().fit(x, y),)
        best_breaks = np.nan

    return best_breaks, best_models


def sigmoid(x, weight:Optional[float]=1):
    return 1 / (1 + np.exp(-x/weight))


def model_1(data, q):
    """
     Heaviside function model with one slope switchpoint.
    Designed to interface with the package pymcmcstat
    """

    alpha_1, delta_beta_1, gamma_1, theta_1, theta_2 = q
    x = data.xdata[0]
    y = data.ydata[0]
    x = x.reshape((x.shape[0],))
    tau_min = min( x.shape[0]//10, 2 )
    tau_max = x.shape[0]
    delta = min( x.shape[0]//10, 2 )

    # ensures beta_1 < alpha_1
    beta_1 = alpha_1 - delta_beta_1
    tau_1 = tau_min + (tau_max - tau_min - delta) *sigmoid(theta_1)
    tau_2 = (tau_1 + delta) + (tau_max - tau_1 - delta) *sigmoid(theta_2)

    x_idx = np.arange(x.shape[0])
    heaviside_1 = np.heaviside(x_idx - tau_1, 1)
    heaviside_2 = np.heaviside(x_idx - tau_2, 1)

    return (
        alpha_1 * x
        - alpha_1 * (x - tau_1) * heaviside_1
        + beta_1 * (x - tau_1) * heaviside_1
        - beta_1 * (x - tau_2) * heaviside_2
        + gamma_1 * (x - tau_2) * heaviside_2
        + y[0]
    )

def model_2(data, q):
    """
    Heaviside function model with four slope switchpoints.
    Suitable for pymcmcstat.
    """
    alpha_1, delta_alpha_2, delta_beta_1, delta_beta_2, gamma_1, theta_1, theta_2, theta_3, theta_4 = q
    x = data.xdata[0]
    y = data.ydata[0]
    x = x.reshape((x.shape[0],))

    tau_min = min(x.shape[0] // 10, 2)
    tau_max = x.shape[0]
    delta = min(x.shape[0] // 10, 2)

    # Sequential slope construction
    beta_1 = alpha_1 - delta_beta_1         # β₁ < α₁
    alpha_2 = beta_1 + delta_alpha_2        # α₂ > β₁
    beta_2 = alpha_2 - delta_beta_2         # β₂ < α₂    

    # Sequential transition points with enforced spacing
    tau_1 = tau_min + (tau_max - tau_min - 3 * delta) * sigmoid(theta_1)
    tau_2 = (tau_1 + delta) + (tau_max - tau_1 - 3 * delta) * sigmoid(theta_2)
    tau_3 = (tau_2 + delta) + (tau_max - tau_2 - 2 * delta) * sigmoid(theta_3)
    tau_4 = (tau_3 + delta) + (tau_max - tau_3 - delta) * sigmoid(theta_4)

    x_idx = np.arange(x.shape[0])
    h1 = np.heaviside(x_idx - tau_1, 1)
    h2 = np.heaviside(x_idx - tau_2, 1)
    h3 = np.heaviside(x_idx - tau_3, 1)
    h4 = np.heaviside(x_idx - tau_4, 1)

    return (
        y[0]
        + alpha_1 * x
        + (beta_1 - alpha_1) * (x - tau_1) * h1
        + (alpha_2 - beta_1) * (x - tau_2) * h2
        + (beta_2 - alpha_2) * (x - tau_3) * h3
        + (gamma_1 - beta_2) * (x - tau_4) * h4
    )

def ssfun_1(q, data):
    """
    Heteroscedastic error function using combined Poisson + Gaussian noise.
    """
    a = 0.1 # Poisson noise scale
    b = 0.01  # Gaussian noise floor

    ydata = data.ydata[0]
    ymodel = model_1(data, q)
    res = ymodel.reshape(ydata.shape) - ydata

    sigma2 = a**2 * np.abs(ymodel) + b**2
    ssr = np.sum((res**2) / sigma2 + np.log(sigma2))

    return ssr


def ssfun_2(q, data):
    """
    Heteroscedastic error function using combined Poisson + Gaussian noise, with penalty for short third regime.
    """
    a = 0.1  # Poisson noise scale
    b = 0.01  # Gaussian noise floor

    ydata = data.ydata[0]
    ymodel = model_2(data, q)
    res = ymodel.reshape(ydata.shape) - ydata

    sigma2 = a**2 * np.abs(ymodel) + b**2
    ssr = np.sum((res**2) / sigma2 + np.log(sigma2))

    alpha_1, delta_alpha_2, delta_beta_1, delta_beta_2, delta_gamma_2, theta_1, theta_2, theta_3, theta_4 = q
    tau_min = min(data.xdata[0].shape[0] // 10, 2 )
    delta = min(data.xdata[0].shape[0] // 10, 2 )

    tau_1 = tau_min + (data.xdata[0].shape[0] - tau_min - 3 * delta) * sigmoid(theta_1)
    tau_2 = tau_1 + delta + (data.xdata[0].shape[0] - tau_1 - 3 * delta) * sigmoid(theta_2)
    tau_3 = tau_2 + delta + (data.xdata[0].shape[0] - tau_2 - 2 * delta) * sigmoid(theta_3)
    tau_4 = tau_3 + delta + (data.xdata[0].shape[0] - tau_3 - delta) * sigmoid(theta_4)

    d_min = 20.0  # minimum acceptable segment length
    lambda_penalty = 100000.0  # strength of the penalty

    penalty = max(0, d_min - (tau_3 - tau_2))
    ssr += lambda_penalty * penalty**2

    return ssr



def compute_bic(mcstat, model_func) -> float:
    
    ndata = mcstat.data.ydata[0].size
    k = len(mcstat.simulation_results.results["names"])
    results = mcstat.simulation_results.results
    burnin = int(results["nsimu"] / 2)
    chain = results["chain"][burnin:, :]
    map_theta = np.mean(chain, axis=0)

    y_model = model_func(mcstat.data, map_theta).flatten()
    y_data = mcstat.data.ydata[0].flatten()
    ssr = np.sum((y_data - y_model) ** 2)

    # If updatesigma=True, use s2chain; otherwise fallback to SSR-based estimate
    s2chain = results.get("s2chain")
    if s2chain is None or np.all(s2chain == None):
        sigma2 = ssr / ndata
    else:
        sigma2 = np.mean(s2chain)

    logL = -0.5 * ndata * np.log(2 * np.pi * sigma2) - 0.5 * ssr / sigma2
    bic = k * np.log(ndata) - 2 * logL
    return bic



def bayes_fit_cycb_regimes(
    x: npt.NDArray, y: npt.NDArray, ssfuns: list[Callable], models: list[Callable]
) -> list[float]:
    """
    Computes the fit parameters of either the
    one-switchpoint or three-switchpoint heaviside
    model.
    INPUTS:
        x: domain over with to fit
        y: values to fit
        ssfuns: list containing model evalutation functions
        models: list containing models. models in this list should be in the same order as their corresponding ssfun in ssfuns
    OUTPUTS:
    """

    # Model 1 simulation
    mcstat = MCMC()
    mcstat.data.add_data_set(x, y)
    mcstat.model_settings.define_model_settings(sos_function=ssfuns[0])
    mcstat.simulation_options.define_simulation_options(nsimu=1e4, updatesigma=False)
    mcstat.parameters.add_model_parameter(name="alpha_1", theta0=-0.2, maximum=0, prior_mu=-0.2, prior_sigma=0.2) #low and narrow prior
    mcstat.parameters.add_model_parameter(name="delta_beta_1", theta0=0.75, minimum=0.2, prior_mu=0.75, prior_sigma=0.75) #high and wide prior
    mcstat.parameters.add_model_parameter(name="gamma_1", theta0=0, minimum= -1e-3, prior_mu=0, prior_sigma=0.01)
    mcstat.parameters.add_model_parameter(
        name="theta_1", theta0=2
    ) 
    mcstat.parameters.add_model_parameter(
        name="theta_2", theta0=2
    ) 
    mcstat.run_simulation()

    # Model 2 simulation
    mcstat_2 = MCMC()
    mcstat_2.data.add_data_set(x, y)
    mcstat_2.model_settings.define_model_settings(sos_function=ssfuns[1])
    mcstat_2.simulation_options.define_simulation_options(nsimu=10e4, updatesigma=False)
    mcstat_2.parameters.add_model_parameter(name="alpha_1", theta0=-0.2, maximum=0.0, prior_mu=-0.2, prior_sigma=0.2)
    mcstat_2.parameters.add_model_parameter(name="delta_alpha_2", theta0=0.75, minimum=0.2, prior_mu=0.75, prior_sigma=0.75)
    mcstat_2.parameters.add_model_parameter(name="delta_beta_1", theta0=0.75, minimum=0.2, prior_mu=0.75, prior_sigma=0.75)
    mcstat_2.parameters.add_model_parameter(name="delta_beta_2", theta0=0.75, minimum=0.2, prior_mu=0.75, prior_sigma=0.75)#high and wide prio
    mcstat_2.parameters.add_model_parameter(name="gamma_1", theta0=0, minimum= -1e-3, prior_mu=0, prior_sigma=0.01)
    mcstat_2.parameters.add_model_parameter(
        name="theta_1", theta0=0, 
    )
    mcstat_2.parameters.add_model_parameter(
        name="theta_2", theta0=0, 
    )
    mcstat_2.parameters.add_model_parameter(
        name="theta_3", theta0=0, 
    )
    mcstat_2.parameters.add_model_parameter(
        name="theta_4", theta0=0, 
    )
    mcstat_2.run_simulation()

    bic1 = compute_bic(mcstat, models[0])
    bic2 = compute_bic(mcstat_2, models[1])
    delta_bic = bic1 - bic2

    # emperical bayesian information criterion threshold
    model = mcstat if delta_bic < 100 else mcstat_2

    results = model.simulation_results.results
    burnin = int(results["nsimu"] / 2)
    chain = results["chain"][burnin:, :]

    params = np.mean(chain, axis=0)
    delta = min( x.shape[0]//10, 2 )
    tau_min = min( x.shape[0]//10, 2 )
    tau_max = x.shape[0]
    if params.shape[0] == 5:

        theta_1 = params[3]
        theta_2 = params[4]
        tau_1 = tau_min + (tau_max - tau_min - delta) *sigmoid(theta_1)
        tau_2 = (tau_1 + delta) + (tau_max - tau_1 - delta) *sigmoid(theta_2)
        return [
            params[0],
            params[0] - params[1],
            params[2],
            tau_1,
            tau_2
        ]  # alpha_1, #beta_1, #gamma_1, #tau_1, #tau_2

    elif params.shape[0] == 9:
        theta_1 = params[5]
        theta_2 = params[6]
        theta_3 = params[7]
        theta_4 = params[8]
        tau_1 = tau_min + (tau_max - tau_min - 3 * delta) * sigmoid(theta_1)
        tau_2 = (tau_1 + delta) + (tau_max - tau_1 - 3 * delta) * sigmoid(theta_2)
        tau_3 = (tau_2 + delta) + (tau_max - tau_2 - 2 * delta) * sigmoid(theta_3)
        tau_4 = (tau_3 + delta) + (tau_max - tau_3 - delta) * sigmoid(theta_4)

        alpha_1 = params[0]
        beta_1 = alpha_1 - params[2]
        alpha_2 = beta_1 + params[1]
        beta_2 = alpha_2 - params[3]
        gamma_1 = params[4]

        return [
            alpha_1, 
            alpha_2,
            beta_1,
            beta_2,
            gamma_1,
            tau_1,
            tau_2,
            tau_3,
            tau_4
        ]  


if __name__ == "__main__":
    
    root_dir= Path('/nfs/turbo/umms-ajitj/anishjv/for_analysis/1/')
    inference_dirs = [obj.path for obj in os.scandir(root_dir) if '_inference' in obj.name and obj.is_dir()]

    cycb_paths = []
    for dir in inference_dirs:
        print(dir)
        cycb_path = glob.glob(f'{dir}/*chromatin.xlsx')
        cycb_paths += cycb_path
        

    print("Will process: \n", cycb_paths)
    for cycb_path in cycb_paths:
        try:
            writer = pd.ExcelWriter(
                cycb_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
            )
        except BadZipFile:
            continue

        traces = pd.read_excel(cycb_path, sheet_name=0, index_col=0).to_numpy()
        semantic = pd.read_excel(cycb_path, sheet_name=1, index_col=0).to_numpy()

        assert traces.shape == semantic.shape, f"Mismatched shapes: {traces.shape} vs {semantic.shape}"

        fit_info = []
        for i, trace in enumerate(traces):
            # cannot be smoothed or trace ends in mitosis
            if semantic[i][-1] != 1:
                trace[np.isnan(trace)] = 0
            else:
                fit_info.append(np.zeros(3))
                continue
            
            low_bound, high_bound = deg_interval(trace, semantic[i])
            mit_neg_trace = trace[low_bound:high_bound]

            x = np.linspace(1, mit_neg_trace.shape[0], mit_neg_trace.shape[0])
            if x.shape[0] > 0:
                params = bayes_fit_cycb_regimes(
                    x, mit_neg_trace, [ssfun_1, ssfun_2], [model_1, model_2]
                )
            else:
                params = None

            if params == None:
                params = np.zeros(3)
            elif len(params) == 5:
                params[3] += low_bound
                params[4] += low_bound
            elif len(params) == 9:
                params = [
                    param if i < 5 else param + low_bound
                    for i, param in enumerate(params)
                ]
            fit_info.append(tuple(params))


        fit_info = pd.DataFrame(fit_info)
        fit_info.to_excel(writer, sheet_name="fitting info")
        writer.close()
