import numpy as np
from scipy.signal import find_peaks
import numpy.typing as npt
from typing import Optional

def changept(curve: npt.NDArray, search_dist: Optional[int]=1):

    '''
    Simply method to detect when a Cyclin B trace enters its 
    fast degradation phase. 

    Draws a straight line between two anchors: the trace's max value 
    and it's min value. Then finds the point on the trace furthest
    from this line. That point is ~ point of maximum negative curvature.
    -------------------------------------------------------------------
    INPUTS:
        curve: np.array, mitotic portion of Cyclin B trace
        search_dist: np.array, allow the min anchor to drift backwards if new anchor is within search_dist y-axis unit(s) of true min achor
    OUTPUTS:
        cp: int, index at which Cyclin B trace enters fast degradation phase
        i_max: int, max anchor point
        i_min: int, min_anchor point 
    '''

    y = np.asarray(curve)
    x = np.arange(y.size)

    i_max = np.argmax(y)
    i_min = np.argmin(y)

    # handle flat curve edge case
    if i_max == i_min:
        return i_max, i_max, i_min

    # --- look backwards from i_min ---
    candidates = np.where(y[:i_min] <= y[i_min] + search_dist)[0]

    if len(candidates) > 0:
        i_min = candidates[0]  # earliest point within threshold

    # enforce temporal order
    i0, i1 = sorted([i_max, i_min])

    # endpoints of the line
    p0 = np.array([x[i0], y[i0]])
    p1 = np.array([x[i1], y[i1]])
    v = p1 - p0
    v = v / np.linalg.norm(v)

    # only points between max and updated min
    xs = x[i0:i1+1]
    ys = y[i0:i1+1]
    pts = np.column_stack((xs, ys)) - p0

    # perpendicular distance to the line
    dist = np.linalg.norm(pts - np.outer(pts @ v, v), axis=1)

    # convert back to global index
    cp = i0 + np.argmax(dist)

    return cp, i_max, i_min


def validate_cyclin_b_trace(trace: np.ndarray):
    """
    Validates whether a given trace is truly a cyclin B trace based on specific criteria.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
        trace: np.ndarray, cyclin B intensity trace over time
    OUTPUTS:
        np.nan is one or more criterion failed; True if all criterion passed
    """

    # Criterion 1: Check for peak with prominence > 5
    peaks, properties = find_peaks(trace, prominence=3, width=0)
    peaks_criterion = True if len(peaks) > 1 else False

    # Criterion 2: Check if range > 10
    trace_range = np.max(trace) - np.min(trace)
    range_criterion = True if trace_range > 10 else False

    # Criterion 3: Check left base > right base for main (tallest) peak
    # Find the tallest peak (highest prominence)
    if peaks_criterion:
        prominences = properties["prominences"]
        tallest_peak_idx = np.argmax(prominences)

        # Get left and right bases for the tallest peak
        left_bases = properties["left_bases"]
        right_bases = properties["right_bases"]

        left_base_value = trace[left_bases[tallest_peak_idx]]
        right_base_value = trace[right_bases[tallest_peak_idx]]

        hysterisis_criterion = True if left_base_value > right_base_value else False
    
    else:
        hysterisis_criterion = False

    return peaks_criterion, range_criterion, hysterisis_criterion
