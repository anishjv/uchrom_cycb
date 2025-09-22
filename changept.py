import numpy as np
import findiff
from scipy.optimize import curve_fit
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.ndimage import binary_dilation
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math

def falling_half_gaussian(t, A, mu, sigma):
    """
    Compute a half-gaussian function that is zero for t < mu and follows gaussian decay for t >= mu.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	t: np.ndarray, time points
    	A: float, amplitude of the gaussian
    	mu: float, location parameter (start of non-zero region)
    	sigma: float, width parameter of the gaussian decay
    OUTPUTS:
    	result: np.ndarray, half-gaussian values (zero for t < mu, gaussian for t >= mu)
    """
    # Only return gaussian values for t >= mu, otherwise 0
    result = np.zeros_like(t)
    mask = t >= mu
    result[mask] = A * np.exp(-(t[mask] - mu)**2 / (2 * sigma**2))
    return result


def _extract_mitotic_data(intensity, semantic, derivative, dilation_size=21):
    """
    Extract mitotic data and indices from dilated semantic segmentation.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	intensity: np.ndarray, CyclinB intensity trace over time
    	semantic: np.ndarray, binary classification (1=mitotic, 0=non-mitotic)
        derivative: np.ndarray, derivative of CyclinB intensity trace
    	dilation_size: int, size of dilation kernel (default 21 for 10 frames on each side)
    OUTPUTS:
    	mitotic_indices: np.ndarray, original indices where dilated semantic == 1
    	cycb_in_mitosis: np.ndarray, intensity values in dilated mitotic window
        deg_rate: np.ndarray, falling limb of derivative of CyclinB trace over mitotic indices
        max_cycb: timepoint at which CyclinB reaches maximum abundance (intensity)
    	success: bool, whether extraction was successful
    """
    dilated_semantic = binary_dilation(semantic == 1, structure=np.ones(dilation_size)).astype(int)
    mitotic_indices = np.where(dilated_semantic == 1)[0]
    cycb_in_mitosis = intensity[dilated_semantic == 1]
    deg_rate = derivative[dilated_semantic == 1]
    max_cycb = np.argmax(cycb_in_mitosis)
    deg_rate = -deg_rate[max_cycb: ]
    
    if len(cycb_in_mitosis) == 0:
        return None, None, False
    
    return mitotic_indices, cycb_in_mitosis, deg_rate, max_cycb, True


def _extract_falling_limb(deg_rate):
    """
    Extract falling limb of degradation rate for curve fitting.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	deg_rate: np.ndarray, degradation rate array
    OUTPUTS:
    	max_deg: int, index of maximum degradation rate
    	deg_rate_fall: np.ndarray, falling limb of degradation rate
    	success: bool, whether extraction was successful
    """
    max_deg = np.argmax(deg_rate)
    deg_rate_fall = np.maximum(deg_rate[max_deg:], 0)  # rectify
    
    if len(deg_rate_fall) < 3:
        return None, None, False
    
    return max_deg, deg_rate_fall, True


def gauss_changept(smooth_intensity, derivative, semantic):
    """
    Detect change point in CyclinB degradation by fitting half-gaussian to falling degradation rate.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	intensity: np.ndarray, CyclinB intensity trace over time
    	semantic: np.ndarray, binary classification (1=mitotic, 0=non-mitotic)
    OUTPUTS:
    	changept: float, estimated change point (mu - sigma) in original intensity array indexing, or np.nan if error
    """

    try:
        # Extract mitotic data
        mitotic_indices, cycb_in_mitosis, deg_rate, max_cycb, success = _extract_mitotic_data(smooth_intensity, semantic, derivative)
        if not success:
            print("[gauss_changept] No mitotic data found in semantic trace")
            return np.nan
                
        # Extract falling limb
        max_deg, deg_rate_fall, success = _extract_falling_limb(deg_rate)
        if not success:
            print("[gauss_changept] Insufficient data for falling limb fitting")
            return np.nan

        # Fit half-gaussian
        t = np.arange(len(deg_rate_fall))
        A0 = np.max(deg_rate_fall)         # peak value as amplitude
        mu0 = 0                            # mu should be 0 for t (start of falling limb)
        sigma0 = len(deg_rate_fall) / 4    # reasonable initial guess for sigma

        popt, _ = curve_fit(falling_half_gaussian, t, deg_rate_fall, p0=[A0, mu0, sigma0])
        _, mu_fit, sigma_fit = popt
        
        # Calculate fit quality metrics
        fitted_values = falling_half_gaussian(t, *popt)
        residuals = deg_rate_fall - fitted_values
        r_squared = 1 - np.sum(residuals**2) / np.sum((deg_rate_fall - np.mean(deg_rate_fall))**2)
        
        # Check fit quality - return nan if poor fit
        if r_squared < 0.5:  # Adjust threshold as needed
            print(f"[gauss_changept] Poor fit detected (R² = {r_squared:.3f})")
            return np.nan
        
        
        changept_rel = mu_fit - sigma_fit/2
        # Convert back to original array indexing
        changept = mitotic_indices[max_cycb] + max_deg + changept_rel
        
        # Validate that changept is non-negative
        if changept < 0:
            print("[gauss_changept] Negative changept detected, likely due to poor curve fitting")
            return np.nan
        
        return changept
        
    except Exception as error:
        print(error)
        print("[gauss_changept] Curve fitting failed")
        return np.nan


# DEPRECATED: Curvature-based change point detection has been removed due to poor performance
# The gaussian fitting method (gauss_changept) is the recommended approach


def visualize_changepts(intensities, semantics, derivatives, changepts, max_plots=12):
    """
    Visualize CyclinB traces, degradation rates, and change points on single plots.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    	intensities: list[np.ndarray], list of CyclinB intensity traces
    	semantics: list[np.ndarray], list of semantic traces
    	changepts: list[float], list of change points (can contain np.nan)
    	max_plots: int, maximum number of plots to show (default 6)
    OUTPUTS:
    	fig: matplotlib figure object
    """
    # Randomly sample traces
    n_total = len(intensities)
    n_plots = min(n_total, max_plots)
    
    # Generate random indices without replacement
    # Use a fixed seed for reproducible plots when comparing methods
    np.random.seed(42)  # You can change this seed value
    random_indices = np.random.choice(n_total, size=n_plots, replace=False)
    
    # Create subplots grid sized to number of plots requested
    n_rows = math.ceil(n_plots / 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 8))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    for i, idx in enumerate(random_indices):
        intensity = intensities[idx]
        semantic = semantics[idx]
        changept = changepts[idx]
        derivative = derivatives[idx]
        
        # Extract data using helper functions
        mitotic_indices, cycb_in_mitosis, deg_rate, max_cycb, success = _extract_mitotic_data(intensity, semantic, derivative)
        
        if not success:
            # Plot original trace if no mitotic data
            axes[i].plot(intensity, 'b-', label='CyclinB')
            #axes[i].set_title(f'Trace {i+1}: No mitotic data')
            continue
        
        # Create twin axes for dual y-axis plot
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        # Plot CyclinB trace in dilated window (left y-axis)
        line1 = ax1.plot(mitotic_indices, cycb_in_mitosis, color='#9A3324', label='CyclinB', linewidth=2)
        
        if success:
            deg_indices = mitotic_indices[max_cycb:]
            line2 = ax2.plot(deg_indices, deg_rate, color='#575294', label='Degradation rate', linewidth=2)
            
            # Add change point line if valid
            if not np.isnan(changept):
                ax1.axvline(x=changept, color='gray', linestyle='--', linewidth=2, label=f'Change point: {changept:.1f}')
                ax2.axvline(x=changept, color='gray', linestyle='--', linewidth=2)
        else:
            ax2.text(0.5, 0.5, 'No degradation data', ha='center', va='center', transform=ax2.transAxes)
                
        # Move ticks inside and set legend to bottom left
        ax1.tick_params(axis='both', direction='in')
        ax2.tick_params(axis='y', direction='in')
        
        # Combine legends and place at bottom left
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    
    # Hide unused subplots in the created grid
    total_axes = n_rows * 3
    for i in range(n_plots, total_axes):
        axes[i].set_visible(False)
    
    plt.tight_layout()

    return fig