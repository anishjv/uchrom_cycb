import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skimage.draw import line, polygon
from scipy.stats import binned_statistic
from skimage.morphology import binary_dilation, disk
import tifffile as tif
from skimage.transform import resize
from pathlib import Path
import os
import re
import glob
import math
import sys

def measure_diagonal_network_path(img: np.ndarray, seg: np.ndarray, strip_width: int = 10):
    """
    Measures the intensity profile along a diagonal path in an image, excluding segmented regions.

    The path is either a single line or a polygon strip of a given width. The coordinates
    are projected onto a 1D diagonal index.
    ---------------------------------------------------------------------------------------------
    INPUTS:
        img: np.ndarray, the image intensity data (2D).
        seg: np.ndarray, the segmentation mask (2D, values > 0 are cells).
        strip_width: int, the width of the diagonal strip to sample. If <= 1, a single line is used.
    OUTPUTS:
        path_intensities: np.ndarray, intensities sampled along the path.
        diag_indices: np.ndarray, 1D projected index along the diagonal for each sampled point.
        is_valid_bg: np.ndarray, boolean mask indicating if the sampled point is background (not blocked by a cell).
        path_coords: tuple (rr, cc), the (row, column) coordinates of the sampled path.
    """
    h, w = img.shape
    
    # 1. Create Exclusion Mask
    mask_cells = seg > 0
    # The dilation makes the exclusion zone around the cells slightly larger
    mask_cells = binary_dilation(mask_cells, disk(5))
    
    # 2. Get Coordinates
    if strip_width <= 1:
        # Single line path
        rr, cc = line(0, 0, h-1, w-1)
    else:
        # Polygon strip path
        dr_diag = h - 1
        dc_diag = w - 1
        length = np.hypot(dr_diag, dc_diag)
        # Perpendicular unit vector components
        r_perp = -dc_diag / length
        c_perp = dr_diag / length
        
        half_w = strip_width / 2.0
        # Offsets for the perpendicular width
        off_r = r_perp * half_w
        off_c = c_perp * half_w
        
        # Define the four corners of the polygon strip
        p1 = (0 - off_r, 0 - off_c)
        p2 = (0 + off_r, 0 + off_c)
        p3 = (h-1 + off_r, w-1 + off_c)
        p4 = (h-1 - off_r, w-1 - off_c)
        
        poly_r = np.array([p1[0], p2[0], p3[0], p4[0]])
        poly_c = np.array([p1[1], p2[1], p3[1], p4[1]])
        
        # Get coordinates within the polygon
        rr, cc = polygon(poly_r, poly_c, shape=img.shape)

    # 3. Extract Data
    path_intensities = img[rr, cc]
    
    # 4. Validity
    # Check which sampled points are blocked by the exclusion mask
    is_blocked = mask_cells[rr, cc]
    is_valid_bg = ~is_blocked
    
    # 5. Project to Diagonal Axis
    # Project (r, c) onto the diagonal vector (dy, dx)
    dy = h - 1
    dx = w - 1
    diag_len_sq = dy*dy + dx*dx
    # Normalized projection 't' along the diagonal (0 to 1)
    projection_t = (rr * dy + cc * dx) / diag_len_sq
    # Scale to max dimension for a meaningful index (0 to max(h,w))
    max_dim = max(h, w)
    diag_indices = projection_t * max_dim
    
    # 6. Format
    path_coords = (rr, cc)
    
    return path_intensities, diag_indices, is_valid_bg, path_coords

def plot_background_quality_scan(img: np.ndarray, seg: np.ndarray, strip_width: int = 10, bin_width: int = 50):
    """
    Generates a two-panel plot: one showing the sampled diagonal path (and excluded cells), 
    and the other showing the intensity profile along that path, with binned mean and SEM.
    ---------------------------------------------------------------------------------------------
    INPUTS:
        img: np.ndarray, the image intensity data (2D).
        seg: np.ndarray, the segmentation mask (2D).
        frame_idx: int, index of the frame (for title only, currently unused in the function body).
        strip_width: int, the width of the diagonal strip to sample.
        bin_width: int, the width in pixels for binning the intensity data along the diagonal.
    OUTPUTS:
        fig: matplotlib.figure.Figure, the generated figure object.
        ax1, ax2: tuple, the axes objects for the path visualization and the intensity profile.
    """
    intensities, diag_indices, is_valid_bg, path_coords = measure_diagonal_network_path(
        img, seg, strip_width=strip_width
    )
    
    valid_intensities = intensities[is_valid_bg]
    valid_indices = diag_indices[is_valid_bg] 
    
    bg_median = np.median(valid_intensities) if len(valid_intensities) > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Plot 1: Path Visualization ---
    mask_cells = seg > 0
    # Show the cell exclusion mask (black for cells, white for background)
    ax1.imshow(mask_cells, cmap='gray_r') 
    y_path, x_path = path_coords
    # Color-code the sampled points (Cyan=Valid Background, Magenta=Blocked)
    colors = np.where(is_valid_bg, 'cyan', 'magenta')
    s_size = 1 if strip_width <= 1 else 0.1
    ax1.scatter(x_path, y_path, c=colors, s=s_size, label='Sample')
    ax1.set_title(f"Diagonal Sampling (Width={strip_width})\n(Cyan=Valid, Magenta=Blocked)")
    ax1.axis('off')
    
    # --- Plot 2: Intensity Profile ---
    if len(valid_intensities) > 0:
        # Scatter plot of all valid background intensity points
        ax2.scatter(valid_indices, valid_intensities, color='gray', s=2, alpha=0.1, label='Background intensities')
        
        max_dim = max(img.shape)
        num_bins = int(max_dim / bin_width)
        bins = np.linspace(0, max_dim, num_bins + 1)
        
        # Binning statistics (Mean, Std Dev, Count)
        bin_y_means, _, _ = binned_statistic(valid_indices, valid_intensities, statistic='mean', bins=bins)
        # Calculate mean x-position for plotting the center of the bin
        bin_x_means, _, _ = binned_statistic(valid_indices, valid_indices, statistic='mean', bins=bins)
        bin_stds, _, _ = binned_statistic(valid_indices, valid_intensities, statistic='std', bins=bins)
        bin_counts, _, _ = binned_statistic(valid_indices, valid_indices, statistic='count', bins=bins)
        
        # Calculate Standard Error of the Mean (SEM = StdDev / sqrt(Count))
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_sem = bin_stds / np.sqrt(bin_counts)
        
        # Filter out bins with no data (NaNs)
        valid_mask = ~np.isnan(bin_y_means)
        x_plot = bin_x_means[valid_mask]
        y_plot = bin_y_means[valid_mask]
        sem_plot = bin_sem[valid_mask]
        
        # Plot binned mean and SEM area
        ax2.plot(x_plot, y_plot, "red", lw=2, label=f"Binned Mean (~{bin_width}px)")
        ax2.fill_between(x_plot, y_plot - sem_plot, y_plot + sem_plot, color='red', alpha=0.2, label='SEM')

    ax2.set_xlabel("Diagonal Position")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.set_title(f"Profile (Median={bg_median:.1f})")
    ax2.set_xlim(0, max(img.shape))
    ax2.legend()
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_comparative_background_profiles(
    df: pd.DataFrame, 
    filter_col: str,
    filter_val: str,
    color_col: str,
    ax: plt.Axes = None,
    cmap_name: str = "Set2",
    max_image_dim: int = 2048,
    bin_width: int = 50
):
    """
    Plots multiple binned mean background profiles (with SEM) on a single axis for comparison.
    
    This is the base function used by plot_background_panels to create a single subplot.
    ---------------------------------------------------------------------------------------------
    INPUTS:
        df: pd.DataFrame, Long-form DataFrame containing 'intensities', 'indices', 'validity', 
            and columns for filtering and coloring.
        filter_col: str, The column name used to select data for this specific plot (panel grouping).
        filter_val: str, The specific value in 'filter_col' to include in this plot.
        color_col: str, The column name used to group and color the individual lines (curve grouping).
        ax: plt.Axes, The matplotlib axis to draw on. If None, a new figure/axis is created.
        cmap_name: str, Colormap name to use for line colors.
        max_image_dim: int, Used to set the x-axis limit and binning range.
        bin_width: int, The size in pixels for the binning of the diagonal index.
    OUTPUTS:
        fig: matplotlib.figure.Figure or None, The figure object.
        ax: plt.Axes, The modified axis object.
    """
    group_data = df[df[filter_col] == filter_val]
    
    if group_data.empty:
        # Create dummy plot if no data
        if ax is not None:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        return None, ax

    unique_groups = sorted(group_data[color_col].unique())
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    # Color palette selection
    if len(unique_groups) > 10 and cmap_name == "tab10":
        palette = sns.color_palette("husl", n_colors=len(unique_groups))
    else:
        palette = sns.color_palette(cmap_name, n_colors=len(unique_groups))
        
    color_map = dict(zip(unique_groups, palette))
    bins = np.linspace(0, max_image_dim, int(max_image_dim/bin_width) + 1)
    
    for group in unique_groups:
        subset = group_data[group_data[color_col] == group]
        valid_subset = subset[subset["validity"] == True]
        
        if valid_subset.empty:
            continue
            
        x_vals = valid_subset["indices"].values
        y_vals = valid_subset["intensities"].values
        color = color_map[group]

        # Binned Mean + SEM calculation
        if len(x_vals) > 0:
            bin_y_means, _, _ = binned_statistic(x_vals, y_vals, statistic='mean', bins=bins)
            bin_x_means, _, _ = binned_statistic(x_vals, x_vals, statistic='mean', bins=bins)
            bin_stds, _, _ = binned_statistic(x_vals, y_vals, statistic='std', bins=bins)
            bin_counts, _, _ = binned_statistic(x_vals, x_vals, statistic='count', bins=bins)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                bin_sem = bin_stds / np.sqrt(bin_counts)
            
            valid_mask = ~np.isnan(bin_y_means)
            x_plot = bin_x_means[valid_mask]
            y_plot = bin_y_means[valid_mask]
            sem_plot = bin_sem[valid_mask]
            
            # Plot the binned mean line and fill the SEM area
            ax.plot(x_plot, y_plot, color=color, linewidth=2.5, linestyle='-', label=f"{group}")
            ax.fill_between(x_plot, y_plot - sem_plot, y_plot + sem_plot, color=color, alpha=0.33)
            
    ax.set_xlabel("Diagonal Pos")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{filter_val}", fontsize=10) # Simplified title for panels
    ax.set_xlim(0, max_image_dim)
    
    # Simple Legend
    ax.legend(loc='upper right', fontsize=6, frameon=False)
    
    return fig, ax

def plot_background_panels(
    df: pd.DataFrame, 
    mode: str = 'intra-well', # 'intra-well' or 'inter-well'
    cols: int = 4,
    max_image_dim: int = 2048,
    bin_width: int = 50
):
    """
    Creates a large figure with multiple panels to visualize background variance, 
    either within a well (stage stability) or across wells/experiments (batch consistency).
    ---------------------------------------------------------------------------------------------
    INPUTS:
        df: pd.DataFrame, Long-form DataFrame containing background profile data.
        mode: str, Plotting mode:
            'intra-well': Panels = Date-Wells, Curves = Positions. Checks stage stability.
            'inter-well': Panels = Positions, Curves = Date-Wells. Checks batch consistency.
        cols: int, Number of columns in the subplot grid.
        max_image_dim: int, Maximum image dimension (used for x-axis range).
        bin_width: int, The width in pixels for binning the intensity data.
    OUTPUTS:
        fig: matplotlib.figure.Figure or None, The generated figure object containing the panels.
    """
    
    # 1. Determine Variables based on mode
    if mode == 'intra-well':
        panel_col = 'date-well'  # Each subplot is a specific well (date-well)
        curve_col = 'pos'        # We compare positions (s1, s2, ...) inside that well
        print("Generating Intra-Well Variance Plot (Panel per Date-Well)...")
        
    elif mode == 'inter-well':
        panel_col = 'pos'        # Each subplot is a specific position (s1, s2, ...)
        curve_col = 'date-well'  # We compare experiments (date-wells) for that position
        print("Generating Inter-Well Variance Plot (Panel per Position)...")
        
    else:
        raise ValueError("mode must be 'intra-well' or 'inter-well'")
        
    # 2. Get unique panels
    panels = sorted(df[panel_col].unique())
    n_panels = len(panels)
    
    if n_panels == 0:
        print("No data found.")
        return None
    
    # 3. Setup Grid
    rows = math.ceil(n_panels / cols)
    fig_height = rows * 4
    fig_width = cols * 5
    
    # Create the grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    # 4. Iterate and Plot
    for i, panel_val in enumerate(panels):
        ax = axes_flat[i]
        
        plot_comparative_background_profiles(
            df=df,
            filter_col=panel_col,
            filter_val=panel_val,
            color_col=curve_col,
            ax=ax,
            max_image_dim=max_image_dim,
            bin_width=bin_width
        )
        
    # 5. Cleanup empty axes
    # Hide any remaining axes that didn't get a plot
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
        
    plt.tight_layout()
    return fig

def process_movie(
    gfp_path, seg_path, xlsx_path, 
    bkg_map, intensity_map,
    diagnostic_container # Dictionary to append lists for QC plots
):
    """
    Loads a single movie, calculates offsets for every frame, updates the analysis Excel file,
    and extracts diagnostic data from the first frame.
    ---------------------------------------------------------------------------------------------
    INPUTS:
        gfp_path: str, Path to the GFP movie TIFF file.
        seg_path: str, Path to the segmentation movie TIFF file.
        xlsx_path: str, Path to the analysis results Excel file.
        bkg_map: np.ndarray, 2D background maps.
        intensity_map: np.ndarray, 2D intensity correction map.
        diagnostic_container: dict, Shared dictionary to store plotting data from Frame 0.
    OUTPUTS:
        None (Modifies the Excel file in place and updates diagnostic_container).
    """
    
    # Metadata
    well_pos = re.search(r"[A-H]([1-9]|[0][1-9]|[1][0-2])_s(\d{2}|\d{1})", str(gfp_path)).group()
    date = re.search(r"20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])", str(gfp_path)).group()
    well = well_pos.split('_')[0]
    pos = well_pos.split('_')[1]
    date_well = f"{date}-{well}"
    
    print(f"  Processing: {date_well} {pos}...")

    # Load Images
    gfp_stack = tif.imread(gfp_path)
    seg_stack = tif.imread(seg_path)
    n_frames = gfp_stack.shape[0]

    frame_offsets = {} 
    for t in range(n_frames):

        img = gfp_stack[t]
        seg = seg_stack[t] if len(seg_stack) > t else seg_stack[-1]

        if seg.shape != img.shape:
            seg = resize(seg, img.shape, order=0, preserve_range=True, anti_aliasing=False).astype(seg.dtype)
        
        # Correction
        corr_img = (img.astype(float) - bkg_map) / intensity_map
        intensities, indices, is_valid, _ = measure_diagonal_network_path(corr_img, seg)
        
        # --- A. Offset Calculation (For Excel) ---
        valid_pixels = intensities[is_valid]
        if len(valid_pixels) > 0:
            print(f'Offest sampled from {len(valid_pixels)} pixel values')
            offset_val = np.mean(valid_pixels)
        else:
            print(f'{len(valid_pixels)} valid pixels found; setting offset to np.nan')
            offset_val = np.nan
        frame_offsets[t] = offset_val
        
        if t == 0:
            num_points = len(intensities)
            diagnostic_container['date_wells'].extend([date_well] * num_points)
            diagnostic_container['pos'].extend([pos] * num_points)
            diagnostic_container['intensities'].extend(intensities - offset_val)
            diagnostic_container['indices'].extend(indices)
            diagnostic_container['validity'].extend(is_valid)

    df = pd.read_excel(xlsx_path, engine='openpyxl')
    if 'frame' in df.columns:
        df['offset'] = df['frame'].map(frame_offsets)
        df.to_excel(xlsx_path, index=False, engine='openpyxl')


def main():
    # --- Config ---
    root_dir_path = "/nfs/turbo/umms-ajitj/anishjv/cyclinb_analysis/20251028-cycb-gsk"
    gfp_dir_path = "/nfs/turbo/umms-ajitj/anishjv/cyclinb_analysis/20251028-cycb-gsk"
    

    print("Loading Maps...")
    try:
        intensity_map = tif.imread(glob.glob(f"{root_dir_path}/*intensity_map.tif")[0])
        if intensity_map.ndim == 3: intensity_map = intensity_map[0]
        bkg_map_stack = tif.imread(glob.glob(f"{root_dir_path}/*background_map.tif")[0])
    except IndexError:
        print("CRITICAL: Maps not found.")
        sys.exit(1)

    root_dir = Path(root_dir_path)
    inference_dirs = [obj.path for obj in os.scandir(root_dir) if "_inference" in obj.name and obj.is_dir()]
    
    # Container for plotting data
    qc_data = {
        'date_wells': [], 'pos': [], 'intensities': [], 'indices': [], 'validity': []
    }

    print(f"Found {len(inference_dirs)} inference directories.")

    for dir_path in inference_dirs:
        name_stub_match = re.search(r"[A-H]([1-9]|[0][1-9]|[1][0-2])_s(\d{2}|\d{1})", str(dir_path))
        if not name_stub_match: continue
        name_stub = name_stub_match.group()
        
        seg_files = glob.glob(f"{dir_path}/*semantic_movie.tif")
        xlsx_files = glob.glob(f"{dir_path}/*analysis.xlsx")
        name_stub_prefix = str(name_stub) + "_"
        gfp_files = [p for p in glob.glob(f"{gfp_dir_path}/*GFP.tif") if str(name_stub_prefix) in p]

        if len(seg_files) == 1 and len(gfp_files) == 1 and len(xlsx_files) == 1:
            process_movie(
                gfp_files[0], seg_files[0], xlsx_files[0],
                bkg_map_stack, intensity_map, qc_data
            )
        else:
            print(f"Skipping {name_stub}: Files incomplete.")

    print("Generating Diagnostic Plots...")
    df_background_qc = pd.DataFrame({
        "date-well": qc_data['date_wells'],
        "pos": qc_data['pos'],
        "intensities": qc_data['intensities'],
        "indices": qc_data['indices'],
        "validity": qc_data['validity']
    })
    
    if not df_background_qc.empty:
        # Intra-Well Variance (Stage Stability)
        fig1 = plot_background_panels(df_background_qc, mode='intra-well', cols=3, bin_width=50)
        fig1.savefig(f'{root_dir_path}/background_qc_intra_well.png', dpi=300)
        plt.close(fig1)
        
        # Inter-Well Variance (Batch Consistency)
        fig2 = plot_background_panels(df_background_qc, mode='inter-well', cols=3, bin_width=50)
        fig2.savefig(f'{root_dir_path}/background_qc_inter_well.png', dpi=300)
        plt.close(fig2)
        
        print("Done. Plots saved.")
    else:
        print("No valid background data collected for plotting.")

if __name__ == "__main__":
    main()