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

if __name__ == "__main__":

    root_dir = Path("/Volumes/umms-ajitj/anishjv/cyclinb_analysis/20250621-cycb-noc")
    gfp_dir = Path("/Volumes/SharedHITSX/cdb-Joglekar-Lab-GL/Anish_Virdi/CycB_dynamics/ixn/20250621-cycb-noc/cycb/2025-06-21/20452") #sometimes the fluorescence images are in a different directory
    inference_dirs = [
        obj.path
        for obj in os.scandir(root_dir)
        if "_inference" in obj.name and obj.is_dir()
    ]

    file_pairs = [] # list of (gfp_path, seg_path) tuples
    for dir_path in inference_dirs:
        name_stub_match = re.search(
            r"[A-H]([1-9]|[0][1-9]|[1][0-2])_s(\d{2}|\d{1})", str(dir_path)
        )
        if not name_stub_match:
            print(f"Skipping inference directory with unusual name: {dir_path}")
            continue

        name_stub = name_stub_match.group()
        name_stub_prefix = str(name_stub) + "_"
        
        seg_path_list = glob.glob(f"{dir_path}/*semantic_movie.tif")
        gfp_path_list = [
            path
            for path in glob.glob(f"{gfp_dir}/*GFP.tif")
            if str(name_stub_prefix) in path
        ]


        if len(seg_path_list) == 1 and len(gfp_path_list) == 1:
            file_pairs.append((gfp_path_list[0], seg_path_list[0]))
        else:
            print(f"Warning: Could not find 1:1 match for {name_stub}. Found {len(gfp_path_list)} GFP files and {len(seg_path_list)} SEG files. Skipping.")

    # Intensity Map
    intensity_map_files = glob.glob(f"{root_dir}/*intensity_map.tif")
    if not intensity_map_files:
        print(f"ERROR: No 'intensity_map.tif' found in {root_dir}. Quitting.")
        sys.exit(1)
    intensity_map_f = intensity_map_files[0]
    intensity_map = tif.imread(intensity_map_f, key=0)

    # Background Map (corrected typo from user input)
    bkg_map_files = glob.glob(f"{root_dir}/*background_map.tif") 
    if not bkg_map_files:
        print(f"ERROR: No 'background_map.tif' found in {root_dir}. Quitting.")
        sys.exit(1)
    bkg_map_f = bkg_map_files[0]
    bkg_map = tif.imread(bkg_map_f, key=0)

    all_date_wells = []
    all_pos = []
    all_intensities = []
    all_indices = []
    all_validity = []

    # 2. Iterate through the file pairs
    for gfile, sfile in file_pairs:
        
        well_pos = re.search(r"[A-H]([1-9]|[0][1-9]|[1][0-2])_s(\d{2}|\d{1})", str(gfile)).group()
        date = re.search(r"20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])", str(gfile)).group()
        well = well_pos.split('_')[0]
        pos = well_pos.split('_')[1]
        date_well = date + '-' + well
        
        print(f"Processing: {date} - {well_pos}")

        gfp = tif.imread(gfile, key=0)
        seg = tif.imread(sfile, key=0)
        if seg.shape != gfp.shape:
            seg = resize(
                seg, 
                gfp.shape, 
                order=0, 
                preserve_range=True, 
                anti_aliasing=False
            ).astype(seg.dtype)

        corr_gfp = (gfp.astype(float) - bkg_map) / intensity_map
        intensities, indices, validity, _ = measure_diagonal_network_path(corr_gfp, seg)
        num_points = len(intensities)
        
        all_date_wells.extend([date_well] * num_points)
        all_pos.extend([pos] * num_points)
        all_intensities.extend(intensities)
        all_indices.extend(indices)
        all_validity.extend(validity)

    # 4. Create DataFrame
    df_background_qc = pd.DataFrame({
        "date-well": all_date_wells,
        "pos": all_pos,
        "intensities": all_intensities,
        "indices": all_indices,      # The projected diagonal position (0 to 2048)
        "validity": all_validity     # True if real background, False if jumped over cell
    })   


    fig = plot_background_panels(df_background_qc, mode = 'intra-well', cols = 3, bin_width = 50)
    fig.savefig(f'{root_dir}/{date}_background_intra_well.png', dpi=500)

    fig = plot_background_panels(df_background_qc, mode = 'inter-well', cols = 3, bin_width = 50)
    fig.savefig(f'{root_dir}/{date}_background_inter_well.png', dpi=500)