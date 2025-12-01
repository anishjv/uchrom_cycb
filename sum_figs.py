import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
import seaborn as sns
import glob
import os
from deg_analysis import *


def create_treatmap(
    unpacked_df: pd.DataFrame, experiments: pd.DataFrame, date: Optional[int] = None
):

    wells = unpacked_df["date-well"].unique()
    if date:
        experiments = experiments[experiments["Date"] == date]
    else:
        pass
    experiments["date-well"] = (
        experiments["Date"].astype(str) + "-" + experiments["Well"].astype(str)
    )

    shared = set(wells) & set(experiments["date-well"])
    treatments = experiments["Treatment"].unique()

    mapper = {}
    for treat in treatments:
        mapper[treat] = [
            well
            for well in shared
            if experiments.loc[experiments["date-well"] == well, "Treatment"].iloc[0]
            == treat
        ]

    return mapper


def unaligned_chromatin_hists(unpacked_df: pd.DataFrame, mapper: dict):

    palette = sns.color_palette("Set2", n_colors=len(mapper))
    color_map = dict(zip(mapper.keys(), palette))

    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    for treat, well in mapper.items():

        group_data = unpacked_df[unpacked_df["date-well"].isin(well)]
        group_data = group_data[group_data["phase_flag"] == "slow"]
        if 'Noc' not in treat:
            group_data = group_data[group_data["achromatin"] > 75]

        sns.kdeplot(
            data=group_data,
            x="uchromatin",
            ax=ax,
            label=treat,
            color=color_map[treat],
            fill=True,
            alpha=0.1,
            linewidth=1,
        )

    # Remove all y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", which="both", length=0)

    # Turn on minor ticks for x-axis
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_xlim(left=-300, right=2000)

    # Keep only bottom spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_ylabel("")
    ax.set_xlabel(r"Unaligned Chromosome Area ($pixels^2$)")
    ax.legend()

    plt.tight_layout()

    return fig


def cycb_hists(unpacked_df: pd.DataFrame, mapper: dict):

    palette = sns.color_palette("Set2", n_colors=len(mapper))
    color_map = dict(zip(mapper.keys(), palette))
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    for treat, well in mapper.items():

        group_data = unpacked_df[unpacked_df["date-well"].isin(well)]

        sns.kdeplot(
            data=group_data,
            x="cycb",
            ax=ax,
            label=treat,
            color=color_map[treat],
            fill=True,
            alpha=0.1,
            linewidth=1,
        )

    # Remove all y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", which="both", length=0)

    # Turn on minor ticks for x-axis
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_xlim(left=-30, right=100)

    # Keep only bottom spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_ylabel("")
    ax.set_xlabel(r"Cyclin B (a.u.)")
    ax.legend()

    plt.tight_layout()
    return fig


def time_in_mitosis_hists(unpacked_df: pd.DataFrame, mapper: dict):

    palette = sns.color_palette("Set2", n_colors=len(mapper))
    color_map = dict(zip(mapper.keys(), palette))
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    for treat, well in mapper.items():

        group_data = unpacked_df[unpacked_df["date-well"].isin(well)]
        cell_ids = group_data["tracking_id"].unique()
        durations = []
        for cell in cell_ids:
            pos_in_mitosis = group_data[group_data["tracking_id"] == cell][
                "pos_in_mitosis"
            ]
            pos_in_mitosis = list(pos_in_mitosis)
            rel_step = pos_in_mitosis[1] - pos_in_mitosis[0]
            time_in_mitosis = 4 / rel_step
            durations.append(time_in_mitosis)

        # Plot KDE of durations
        sns.kdeplot(
            x=durations,
            ax=ax,
            label=treat,
            color=color_map[treat],
            fill=True,
            alpha=0.3,
            linewidth=1,
        )
    # Remove all y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", which="both", length=0)

    # Turn on minor ticks for x-axis
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_xlim(left=-100, right=1000)

    # Keep only bottom spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_ylabel("")
    ax.set_xlabel(r"Time in Mitosis (min.)")
    ax.legend()

    plt.tight_layout()
    return fig

def well_wise_cycb_consistency(unpacked_df: pd.DataFrame, mapper: dict):
    """
    Generates one figure per TREATMENT.
    Each figure contains KDE plots for every individual WELL in that treatment group.
    Used to check for inter-well consistency (quality control of background correction).
    """
    
    plt.rcParams.update({"font.size": 8})
    figures = {} # Store figures to return

    for treatment, wells in mapper.items():
        # Create a new figure for this treatment
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        
        # Create a distinct color for each WELL within this treatment
        # 'husl' is good for distinct categorical colors
        palette = sns.color_palette("husl", n_colors=len(wells))
        
        # Sort wells for consistent legend order
        wells = sorted(wells)

        for i, well in enumerate(wells):
            well_data = unpacked_df[unpacked_df["date-well"] == well]
            
            # Skip if well has no data after filtering
            if well_data.empty:
                continue

            sns.kdeplot(
                data=well_data,
                x="cycb",
                ax=ax,
                label=well,  # Legend will be the specific well ID (e.g. 20250612-A01)
                color=palette[i],
                fill=True,
                alpha=0.1,
                linewidth=1.5,
            )

        # Apply same styling as the aggregate plots for comparison
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", length=0)

        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", length=2)
        
        # Use same limits as aggregate to make visual comparison easy
        ax.set_xlim(left=-30, right=100)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_ylabel("")
        ax.set_xlabel(r"Cyclin B (a.u.)")
        
        # Title the plot with the Treatment name so we know what we are looking at
        ax.set_title(f"Consistency Check: {treatment}", fontsize=10)
        
        # Position legend outside if there are many wells, or inside if few
        ax.legend(frameon=False, fontsize=6)

        plt.tight_layout()
        figures[treatment] = fig

    return figures


if __name__ == "__main__":

    exp_date = 20251028
    save_path = "/Volumes/umms-ajitj/anishjv/cyclinb_analysis/20251028-cycb-gsk"
    os.chdir(save_path)

    paths = glob.glob(os.path.join(save_path, "**", "*chromatin.xlsx"), recursive=True)
    experiment_reg = pd.read_excel("/Users/whoisv/Desktop/experiment_registry.xlsx")
    use_changept = []
    for path in paths:
        well = re.search(r"[A-H]([1-9]|0[1-9]|1[0-2])_s(\d{1,2})", str(path)).group()[0]
        date = re.search(
            r"20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])", str(path)
        ).group()
        date = int(date)

        treat = experiment_reg[(experiment_reg['Date'] == date) & 
                       (experiment_reg['Well'] == well)]

        if not treat.empty and 'kd' in treat.iloc[0]['Treatment'].lower():
            use_changept.append(False)
        else:
            use_changept.append(True)
        

    unpacked_df = create_aggregate_df(paths, use_changept=use_changept)
    mapper = create_treatmap(unpacked_df, experiment_reg, exp_date)

    # 1. Generate Standard Aggregate Plots
    chrom = unaligned_chromatin_hists(unpacked_df, mapper)
    cycb = cycb_hists(unpacked_df, mapper)
    durations = time_in_mitosis_hists(unpacked_df, mapper)

    # Save Aggregate Plots
    for fig, name in zip([chrom, cycb, durations], ["chrom.png", "cycb.png", "durations.png"]):
        if fig:
            id_name = f"{exp_date}_{name}"
            fig.savefig(os.path.join(save_path, id_name), dpi=300)
            plt.close(fig)

    # 2. Generate Well-Wise Consistency Plots
    consistency_figs = well_wise_cycb_consistency(unpacked_df, mapper)
    
    # Save Consistency Plots (One per treatment)
    for treatment_name, fig in consistency_figs.items():
        # Sanitize filename (remove spaces/slashes)
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', treatment_name)
        fname = f"{exp_date}_consistency_{safe_name}.png"
        
        print(f"Saving consistency check: {fname}")
        fig.savefig(os.path.join(save_path, fname), dpi=300)
        plt.close(fig)
