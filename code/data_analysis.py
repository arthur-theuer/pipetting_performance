"""Data analysis functions to calculate statistics for absorbance measurement results and visualize them in an HTML report."""

__author__ = "Arthur Theuer <arthur.theuer@outlook.com>"
__maintainer__ = "Arthur Theuer <arthur.theuer@outlook.com>"


import os
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_svg import FigureCanvasSVG

from scipy.stats import norm

from io import BytesIO


# GLOBAL VARIABLES THAT MIGHT NEED TO BE ADJUSTED QUICKLY:
USE_MEDIAN = True  # flag to use median instead of mean as central value for heatmaps
PLOT_SCALING = 2  # scaling factor for the plots, default is 2
AX_SPACING = 0.25  # size of the padding between heatmap subplots, default is 0.25
COLOR_THRESHOLD = 0.2  # fraction of failed wells below which the color of the fail sign switches from red to yellow

# Deviation settings:
DEV_PERCENTAGE = True  # flag to show relative deviation values as percentages, underlying data is never transformed
DEV_THRESHOLD = 0.05  # passing value for quick run
DEV_THRESHOLD_UPPER = 0.1  # second, less stringent passing value
DEV_CMAP_THRESHOLD = 0.1  # for most extreme value of the color map

# Coefficient of variation settings:
CV_PERCENTAGE = True  # flag to show all CV values as percentages, underlying data is never transformed
CV_THRESHOLD = 0.05  # value from which CV is colored yellow and are considered as failed in full run
CV_THRESHOLD_UPPER = 0.1  # second, less stringent CV passing value
CV_CMAP_THRESHOLD = 0.1  # for most extreme value of the color map

# Standard deviation settings:
SD_CMAP_THRESHOLD = 0.1  # for most extreme value of the color map


# Dictionary of heatmap text and outline formatting styles:
text_styles = {
    "style_hm_single_big": {
        "text": {"ha": "center", "va": "center", "color": "black", "fontsize": 7},
        "outline_default": {"foreground": "w", "linewidth": 2},
        "outline_highlight": {"foreground": "y", "linewidth": 2}},
    "style_hm_single_small": {
        "text": {"ha": "center", "va": "center", "color": "black", "fontsize": 6.5},
        "outline_default": {"foreground": "w", "linewidth": 1.5},
        "outline_highlight": {"foreground": "y", "linewidth": 1.5}},
    "style_hm_big": {
        "text": {"ha": "center", "va": "bottom", "color": "black", "fontsize": 7},
        "outline_default": {"foreground": "w", "linewidth": 2},
        "outline_highlight": {"foreground": "y", "linewidth": 2}},
    "style_hm_small": {
        "text": {"ha": "center", "va": "top", "color": "dimgray", "fontsize": 6.5},
        "outline_default": {"foreground": "w", "linewidth": 1.5},
        "outline_highlight": {"foreground": "y", "linewidth": 1.5}},
    }


def calculate_relative_absorbance(df: pd.DataFrame, use_median: bool = USE_MEDIAN) -> pd.DataFrame:
    """
    Calculates the relative deviation from the median or mean using plate reader absorbance measurements.

    :param df: Dataframe of absorbance values.
    :param use_median: Boolean flag to use median instead of mean. Defaults to global variable.
    :return: Deviation from the median or mean in the same format as the input.
    """
    if use_median:
        central_tendency = np.median(df.values.flatten())
    else:
        central_tendency = df.values.sum() / df.size

    df = ((df / central_tendency) - 1)  # values centered around 0

    return df


def convert_plate_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a dataframe in plate layout to a long format.

    :param df: Any dataframe of a microplate in the same format as the microplate.
    :return: A dataframe of the plate in long format (two columns).
    """
    df_long = df.copy()
    df_long.index.name = "well_row"
    df_long.columns.name = "well_col"

    # Melt the dataframe to get a long format:
    df_long.reset_index(inplace=True)
    df_long = df_long.melt(id_vars="well_row", var_name="well_col", value_name="value")
    # Combine row and column labels to get well indices:
    df_long["well"] = df_long["well_row"].astype(str) + df_long["well_col"].astype(str)
    df_long.drop(["well_row", "well_col"], axis=1, inplace=True)  # drop the original row and column labels
    df_long = df_long[["well", "value"]]  # reorder the columns

    return df_long


def compare_dataframe_shapes(df_list: list) -> bool:
    """
    Checks if all dataframes in a list have the same shape (as the first one).

    :param df_list: List of dataframes.
    :return: Boolean flag indicating if all dataframes have the same shape.
    """
    first_shape = df_list[0].shape
    return all(df.shape == first_shape for df in df_list)


def set_heatmap_layout(ax_widths: list, ax_heights: list, ax_spacing: float = AX_SPACING) -> tuple:
    """
    Sets the layout of a heatmap figure with multiple subplots ensuring the correct aspect ratio.

    :param ax_widths: List of the widths of the subplots ordered from top to bottom.
    :param ax_heights: List of the heights of the subplots ordered from left to right.
    :param ax_spacing: Adjustable spacing between the subplots, given in inches (= well size of plate).
    :return: Tuple of the GridSpec object and the width and height of the figure.
    """
    # What happens here is pure absurdity, all because matplotlib does not support absolute subplot dimensions:
    ax_wspaces = len(ax_widths) - 1  # number of horizontal white spaces, given by number of subplots
    ax_hspaces = len(ax_heights) - 1  # number of vertical white spaces, given by number of subplots

    mean_ax_width = np.mean(ax_widths)  # needed to make even/absolute horizontal spacing
    mean_ax_height = np.mean(ax_heights)  # needed to make even/absolute vertical spacing

    ax_wspacing = ax_spacing / mean_ax_width  # absolute horizontal spacing per space
    ax_hspacing = ax_spacing / mean_ax_height  # absolute vertical spacing per space

    # Total size of the figure, given by size of subplots and spacing (to ensure correct aspect ratio):
    fig_width = sum(ax_widths) + ax_wspaces * ax_spacing
    fig_height = sum(ax_heights) + ax_hspaces * ax_spacing

    # Create a grid for the heatmaps:
    gs = GridSpec(len(ax_heights), len(ax_widths),
                  height_ratios=ax_heights,
                  width_ratios=ax_widths,
                  wspace=ax_wspacing,
                  hspace=ax_hspacing)

    return gs, fig_width, fig_height


def set_dashed_spines(ax: plt.Axes) -> None:
    """
    Sets the spines of a matplotlib axis to dashed lines and the color to dimgray.
    Serves as a shortcut for redundant commands in multiple places.

    :param ax: Matplotlib axis object.
    """
    ax.tick_params(axis="both", colors="dimgray")
    for spine in ax.spines.values():
        spine.set_edgecolor("dimgray")
        spine.set_linestyle("dashed")


def set_all_heatmap_ticks(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Sets the ticks of a heatmap axis to the indices of a dataframe.
    Serves as a shortcut for redundant commands in multiple places.

    :param ax: Matplotlib axis object.
    :param df: Dataframe with the indices to be used as ticks.
    """
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)


def set_cell_text(ax: plt.Axes, text: str, style: str, pos: tuple) -> None:
    """
    Sets the text of a heatmap axis and applies custom styling.
    Serves as a shortcut for redundant commands in multiple places.

    :param ax: Matplotlib axis object.
    :param text: Text to be displayed.
    :param style: Dictionary key of the text style to be used.
    :param pos: Tuple of position values used to place the text, defaults to None.
    """
    kwargs = text_styles[style]

    x, y = pos
    t = ax.text(x, y, text, **kwargs["text"])
    t.set_path_effects([pe.withStroke(**kwargs["outline_default"])])


def set_dynamic_cell_text(ax: plt.Axes, array: np.ndarray, style: str, formatting: str = "", threshold: float = None) -> None:
    """
    Sets the text of a heatmap axis to the values of a numpy array and applies custom styling.
    Also accepts pandas dataframes, that are then transformed into numpy arrays.
    Serves as a shortcut for redundant commands in multiple places.

    :param ax: Matplotlib axis object.
    :param array: Numpy array with the values to be used as text.
    :param style: Dictionary key of the text style to be used.
    :param formatting: Formatting of the displayed array values, defaults to "".
    :param threshold: Threshold at which the outline coloring changes to yellow, defaults to None.
    """
    kwargs = text_styles[style]

    if isinstance(array, pd.DataFrame):
        array = array.values

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            t = ax.text(j, i, f"{array[i, j]:{formatting}}", **kwargs["text"])
            if threshold is None or (threshold > array[i, j] > -threshold):
                t.set_path_effects([pe.withStroke(**kwargs["outline_default"])])
            else:
                t.set_path_effects([pe.withStroke(**kwargs["outline_highlight"])])


def generate_quick_plate_heatmap(abs_df: pd.DataFrame, rel_df: pd.DataFrame, scaling: float = PLOT_SCALING) -> tuple:
    """
    Generates a heatmap of relative well absorbance values from a microplate absorbance readout.
    Works independently of the number of wells.

    :param abs_df: Data frame of absolute absorbance values in microplate format.
    :param rel_df: Data frame of relative absorbance values in microplate format.
    :param scaling: Scaling factor for the plot size, defaults to global variable.
    :return: A heatmap figure of well absorbance values and a figure width to base other plots on.
    """
    # Add average value and SD/CV dataframes for both axes of the plot:
    abs_df_mean_row = pd.DataFrame(abs_df.mean(axis=1), columns=["MEAN"])
    abs_df_mean_col = pd.DataFrame(abs_df.mean(axis=0), columns=["MEAN"]).T
    abs_df_sd_row = pd.DataFrame(abs_df.std(axis=1), columns=["SD"])
    abs_df_sd_col = pd.DataFrame(abs_df.std(axis=0), columns=["SD"]).T
    abs_df_cv_row = pd.DataFrame(abs_df.std(axis=1) / abs_df.mean(axis=1), columns=["CV"])
    abs_df_cv_col = pd.DataFrame(abs_df.std(axis=0) / abs_df.mean(axis=0), columns=["CV"]).T

    # Do the same for relative absorbance values:
    rel_df_mean_row = pd.DataFrame(rel_df.mean(axis=1), columns=["MEAN"])
    rel_df_mean_col = pd.DataFrame(rel_df.mean(axis=0), columns=["MEAN"]).T

    # GRIDSPEC CREATION ################################################################################################
    cbar_width = 0.5  # width of the color bar, adjustable, given in inches

    # List for width_ratios and mean_ax_width:
    ax_widths = [cbar_width, 0.5, abs_df.shape[1], abs_df_mean_row.shape[1], abs_df_sd_row.shape[1], abs_df_cv_row.shape[1]]
    # List for height_ratios and mean_ax_height:
    ax_heights = [abs_df_cv_col.shape[0], abs_df_sd_col.shape[0], abs_df_mean_col.shape[0], abs_df.shape[0]]

    # Use custom function to ensure correct aspect ratio and spacing between subplots:
    gs, fig_width, fig_height = set_heatmap_layout(ax_widths, ax_heights)

    """
    The following diagram shows the layout of the subplots and the grid.
    The grid is used to place the subplots.

        [0|1]                      [2]                        [3]   [4]   [5]
        ┌─┬─┬───────────────────────────────────────────────┬─────────────────┐
    [0] │C│ │                      CVs                      │    CV LEGEND    │
        ├─┼─┼───────────────────────────────────────────────┼─────────────────┤
    [1] │C│ │                      SDs                      │    SD LEGEND    │
        ├─┼─┼───────────────────────────────────────────────┼─────────────────┤
    [2] │ │ │                     MEANS                     │   LEGEND TEXT   │
        ├─┼─┼───────────────────────────────────────────────┼─────┬─────┬─────┤
        │ │ │                                               │     │     │     │
        │C│ │                                               │     │     │     │
        │O│ │                                               │     │     │     │
        │L│ │                                               │     │     │     │
        │O│ │                                               │     │     │     │
    [3] │R│ │                 PLATE HEATMAP                 │MEANS│ SDs │ CVs │
        │ │ │                     [3|2]                     │     │     │     │
        │B│ │                                               │     │     │     │
        │A│ │                                               │     │     │     │
        │R│ │                                               │     │     │     │
        │ │ │                                               │     │     │     │
        └─┴─┴───────────────────────────────────────────────┴─────┴─────┴─────┘
    """

    # Create the figure with correct aspect ratio and scaling, also scale down subplots to 80 percent and add title:
    fig = plt.figure(figsize=(fig_width / scaling, fig_height / scaling), dpi=300)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.suptitle("Relative deviation from the average (median) absorbance [%]")

    # PLATE HEATMAP (bottom left) ######################################################################################
    colormap = "RdBu_r"  # colormap for the heatmap
    rel_norm = TwoSlopeNorm(vmin=-DEV_CMAP_THRESHOLD, vcenter=0, vmax=+DEV_CMAP_THRESHOLD)  # normalization for heatmap

    ax1 = fig.add_subplot(gs[3, 2])
    ax1.imshow(rel_df, cmap=colormap, norm=rel_norm, aspect="auto")
    set_all_heatmap_ticks(ax1, rel_df)

    # AVERAGE PER ROW (to the right of the heatmap) ####################################################################
    ax2 = fig.add_subplot(gs[3, 3], sharey=ax1)
    ax2.imshow(rel_df_mean_row, cmap=colormap, norm=rel_norm, aspect="auto")
    ax2.tick_params(axis="y", left=False, labelleft=False)
    ax2.set_xticks(range(len(rel_df_mean_row.columns)))
    ax2.set_xticklabels(rel_df_mean_row.columns)

    # SD PER ROW (to the right of the heatmap) #########################################################################
    ax2a = fig.add_subplot(gs[3, 4], sharey=ax1)
    ax2a.imshow(abs_df_sd_row, cmap="Greys", vmin=0, vmax=SD_CMAP_THRESHOLD, aspect="auto")

    ax2a.tick_params(axis="y", left=False, labelleft=False)
    ax2a.set_xticks(range(len(abs_df_sd_row.columns)))
    ax2a.set_xticklabels(abs_df_sd_row.columns)

    set_dashed_spines(ax2a)  # custom function to set spine formatting

    # CV PER ROW (to the right of the heatmap) #########################################################################
    ax2b = fig.add_subplot(gs[3, 5], sharey=ax1)
    ax2b.imshow(abs_df_cv_row, cmap="Purples", vmin=0, vmax=CV_CMAP_THRESHOLD, aspect="auto")
    ax2b.tick_params(axis="y", left=False, labelleft=False)
    ax2b.set_xticks(range(len(abs_df_cv_row.columns)))
    ax2b.set_xticklabels(abs_df_cv_row.columns)

    set_dashed_spines(ax2b)  # custom function to set spine formatting

    # AVERAGE PER COLUMN (above heatmap) ###############################################################################
    ax3 = fig.add_subplot(gs[2, 2], sharex=ax1)
    ax3.imshow(rel_df_mean_col, cmap=colormap, norm=rel_norm, aspect="auto")
    ax3.tick_params(axis="x", bottom=False, labelbottom=False)
    ax3.set_yticks(range(len(rel_df_mean_col.index)))
    ax3.set_yticklabels(rel_df_mean_col.index)

    # SD PER COLUMN (above heatmap) ####################################################################################
    ax3a = fig.add_subplot(gs[1, 2], sharex=ax1)
    ax3a.imshow(abs_df_sd_col, cmap="Greys", vmin=0, vmax=SD_CMAP_THRESHOLD, aspect="auto")
    ax3a.tick_params(axis="x", bottom=False, labelbottom=False)
    ax3a.set_yticks(range(len(abs_df_sd_col.index)))
    ax3a.set_yticklabels(abs_df_sd_col.index)

    set_dashed_spines(ax3a)  # custom function to set spine formatting

    # CV PER COLUMN (above heatmap) ####################################################################################
    ax3b = fig.add_subplot(gs[0, 2], sharex=ax1)
    ax3b.imshow(abs_df_cv_col, cmap="Purples", vmin=0, vmax=0.1, aspect="auto")
    ax3b.tick_params(axis="x", bottom=False, labelbottom=False)
    ax3b.set_yticks(range(len(abs_df_cv_col.index)))
    ax3b.set_yticklabels(abs_df_cv_col.index)

    set_dashed_spines(ax3b)  # custom function to set spine formatting

    # Use custom formatting function to add all texts to the heatmap according to style dictionary:
    if DEV_PERCENTAGE:  # display relative deviation, row and column means in percent
        set_dynamic_cell_text(ax1, rel_df.values * 100, style="style_hm_big", formatting=".1f")
        set_dynamic_cell_text(ax2, rel_df_mean_row.values * 100, style="style_hm_big", formatting=".1f")
        set_dynamic_cell_text(ax3, rel_df_mean_col.values * 100, style="style_hm_big", formatting=".1f")
    else:
        set_dynamic_cell_text(ax1, rel_df.values, style="style_hm_big", formatting=".3f")  # relative deviation
        set_dynamic_cell_text(ax2, rel_df_mean_row.values, style="style_hm_big", formatting=".3f")  # row means
        set_dynamic_cell_text(ax3, rel_df_mean_col.values, style="style_hm_big", formatting=".3f")  # col means

    if CV_PERCENTAGE:  # row and col CVs in %
        set_dynamic_cell_text(ax2b, abs_df_cv_row.values * 100, style="style_hm_single_small", formatting=".2f")
        set_dynamic_cell_text(ax3b, abs_df_cv_col.values * 100, style="style_hm_single_small", formatting=".2f")
    else:
        set_dynamic_cell_text(ax2b, abs_df_cv_row.values, style="style_hm_single_small", formatting=".3f")  # row CV
        set_dynamic_cell_text(ax3b, abs_df_cv_col.values, style="style_hm_single_small", formatting=".3f")  # col CV

    # Absolute values:
    set_dynamic_cell_text(ax1, abs_df.values, style="style_hm_small", formatting=".3f")
    set_dynamic_cell_text(ax2, abs_df_mean_row.values, style="style_hm_small", formatting=".3f")
    set_dynamic_cell_text(ax3, abs_df_mean_col.values, style="style_hm_small", formatting=".3f")

    set_dynamic_cell_text(ax2a, abs_df_sd_row.values, style="style_hm_single_small", formatting=".3f")
    set_dynamic_cell_text(ax3a, abs_df_sd_col.values, style="style_hm_single_small", formatting=".3f")

    # MAIN COLOR BAR ###################################################################################################
    ax4 = fig.add_subplot(gs[3, 0])
    if DEV_PERCENTAGE:
        cbar1 = fig.colorbar(ax1.images[0], ax=[ax1, ax2, ax3], cax=ax4, format=PercentFormatter(xmax=1))
        cbar1.set_label("relative absorbance deviation per well in %")
    else:
        cbar1 = fig.colorbar(ax1.images[0], ax=[ax1, ax2, ax3], cax=ax4)
        cbar1.set_label("relative absorbance deviation per well")

    cbar1.ax.tick_params(labelsize=7)
    cbar1.ax.yaxis.set_ticks_position("left")
    cbar1.ax.yaxis.set_label_position("left")

    # TEXT LEGEND ######################################################################################################
    spanning = gs.new_subplotspec((2, 3), colspan=3)  # text legend that spans multiple subplots
    span = fig.add_subplot(spanning)  # add the spanning subplot to the figure
    span.set_facecolor("0.95")  # add background color

    # Add the text to the spanning subplot:
    if DEV_PERCENTAGE:
        set_cell_text(span, "Deviation from 0 in %", style="style_hm_big", pos=(0.5, 0.5))
    else:
        set_cell_text(span, "Deviation from 0", style="style_hm_big", pos=(0.5, 0.5))

    set_cell_text(span, "Absolute absorbance value", style="style_hm_small", pos=(0.5, 0.5))

    # Hide axis ticks while keeping the background color and border:
    span.set_xticks([])
    span.set_yticks([])

    # SECOND TEXT LEGEND ###############################################################################################
    spanning1 = gs.new_subplotspec((0, 3), colspan=3)  # second text legend for the top right corner

    span1 = fig.add_subplot(spanning1)  # add the spanning subplot to the figure
    span1.set_facecolor("0.95")  # add background color

    # Add the text to the spanning subplot:
    if CV_PERCENTAGE:
        set_cell_text(span1, "CV = SD/mean*100; using\nabsolute absorbance values",
                      style="style_hm_single_small",
                      pos=(0.5, 0.5))
    else:
        set_cell_text(span1, "CV = SD/mean; using\nabsolute absorbance values",
                      style="style_hm_single_small",
                      pos=(0.5, 0.5))

    # Hide axis ticks while keeping the background color and border:
    span1.set_xticks([])
    span1.set_yticks([])
    set_dashed_spines(span1)  # custom function to set spine formatting

    # THIRD TEXT LEGEND ################################################################################################
    spanning2 = gs.new_subplotspec((1, 3), colspan=3)  # second text legend for the top right corner

    span2 = fig.add_subplot(spanning2)  # add the spanning subplot to the figure
    span2.set_facecolor("0.95")  # add background color

    # Add the text to the spanning subplot:
    set_cell_text(span2, "Standard deviation using\nabsolute absorbance values",
                  style="style_hm_single_small",
                  pos=(0.5, 0.5))

    # Hide axis ticks while keeping the background color and border:
    span2.set_xticks([])
    span2.set_yticks([])
    set_dashed_spines(span2)  # custom function to set spine formatting

    # SECOND COLOR BAR #################################################################################################
    ax5 = fig.add_subplot(gs[0, 0])
    if CV_PERCENTAGE:
        cbar2 = fig.colorbar(ax2b.images[0], ax=[ax2b, ax3b], cax=ax5, label="%CV", format=PercentFormatter(xmax=1))
    else:
        cbar2 = fig.colorbar(ax2b.images[0], ax=[ax2b, ax3b], cax=ax5, label="CV")

    cbar2.ax.tick_params(labelsize=7)
    cbar2.ax.yaxis.label.set_color("dimgray")
    cbar2.ax.yaxis.set_ticks_position("left")
    cbar2.ax.yaxis.set_label_position("left")
    set_dashed_spines(ax5)

    # THIRD COLOR BAR ##################################################################################################
    ax6 = fig.add_subplot(gs[1, 0])
    cbar3 = fig.colorbar(ax2a.images[0], ax=[ax2a], cax=ax6, label="SD")
    cbar3.ax.tick_params(labelsize=7)
    cbar3.ax.yaxis.label.set_color("dimgray")
    cbar3.ax.yaxis.set_ticks_position("left")
    cbar3.ax.yaxis.set_label_position("left")
    set_dashed_spines(ax6)

    return fig, fig_width, rel_df_mean_row, abs_df_cv_row


def generate_quick_plate_histogram(abs_df: pd.DataFrame, fig_width: float, scaling: float = PLOT_SCALING) -> plt.Figure:
    """
    Generates a histogram of all absorbance values in the microplate.

    :param abs_df: Data frame of absolute absorbance values in microplate format.
    :param fig_width: Width of the heatmap figure for reference.
    :param scaling: Scaling factor for the plot size, defaults to global variable.
    :return: A histogram of all absorbance values in the microplate.
    """
    data = abs_df.values.flatten()

    mean = np.mean(data)
    median = np.median(data)
    min_val = np.min(data)
    max_val = np.max(data)
    sd = np.std(data, ddof=1)
    cv = sd / mean

    num_bins = int(np.sqrt(len(data)))  # number of bins for the histogram
    xticks = [min_val, mean-sd, mean, mean+sd, max_val]  # xticks for the histogram

    x = np.linspace(min_val, max_val, 200)  # x values for the normal distribution
    p = norm.pdf(x, mean, sd)  # y values for the normal distribution

    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0)

    fig = plt.figure(figsize=(fig_width / scaling, fig_width / (scaling*1.5)), dpi=300)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
    fig.suptitle("Distribution of absorbance measurements (whole plate)")

    # Add histogram subplot:
    ax1 = fig.add_subplot(gs[0])
    ax1.set_ylabel("Number of wells")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(["MIN", "SD", "MEAN", "SD", "MAX"])
    ax1.tick_params(axis="x", direction="in", bottom=False, top=False, labelbottom=False, labeltop=True)

    # Plot histogram:
    _, _, handle_his = ax1.hist(data, bins=num_bins, edgecolor="gray", color="lightgray", alpha=0.8,
                                label="Distribution of absorbance values")

    # Plot normal distribution:
    handle_pdf, = ax1.plot(x, p, color="black", linestyle="dashed", linewidth=1.5,
                           label="Normal distribution for comparison")

    # Plot lines for statistics:
    handle_avg = ax1.axvline(mean, color="red", linestyle="solid", linewidth=1.5, label=f"Mean:   {mean:.4f}")
    handle_min = ax1.axvline(min_val, color="orchid", linestyle="solid", linewidth=1.5, label=f"Min:    {min_val:.4f}")
    handle_max = ax1.axvline(max_val, color="orchid", linestyle="solid", linewidth=1.5, label=f"Max:    {max_val:.4f}")
    handle_std = ax1.axvline(mean+sd, color="orange", linestyle="solid", linewidth=1.5, label=f"±SD:    {sd:.4f}")
    ax1.axvline(mean-sd, color="orange", linestyle="solid", linewidth=1.5)

    handle_med = ax1.axvline(median, color="red", linestyle="dashed", label=f"Median: {median:.4f}")
    if CV_PERCENTAGE:
        handle_cv = ax1.axvline(mean, color="none", label=f"%CV:    {cv*100:.2f}")
    else:
        handle_cv = ax1.axvline(mean, color="none", label=f"CV:     {cv:.4f}")

    # Add legend with specific order and special formatting:
    handles = [handle_his[0], handle_pdf, handle_avg, handle_std, handle_cv, handle_min, handle_med, handle_max]
    legend = ax1.legend(handles=handles, loc="upper right", fontsize="small")
    for i in [2, 3, 4, 5, 6, 7]:
        legend.get_texts()[i].set_family("monospace")

    # Add box plot subplot:
    ax2 = fig.add_subplot(gs[1])
    medianprops = dict(linewidth=1.5, color="red", linestyle="dashed")
    capprops = dict(linewidth=1.5, color="orchid", linestyle="solid")
    ax2.boxplot(data, vert=False, whis=np.inf, widths=0.5, medianprops=medianprops, capprops=capprops)
    ax2.scatter(data, np.ones(shape=data.shape), color="gray", alpha=0.25)
    ax2.set_xlabel("Absorbance")
    ax2.set_yticks([])

    return fig


def generate_quick_summary(rel_df: np.ndarray) -> str:
    """
    Generates a figure that shows the outcome (pass or fail) of the quick run.

    :param rel_df: Data frame of relative absorbance values in microplate format.
    :return: A figure that shows the outcome of the quick run.
    """
    questionable_devs = np.sum(np.abs(rel_df) >= DEV_THRESHOLD_UPPER)
    failed_devs = np.sum(np.abs(rel_df) >= DEV_THRESHOLD)
    if questionable_devs > 0:
        outcome = "Some channels pipette the wrong volume."
        facecolor = "red"
        edgecolor = "darkred"
        explanation = f"{questionable_devs} channels have an average deviation from the median of ±10% or more.<br>" \
                      f"{failed_devs} channels have an average deviation from the median of ±5% or more."
    elif failed_devs > 0:
        outcome = "All channels pipette relatively similar volumes."
        facecolor = "goldenrod"
        edgecolor = "darkgoldenrod"
        explanation = f"All channels have an average deviation from the median of less than ±10%.<br>" \
                      f"{failed_devs} channels have an average deviation from the median of ±5% or more."
    else:
        outcome = "All channels pipette very similar volumes."
        facecolor = "green"
        edgecolor = "darkgreen"
        explanation = "All channels have an average deviation from the median of less than ±5%."

    outcome_devs = generate_summary_box(facecolor, edgecolor, outcome, explanation)

    return outcome_devs


def generate_eight_channel_overview(abs_df, fig_width: float, scaling: float = PLOT_SCALING) -> plt.Figure:
    """
    Creates a per-channel plot of the pipetting performance over the replicates.

    :param abs_df: Data frame of absolute absorbance values in microplate format.
    :param fig_width: Width of the heatmap figure for reference.
    :param scaling: Scaling factor for the plot size, defaults to global variable.
    :return: Plot of the average pipetting performance per channel.
    """
    fig, ax = plt.subplots(figsize=(fig_width / scaling, fig_width / (scaling*1.5)))
    ax.axhline(y=np.median(abs_df.values.flatten()), color="grey", linestyle="--",
               label="overall median plate absorbance")  # line for predicted volume
    ax.boxplot(abs_df.transpose().values, showfliers=False)  # boxplots for each channel
    ax.set_title("Average pipetting performance per channel")
    ax.set_xlabel("Channels")
    ax.set_xticks(range(1, len(abs_df.index) + 1))
    ax.set_xticklabels(abs_df.index)
    ax.set_ylabel(r"Absorbance per channel")
    for i in range(abs_df.shape[0]):
        y = abs_df.iloc[i, :]
        x = np.random.normal(i + 1, 0.1, size=len(y))
        ax.scatter(x, y, s=20, color="none", edgecolors="k", alpha=0.33)
    ax.legend()

    return fig


def generate_full_plate_heatmap(all_abs: list, all_rel: list, scaling: float = PLOT_SCALING) -> tuple:
    """
    Generates a heatmap of relative well absorbance values from a microplate absorbance readout.
    Works independently of the number of wells and combines replicates.

    :param all_abs: List of data frames of absolute absorbance values in microplate format.
    :param all_rel: List of data frames of relative absorbance values in microplate format.
    :param scaling: Scaling factor for the plot size, defaults to global variable.
    :return: A heatmap figure of well absorbance values and a figure width to base other plots on.
    """
    # Calculate statistics across absolute data frames using numpy arrays:
    if compare_dataframe_shapes(all_abs):
        abs_df = pd.DataFrame(index=all_abs[0].index, columns=all_abs[0].columns)  # save columns and index
        array_abs_3d = np.stack([df.values for df in all_abs], axis=2)  # all plate values stacked on top of each other
        array_abs_means = array_abs_3d.mean(axis=2)
    else:
        raise ValueError("The plates do not have the same dimensions!")

    # Do the same for relative data frames:
    if compare_dataframe_shapes(all_rel):
        # rel_df = pd.DataFrame(index=all_rel[0].index, columns=all_rel[0].columns)  # save columns and index
        array_rel_3d = np.stack([df.values for df in all_rel], axis=2)  # all plate values stacked on top of each other
        array_rel_means = array_rel_3d.mean(axis=2)
    else:
        raise ValueError("The plates do not have the same dimensions!")

    # GRIDSPEC CREATION ################################################################################################
    cbar_width = 0.5  # width of the color bar, adjustable, given in inches

    ax_widths = [abs_df.shape[1], 0, cbar_width]  # list for width_ratios and mean_ax_width
    ax_heights = [1, abs_df.shape[0]]  # list for height_ratios and mean_ax_height

    # Use custom function to ensure correct aspect ratio and spacing between subplots:
    gs, fig_width, fig_height = set_heatmap_layout(ax_widths, ax_heights)

    """
    The following diagram shows the layout of the subplots and the grid.
    The grid is used to place the subplots.

        [0|1]                      [2]
        ┌─┬─┬───────────────────────────────────────────────┐
    [0] │ │ │                  TEXT LEGEND                  │
        ├─┼─┼───────────────────────────────────────────────┤
        │ │ │                                               │
        │C│ │                                               │
        │O│ │                                               │
        │L│ │                                               │
        │O│ │                                               │
    [1] │R│ │                 PLATE HEATMAP                 │
        │ │ │                     [1|2]                     │
        │B│ │                                               │
        │A│ │                                               │
        │R│ │                                               │
        │ │ │                                               │
        └─┴─┴───────────────────────────────────────────────┘
    """

    # Create the figure with correct aspect ratio and scaling, also scale down subplots to 75 percent and add title:
    fig = plt.figure(figsize=(fig_width / scaling, fig_height / scaling), dpi=300)
    fig.subplots_adjust(left=0.125, right=0.875, top=0.95, bottom=0.2)

    # PLATE HEATMAP (bottom left) ######################################################################################
    colormap = "RdBu_r"  # colormap for the heatmap
    rel_norm = TwoSlopeNorm(vmin=-DEV_CMAP_THRESHOLD, vcenter=0, vmax=+DEV_CMAP_THRESHOLD)  # heatmap normalization

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(array_rel_means, cmap=colormap, norm=rel_norm, aspect="auto")
    set_all_heatmap_ticks(ax1, abs_df)  # custom function to set ticks for heatmaps

    # Use custom formatting function to add text to the heatmap according to style dictionary:
    if DEV_PERCENTAGE:
        set_dynamic_cell_text(ax1, array_rel_means * 100,
                              style="style_hm_big",
                              formatting=".1f",
                              threshold=DEV_THRESHOLD * 100)
    else:
        set_dynamic_cell_text(ax1, array_rel_means,
                              style="style_hm_big",
                              formatting=".1f",
                              threshold=DEV_THRESHOLD)

    set_dynamic_cell_text(ax1, array_abs_means, style="style_hm_small", formatting=".3f")

    # MAIN COLOR BAR ###################################################################################################
    ax4 = fig.add_subplot(gs[1, 2])
    if DEV_PERCENTAGE:
        cbar1 = fig.colorbar(ax1.images[0], ax=[ax1], cax=ax4,
                             label="relative absorbance deviation per well in %", format=PercentFormatter(xmax=1))
    else:
        cbar1 = fig.colorbar(ax1.images[0], ax=[ax1], cax=ax4,
                             label="relative absorbance deviation per well")

    cbar1.ax.tick_params(labelsize=7)
    cbar1.ax.yaxis.set_ticks_position("right")
    cbar1.ax.yaxis.set_label_position("right")

    # TEXT LEGEND ######################################################################################################
    spanning1 = gs.new_subplotspec((0, 0), colspan=3)  # text legend that spans multiple subplots
    span1 = fig.add_subplot(spanning1)  # add the spanning subplot to the figure
    span1.set_facecolor("0.95")  # add background color

    # Add the text to the spanning subplot using custom function:
    if DEV_PERCENTAGE:
        set_cell_text(span1, "TOP: Relative deviation from the median plate absorbance in % averaged over all replicates.",
                      style="style_hm_big", pos=(0.5, 0.5))
    else:
        set_cell_text(span1, "TOP: Relative deviation from the median plate absorbance averaged over all replicates.",
                      style="style_hm_big", pos=(0.5, 0.5))

    set_cell_text(span1, "BOTTOM: Absolute mean absorbance value over all replicates.", style="style_hm_small", pos=(0.5, 0.5))

    # Hide axis ticks while keeping the background color and border:
    span1.set_xticks([])
    span1.set_yticks([])

    return fig, fig_width, array_rel_means


def generate_full_cv_heatmap(all_abs: list, scaling: float = PLOT_SCALING) -> tuple:
    """
    Generates a heatmap of coefficient of variation values from a microplate absorbance readout.
    Works independently of the number of wells.

    :param all_abs: List of data frames of absolute absorbance values in microplate format.
    :param scaling: Scaling factor for the plot size to increase text size, defaults to global variable.
    :return: A dataframe of well volumes in plate and long format, a heatmap of well volumes.
    """
    # Calculate statistics across absolute data frames using numpy arrays:
    if compare_dataframe_shapes(all_abs):
        abs_df = pd.DataFrame(index=all_abs[0].index, columns=all_abs[0].columns)  # save columns and index
        array_abs_3d = np.stack([df.values for df in all_abs], axis=2)  # all plate values stacked on top of each other
        array_abs_means = array_abs_3d.mean(axis=2)
        array_abs_stds = array_abs_3d.std(axis=2, ddof=1)
        array_abs_cvs = array_abs_stds / array_abs_means
    else:
        raise ValueError("The plates do not have the same dimensions!")

    # GRIDSPEC CREATION ################################################################################################
    cbar_width = 0.5  # width of the color bar, adjustable, given in inches

    ax_widths = [abs_df.shape[1], 0, cbar_width]  # list for width_ratios and mean_ax_width
    ax_heights = [1, abs_df.shape[0]]  # list for height_ratios and mean_ax_height

    # Use custom function to ensure correct aspect ratio and spacing between subplots:
    gs, fig_width, fig_height = set_heatmap_layout(ax_widths, ax_heights)

    """
    The following diagram shows the layout of the subplots and the grid.
    The grid is used to place the subplots.

        [0|1]                      [2]
        ┌─┬─┬───────────────────────────────────────────────┐
    [0] │ │ │                  TEXT LEGEND                  │
        ├─┼─┼───────────────────────────────────────────────┤
        │ │ │                                               │
        │C│ │                                               │
        │O│ │                                               │
        │L│ │                                               │
        │O│ │                                               │
    [1] │R│ │                 PLATE HEATMAP                 │
        │ │ │                     [1|2]                     │
        │B│ │                                               │
        │A│ │                                               │
        │R│ │                                               │
        │ │ │                                               │
        └─┴─┴───────────────────────────────────────────────┘
    """

    # Create the figure with correct aspect ratio and scaling, also scale down subplots to 75 percent and add title:
    fig = plt.figure(figsize=(fig_width / scaling, fig_height / scaling), dpi=300)
    fig.subplots_adjust(left=0.125, right=0.875, top=0.95, bottom=0.2)

    # PLATE HEATMAP (bottom left) ######################################################################################
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(array_abs_cvs, cmap="Purples", vmin=0, vmax=CV_CMAP_THRESHOLD, aspect="auto")  # in percent
    set_all_heatmap_ticks(ax1, abs_df)  # custom function to set ticks for heatmaps

    # Use custom formatting function to add text to the heatmap according to style dictionary:
    if CV_PERCENTAGE:
        set_dynamic_cell_text(ax1, array_abs_cvs * 100,
                              style="style_hm_single_big",
                              threshold=CV_THRESHOLD * 100,
                              formatting=".2f")
    else:
        set_dynamic_cell_text(ax1, array_abs_cvs,
                              style="style_hm_single_big",
                              threshold=CV_THRESHOLD,
                              formatting=".3f")

    # MAIN COLOR BAR ###################################################################################################
    ax4 = fig.add_subplot(gs[1, 2])

    if CV_PERCENTAGE:
        cbar1 = fig.colorbar(ax1.images[0], ax=[ax1], cax=ax4,
                             label="coefficient of variation (CV) in %", format=PercentFormatter(xmax=1))
    else:
        cbar1 = fig.colorbar(ax1.images[0], ax=[ax1], cax=ax4,
                             label="coefficient of variation (CV)")

    cbar1.ax.tick_params(labelsize=7)
    cbar1.ax.yaxis.set_ticks_position("right")
    cbar1.ax.yaxis.set_label_position("right")

    # TEXT LEGEND ######################################################################################################
    spanning1 = gs.new_subplotspec((0, 0), colspan=3)  # text legend that spans multiple subplots
    span1 = fig.add_subplot(spanning1)  # add the spanning subplot to the figure
    span1.set_facecolor("0.95")  # add background color

    # Add the text to the spanning subplot:
    if CV_PERCENTAGE:
        set_cell_text(span1, "Coefficient of variation (CV = SD/mean) per well in % using the replicates of each well.",
                      style="style_hm_single_big", pos=(0.5, 0.5))
    else:
        set_cell_text(span1, "Coefficient of variation (CV = SD/mean) per well using the replicates of each well.",
                      style="style_hm_single_big", pos=(0.5, 0.5))

    # Hide axis ticks while keeping the background color and border:
    span1.set_xticks([])
    span1.set_yticks([])

    return fig, fig_width, array_abs_cvs


def generate_full_summary(rel_means: np.ndarray, abs_cvs: np.ndarray) -> list:
    """
    Generates a summary plot of the full run that gives at-a-glance information about the results.
    Includes a "passed" variant and two stages of "failed" variants (depending on severity).
    Uses absorbance deviations and CVs over replicates to decide if performance criteria are met.

    :param rel_means: DataFrame of mean deviations from the median absorbance per channel.
    :param abs_cvs: DataFrame of CV values in microplate format per channel using replicates.
    :return: A figure that shows the outcome of the full run.
    """
    questionable_devs = np.sum(np.abs(rel_means) >= DEV_THRESHOLD_UPPER)
    failed_devs = np.sum(np.abs(rel_means) >= DEV_THRESHOLD)
    if questionable_devs > 0:
        outcome = "Some channels pipette the wrong volume."
        facecolor = "red"
        edgecolor = "darkred"
        explanation = f"{questionable_devs} channels have an average deviation from the median of ±10% or more.<br>" \
                      f"{failed_devs} channels have an average deviation from the median of ±5% or more."
    elif failed_devs > 0:
        outcome = "All channels pipette relatively similar volumes."
        facecolor = "goldenrod"
        edgecolor = "darkgoldenrod"
        explanation = f"All channels have an average deviation from the median of less than ±10%.<br>" \
                      f"{failed_devs} channels have an average deviation from the median of ±5% or more."
    else:
        outcome = "All channels pipette very similar volumes."
        facecolor = "green"
        edgecolor = "darkgreen"
        explanation = "All channels have an average deviation from the median of less than ±5%."

    outcome_devs = generate_summary_box(facecolor, edgecolor, outcome, explanation)

    questionable_cvs = np.sum(abs_cvs >= CV_THRESHOLD_UPPER)
    failed_cvs = np.sum(abs_cvs >= CV_THRESHOLD)
    if questionable_cvs > 0:
        outcome = "Some channels vary too much between replicates."
        facecolor = "red"
        edgecolor = "darkred"
        explanation = f"{questionable_cvs} channels have a CV of 10% or more.<br>" \
                      f"{failed_cvs} channels have a CV of 5% or more."
    elif failed_cvs > 0:
        outcome = "All channels have relatively low variation between replicates."
        facecolor = "goldenrod"
        edgecolor = "darkgoldenrod"
        explanation = f"All channels have a CV of less than 10%.<br>" \
                      f"{failed_cvs} channels have a CV of 5% or more."
    else:
        outcome = "All channels have very low variation between replicates."
        facecolor = "green"
        edgecolor = "darkgreen"
        explanation = "All channels have a CV of less than 5%."

    outcome_cvs = generate_summary_box(facecolor, edgecolor, outcome, explanation)

    return [outcome_devs, outcome_cvs]


def convert_plot_to_svg(fig: plt.Figure) -> str:
    """
    Shortcut to create an SVG from a plot which is then embedded into an HTML.

    :param fig: Matplotlib figure.
    :return: SVG image in form of a string.
    """
    f = BytesIO()
    canvas = FigureCanvasSVG(fig)
    canvas.print_svg(f)
    svg = f.getvalue().decode("utf-8")

    return svg


def generate_summary_box(background_color: str, border_color: str, big_text: str, small_text: str) -> str:
    """
    Can be used to draw a box around predefined text. Used in the HTML report to convey different degrees of severity
    when assessing pipetting performance.

    :param background_color: Color of the box drawn.
    :param border_color: Color of the outline of the box.
    :param big_text: Main message displayed.
    :param small_text: Further information displayed below the main message.
    :return: An HTML div containing the text box.
    """
    return f"""
        <div class="status-box" style="background-color: {background_color}; border-color: {border_color};">
            <div class="status-box-text">{big_text}</div>
            <p>{small_text}</p>
        </div>
        """


def main(processed_files: list, run_mode: str, variables: dict):
    """
    Function that is run when this script is called. Uses variables from the Camunda workflow.
    Here, also a list of HTML elements is created that is extended with the relevant information of each workflow mode.
    The list is joined in the end to create a full HTMl string that is written and saved to a file.

    :param processed_files: List of files that were created in the previous transformation.
    :param run_mode: Variant of the workflow that is executed.
    :param variables: Dictionary of entered variables.
    :return: An HTML report saved as a file.
    """
    # Create lists to save all DataFrames:
    all_abs = []
    all_rel = []
    long_abs = []
    long_rel = []

    all_rel_row_means = []
    all_abs_row_cvs = []

    # Create lists to save all plots:
    all_histograms = []
    all_heatmaps = []
    all_eight = []  # for 8-channel overview plot(s)

    # Loop through processed files to create and save everything that is plate-specific:
    for file in processed_files:
        base_name = os.path.splitext(os.path.basename(file))[0].replace("processed_", "", 1)

        abs_df = pd.read_csv(file, index_col=0)  # read as DataFrame
        rel_df = calculate_relative_absorbance(abs_df, use_median=True)  # get DataFrame with relative values
        abs_df_long = convert_plate_to_long(abs_df)  # get long format DataFrame
        rel_df_long = convert_plate_to_long(rel_df)  # same for relative DataFrame

        all_abs.append(abs_df)
        all_rel.append(rel_df)
        long_abs.append(abs_df_long)
        long_rel.append(rel_df_long)

        # Save the DataFrames with distinct names:
        rel_df.to_csv(f"data/processed/relative_absorbance_{base_name}.csv")
        rel_df_long.to_csv(f"data/processed/relative_absorbance_long_{base_name}.csv", index=False)
        abs_df.to_csv(f"data/processed/absolute_absorbance_{base_name}.csv")
        abs_df_long.to_csv(f"data/processed/absolute_absorbance_long_{base_name}.csv", index=False)
    
        # Generate plate-specific plots:
        fig_heatmap, fig_width, rel_df_mean_row, abs_df_cv_row = generate_quick_plate_heatmap(abs_df, rel_df)
        fig_hist = generate_quick_plate_histogram(abs_df, fig_width)
        fig_eight = generate_eight_channel_overview(abs_df, fig_width)

        # Save row summaries for 8-channel summary figure:
        all_rel_row_means.append(rel_df_mean_row)
        all_abs_row_cvs.append(abs_df_cv_row)

        # Save these plots in separate plot lists:
        all_heatmaps.append(fig_heatmap)
        all_histograms.append(fig_hist)
        all_eight.append(fig_eight)

    # To store all HTML elements:
    html_list = ["""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pipetting Performance Report</title>
        <style>
        body {
            font-family: 'Roche Sans', sans-serif;
            background-color: #bde3ffff; /* Roche light blue background */
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
        }
        .container {
            background-color: #ffffff; /* White foreground */
            width: 210mm; /* A4 width */
            padding: 20px;
        }
        h1, h2, h3 {
            text-align: center;
        }
        h1 {
            background-color: #f0f0f0; /* Light gray highlight */
            padding: 10px; /* Adjust padding as needed */
        }
        h2 {
            background-color: #f0f0f0; /* Light gray highlight */
            padding: 10px; /* Adjust padding as needed */
        }
        table {
            border-collapse: collapse;
            width: 100%;
            font-size: 0.9em;
        }
        th {
            background-color: #f0f0f0; /* Light gray background */
            text-align: center;
        }
        th, td {
            border: 1px solid #000000;
            padding: 4px;
        }
        .svg-container {
            text-align: center; /* Center the SVGs */
        }
        .svg-container svg {
            width: 100%; /* Make the SVGs fit the width of the page */
            height: auto; /* Maintain the aspect ratio of the SVGs */
        }
        .status-box {
            box-sizing: border-box;
            width: 100%;
            color: white; /* Text color */
            padding: 0px; /* Space inside the box */
            text-align: center; /* Center the text */
            border: 3px solid darkorange; /* Border color and width */
            background-color: orange; /* Background color */
            margin-top: 10px; /* Add spacing above the box */
            margin-bottom: 10px; /* Add spacing below the box */
        }
        .status-box-text {
            font-size: 1.5em; /* Adjust as needed */
            font-weight: normal; /* Makes the text not bold */
            color: white; /* Text color */
            padding-top: 20px; /* Add padding above the big text */
        }
        .status-box p {
            font-size: 1em; /* Adjust as needed */
            padding-bottom: 10px; /* Reduce padding below the small text */
        }
        </style>
    </head>
    <body>
    <div class="container">
    """]

    if run_mode == "full_run":
        report_title = "Full Run"
    elif run_mode == "quick_run":
        report_title = "Quick Run"
    elif run_mode == "eight_run":
        report_title = "8-Channel Run"
    else:
        report_title = ""

    # Add title:
    html_list.append(f"<h1>Pipetting Performance Report ({report_title})</h1>")

    # Add assay table:
    variables_df = pd.DataFrame(list(variables.items()), columns=["Variable", "Value"])
    html_list.append(variables_df.to_html(index=False, classes="my-table", escape=False))

    # Add overview images:
    if run_mode == "full_run":
        fig_overall, fig_width, array_rel_means = generate_full_plate_heatmap(all_abs, all_rel)
        fig_cvs, fig_width, array_abs_cvs = generate_full_cv_heatmap(all_abs)

        # Overview boxes:
        html_list.extend(generate_full_summary(array_rel_means, array_abs_cvs))

        # Deviation overview plot:
        html_list.append("<h2>How different are the channels from each other?</h2>")
        svg = convert_plot_to_svg(fig_overall)
        html_list.append('<div class="svg-container">' + svg + '</div>')

        # CV overview plot:
        html_list.append("<h2>Are the values for each channel consistent?</h2>")
        svg = convert_plot_to_svg(fig_cvs)
        html_list.append('<div class="svg-container">' + svg + '</div>')

        # Individual plate summaries:
        html_list.append("<h2>Plate summaries</h2>")
        for i, _ in enumerate(processed_files):
            plots = [all_heatmaps[i], all_histograms[i]]
            html_list.append(f"<h3>PLATE {i+1}: {processed_files[i]}</h3>")

            for fig in plots:
                svg = convert_plot_to_svg(fig)
                html_list.append('<div class="svg-container">' + svg + '</div>')

    elif run_mode == "eight_run":
        # Overview boxes:
        html_list.extend(generate_full_summary(all_rel_row_means[0].values, all_abs_row_cvs[0].values))

        # 8-channel overview plot:
        html_list.append("<h2>How different are the channels from each other?</h2>")
        svg = convert_plot_to_svg(all_eight[0])
        html_list.append('<div class="svg-container">' + svg + '</div>')

        # Deviation heatmap plot:
        html_list.append("<h2>Plate summary</h2>")
        svg = convert_plot_to_svg(all_heatmaps[0])
        html_list.append('<div class="svg-container">' + svg + '</div>')

        # Whole-plate histogram:
        svg = convert_plot_to_svg(all_histograms[0])
        html_list.append('<div class="svg-container">' + svg + '</div>')

    elif run_mode == "quick_run":
        # Overview box:
        html_list.append(generate_quick_summary(all_rel[0].values))

        # Deviation heatmap plot:
        html_list.append("<h2>Plate summary</h2>")
        svg = convert_plot_to_svg(all_heatmaps[0])
        html_list.append('<div class="svg-container">' + svg + '</div>')

        # Whole-plate histogram:
        svg = convert_plot_to_svg(all_histograms[0])
        html_list.append('<div class="svg-container">' + svg + '</div>')

    # And finally, close the HTML tags:
    html_list.append("""
    </div>
    </body>
    </html>
    """)

    # Combine strings to form the complete HTML document:
    html_doc = "".join(html_list)

    html_report = f"data/end_report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"
    with open(html_report, "w") as f:
        f.write(html_doc)

    return html_report
