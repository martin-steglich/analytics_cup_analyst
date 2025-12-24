from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from .preprocessing import (PITCH_WIDTH,PITCH_LENGTH,PITCH_MIN_X, PITCH_MAX_X,PITCH_MIN_Y,PITCH_MAX_Y,ZONE_X,ZONE_Y)
from . import preprocessing as pre
from .metrics import METRIC_UNITS
from matplotlib import font_manager, rcParams
from pathlib import Path



PITCH_LENGTH = 105
PITCH_HALF_LENGTH = 52.5
PITCH_WIDTH = 68
BACKGROUND_COLOR = '#F5F5F5'
PRIMARY_TEXT_COLOR = '#000000'
SECONDARY_TEXT_COLOR = '#757575'
PRIMARY_COLOR = '#649CCB'
HIGHLIGHT_COLOR = "#203490"

CMAP = LinearSegmentedColormap.from_list("Pearl Earring - 100 colors",
                                                           [BACKGROUND_COLOR, "#203490"], N=200)


def setup_fonts():
    """
    Register and activate the project font family for Matplotlib plots.

    This function loads Manrope (Regular and Bold) from the local assets folder
    and sets Matplotlib's default font family accordingly.

    Raises
    ------
    FileNotFoundError
        If any of the required font files cannot be found.
    """
    here = Path(__file__).resolve().parent
    root = here.parent                               
    fonts_dir = here / "assets" / "fonts" 

    regular = fonts_dir / "Manrope-Regular.ttf"
    bold = fonts_dir / "Manrope-Bold.ttf"

    
    if not regular.exists():
        raise FileNotFoundError(f"Font not found: {regular}")
    if not bold.exists():
        raise FileNotFoundError(f"Font not found: {bold}")

    font_manager.fontManager.addfont(str(regular))
    font_manager.fontManager.addfont(str(bold))
    rcParams["font.family"] = "Manrope"

setup_fonts()

def height_from_own_goal(x: float) -> float:
    """
    Convert a SkillCorner x-coordinate to distance from the team's own goal.

    SkillCorner coordinates use x in [-52.5, 52.5] for a full 105m pitch.
    This helper maps x to [0, 105] by shifting the origin.

    Parameters
    ----------
    x : float
        SkillCorner x-coordinate in meters (range approximately [-52.5, 52.5]).

    Returns
    -------
    float
        Distance from own goal line in meters (range approximately [0, 105]).
    """
    return float(x + PITCH_HALF_LENGTH)

def get_pitch(ax:Axes, vertical: bool = True, draw: bool = True) -> tuple:
    """
    Create a mplsoccer pitch (vertical or horizontal) and optionally draw it.

    Adds an "Attack" direction arrow annotation to the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the pitch will be created/drawn.
    vertical : bool, default=True
        If True, uses a VerticalPitch. If False, uses a horizontal Pitch.
    draw : bool, default=True
        If True, draws the pitch immediately on `ax`.

    Returns
    -------
    tuple
        (pitch, ax) where pitch is an mplsoccer Pitch/VerticalPitch instance
        and ax is the same matplotlib Axes passed in.
    """
    if vertical:
        pitch = VerticalPitch(pitch_type='skillcorner',
                    line_zorder=10,
                    line_color=SECONDARY_TEXT_COLOR,
                    half=False,
                    pitch_color=BACKGROUND_COLOR,
                    linewidth=.75,
                    corner_arcs=True,
                    goal_type='box',
                    pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                    axis=False, label=False)
        arrow_xytext=(36, -8)
        arrow_xy=(36, 8)
        arrow_text_xy=(37.2, 0)
        rotation='vertical'
    else:
        pitch = Pitch(pitch_type='skillcorner',
                    line_zorder=10,
                    line_color=SECONDARY_TEXT_COLOR,
                    half=False,
                    pitch_color=BACKGROUND_COLOR,
                    linewidth=.75,
                    corner_arcs=True,
                    goal_type='box',
                    pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                    axis=False, label=False)
        arrow_xytext=(-8, -36)
        arrow_xy=(8, -36)
        arrow_text_xy=(0,-37.2)
        rotation='horizontal'

    if draw:
        pitch.draw(ax=ax)

    ax.annotate("", xytext=arrow_xytext, xy=arrow_xy,
                        arrowprops=dict(arrowstyle="->", color=SECONDARY_TEXT_COLOR, lw=.75), color=SECONDARY_TEXT_COLOR)
    ax.annotate("Attack", xy=arrow_text_xy, fontsize=8, ha='center', va='center', color=SECONDARY_TEXT_COLOR,rotation=rotation)
    return pitch, ax

def plot_line_by_state(ax, t, y, state, *, lw=2,
                       color_in=None, color_out="0.6"):
    """
    Plot a time series with line color changing according to a boolean state.

    The series is segmented whenever `state` changes, and each segment is
    plotted in a different color (e.g., in-possession vs out-of-possession).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to plot.
    t : array-like
        Time values (x-axis).
    y : array-like
        Metric values (y-axis).
    state : array-like of bool
        Boolean state for each time step. Consecutive values form segments.
    lw : float, default=2
        Line width.
    color_in : any, optional
        Color used when state is True.
    color_out : any, default="0.6"
        Color used when state is False.

    Returns
    -------
    None
    """
    t = np.asarray(t)
    y = np.asarray(y)
    state = np.asarray(state).astype(bool)

    start = 0
    for i in range(1, len(state)):
        if state[i] != state[i-1]:
            c = color_in if state[i-1] else color_out
            ax.plot(t[start:i], y[start:i], color=c, linewidth=lw)
            start = i

    c = color_in if state[-1] else color_out
    ax.plot(t[start:], y[start:], color=c, linewidth=lw)

def plot_timeline_by_metric_half(timeline_df, metric_col, title=None):
    """
    Plot a metric timeline split by halves (two subplots).

    The input timeline is expected to contain `period_id` and `time_bin`.
    Time is displayed in minutes relative to the start of each half.

    If `in_possession_majority` is present, the line is colored by that state.

    Parameters
    ----------
    timeline_df : pandas.DataFrame
        Timeline DataFrame (typically output of build_timeline/build_half_timelines).
        Must include: time_bin, period_id, and the selected metric column.
        Optionally includes: in_possession_majority.
    metric_col : str
        Column name of the metric to plot.
    title : str, optional
        Title to use for the figure. If None, a default title is used.

    Returns
    -------
    None
        Displays the plot using matplotlib.
    """
    df = timeline_df.copy().sort_values("time_bin")

    first_half  = df[df["period_id"] == 1].copy()
    second_half = df[df["period_id"] == 2].copy()
    
    first_half["t_min"]  = (first_half["time_bin"]  - first_half["time_bin"].min())  / 60
    second_half["t_min"] = (second_half["time_bin"] - second_half["time_bin"].min()) / 60
    # Primer tiempo en segundos:
    t1_start = first_half["time_bin"].min()
    t1_end   = first_half["time_bin"].max()

    # Segundo tiempo en segundos:
    t2_start = second_half["time_bin"].min()
    t2_end   = second_half["time_bin"].max()

    # Conversion a minutos relativos
    t1_end_min = (t1_end - t1_start) / 60
    t2_end_min = (t2_end - t2_start) / 60

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)
    
    if 'in_possession_majority' in first_half.columns:
       plot_line_by_state(
        axes[0],
        first_half["t_min"],
        first_half[metric_col],
        first_half["in_possession_majority"],
        lw=2,
        color_in=HIGHLIGHT_COLOR,      # default matplotlib color
        color_out=PRIMARY_COLOR     # grey
        )
    else:
        axes[0].plot(first_half["t_min"], first_half[metric_col], linewidth=2)

    axes[0].set_title(f"{metric_col} - First Half", pad=20)
    axes[0].set_xlabel("Minutes")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks([0, 15, 30, t1_end_min])     # primer tiempo
    axes[0].set_xticklabels(["0", "15", "30", f"{t1_end_min:.1f}"])

    
    if 'in_possession_majority' in second_half.columns:
        plot_line_by_state(
        axes[1],
        second_half["t_min"],
        second_half[metric_col],
        second_half["in_possession_majority"],
        lw=2,
        color_in=HIGHLIGHT_COLOR,      # default matplotlib color
        color_out=PRIMARY_COLOR     # grey
        )
    else:
        axes[1].plot(second_half["t_min"], second_half[metric_col])

    axes[1].set_title(f"{metric_col} - Second Half", pad=20)
    axes[1].set_xlabel("Minutes")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks([0, 15, 30, t2_end_min])
    axes[1].set_xticklabels(["0", "15", "30", f"{t2_end_min:.1f}"])

    plt.tight_layout()
    plt.show()

def plot_timeline_by_metric_full_match(timeline_df, metric_col, title=None):
    """
    Plot a metric timeline for the full match on a single axis.

    Adds standard x-ticks every 15 minutes and draws a vertical line at
    the end of the first half (HT) based on `period_id`.

    Parameters
    ----------
    timeline_df : pandas.DataFrame
        Timeline DataFrame containing at least: time_bin, period_id, and the
        selected metric column.
    metric_col : str
        Column name of the metric to plot.
    title : str, optional
        Title to use for the figure. If None, a default title is used.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    - This function calls `ax.legend()` but does not assign labels explicitly
      unless the plotted artists include them.
    """
    df = timeline_df.copy().sort_values("time_bin")

    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(df["time_bin"], df[metric_col], linewidth=2)

    # Ticks estándar cada 15 min
    ticks = [0, 15*60, 30*60, 45*60, 60*60, 75*60, 90*60]
    labels = ["0", "15", "30", "45", "60", "75", "90"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    # Líneas especiales del partido real
    t_start = df["time_bin"].min()
    t_end   = df["time_bin"].max()
    ht_end  = df[df["period_id"]==1]["time_bin"].max()
    
    ax.axvline(ht_end, color="red", linestyle="--", alpha=0.8)
    ax.text(ht_end, ax.get_ylim()[1], "HT", ha="center", va="bottom", color="red")
    

    ax.set_xlabel("Minutes")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title(f"{metric_col} - Match Overview", pad=20)
    ax.legend()
    plt.tight_layout()
    plt.show()

def build_zone_matrix_from_metrics(
    metrics_by_zone: dict,
    metric: str,
    possession: str  # "in" o "out"
) -> np.ndarray:
    """
    Build a 3x3 zone matrix for a given metric and possession state.

    Converts a dictionary keyed by (zone_x, zone_y, "in"|"out") into a dense
    matrix of shape (ZONE_X, ZONE_Y), where each cell contains the metric value
    for that ball zone and possession state.

    Parameters
    ----------
    metrics_by_zone : dict
        Dictionary produced by `build_metrics_by_zone_possession`, where keys are
        (zone_x, zone_y, possession_str) and values contain metric fields.
    metric : str
        Metric name to extract (e.g., "compactness", "width", "block_height").
    possession : str
        Possession filter: "in" or "out".

    Returns
    -------
    numpy.ndarray
        A (ZONE_X, ZONE_Y) matrix with metric values and NaNs where unavailable.
    """
    m = np.full((ZONE_X, ZONE_Y), np.nan)

    for (zx, zy, pos), info in metrics_by_zone.items():
        if pos != possession:
            continue
        if metric in info:
            m[zx, zy] = info[metric]

    return m

def print_zone_matrix_with_labels(matrix):
    """
    Print a 3x3 zone matrix with human-readable zone labels.

    Parameters
    ----------
    matrix : array-like
        Matrix of shape (3, 3) aligned with (zone_x, zone_y).

    Returns
    -------
    None
        Prints formatted values to stdout.
    """
    for zx in range(3):
        row = []
        for zy in range(3):
            label = pre.get_zone_label(zx, zy)
            val = matrix[zx, zy]
            row.append(f"{label}: {val:.1f}")
        print(" | ".join(row))

def plot_zone_heatmap(
    metrics_by_zone: dict,
    metric_name: str,
    possession: str,         # "in" o "out"
    vmin=None,
    vmax=None,
    cmap=CMAP
):
    """
    Plot a 3x3 pitch heatmap for an aggregated metric by ball zone.

    Builds a 3x3 matrix from `metrics_by_zone` for the requested metric and
    possession state, then visualizes it on a vertical mplsoccer pitch.

    Parameters
    ----------
    metrics_by_zone : dict
        Dictionary keyed by (zone_x, zone_y, "in"|"out") containing aggregated
        metric values (output of `build_metrics_by_zone_possession`).
    metric_name : str
        Metric to visualize (e.g., "compactness", "width", "depth").
    possession : str
        Possession state to visualize: "in" or "out".
    vmin, vmax : float, optional
        Color scale bounds passed to `pitch.heatmap`.
    cmap : matplotlib.colors.Colormap, optional
        Colormap used for the heatmap.

    Returns
    -------
    tuple
        (fig, ax) where fig is a matplotlib Figure and ax is the pitch Axes.

    Notes
    -----
    - For "line_height" and "block_height", values are shifted to represent
      meters from own goal (0..105) and the colorbar unit is updated.
    - `pitch.label_heatmap` is used to print cell values on the pitch.
    """
 
    
    matrix = build_zone_matrix_from_metrics(
        metrics_by_zone,
        metric_name,
        possession
    )

    unit = METRIC_UNITS.get(metric_name, "")

    if metric_name in {"line_height", "block_height"}:
        matrix = np.where(np.isnan(matrix), np.nan, matrix + PITCH_HALF_LENGTH)
        unit = "m (from own goal)"


    statistic = np.flipud(matrix.T)   # (ZONE_Y, ZONE_X)

    x_grid = np.linspace(PITCH_MIN_X, PITCH_MAX_X, ZONE_X + 1)
    y_grid = np.linspace(PITCH_MIN_Y, PITCH_MAX_Y, ZONE_Y + 1)

    fig = plt.figure(figsize=(8,12))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax_pitch = fig.add_axes([0, .1,.91, .5])
    pitch, ax = get_pitch(ax_pitch)

    x_centers = (x_grid[:-1] + x_grid[1:]) / 2
    y_centers = (y_grid[:-1] + y_grid[1:]) / 2

    # cx, cy 2D con shape (3,3)
    cx, cy = np.meshgrid(x_centers, y_centers)

    stats = dict(
        statistic=statistic,  # shape (3,3)
        x_grid=x_grid,
        y_grid=y_grid,
        cx=cx,                # shape (3,3)
        cy=cy,                # shape (3,3)
    )

    hm = pitch.heatmap(stats, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)

    pitch.label_heatmap(
        stats,
        ax=ax,
        str_format="{:.1f}",
        color='white',
        fontsize=8
    )



    cbar = fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
    if unit:
        cbar.set_label(unit)
    plt.tight_layout()
    return fig, ax

def plot_team_shape_for_zone(
    shapes, zone_x, zone_y, in_possession,
    title_prefix="Average team shape"
):
    """
    Plot an average team shape for a specific ball zone and possession state.

    Parameters
    ----------
    shapes : dict
        Dictionary of shapes keyed by (zone_x, zone_y, possession_key).
        The expected structure is produced by your average-shape builders and
        contains at least: players, hull, ball_x_mean, ball_y_mean.
    zone_x : int
        Longitudinal zone index.
    zone_y : int
        Lateral zone index.
    in_possession : bool or str
        Possession selector used for the key lookup. (Note: this function uses
        the value directly in the key.)
    title_prefix : str, default="Average team shape"
        Title prefix for the plot (currently not enforced in the code).

    Returns
    -------
    tuple
        (fig, ax) where fig is a matplotlib Figure and ax is the pitch Axes.

    Raises
    ------
    ValueError
        If the requested (zone_x, zone_y, in_possession) key is not available.
    """
    key = (zone_x, zone_y, in_possession)
    if key not in shapes:
        raise ValueError(f"No shape for key {key}")

    shape = shapes[key]
    players = shape["players"]
    hull = shape["hull"]
    ball_x = shape["ball_x_mean"]
    ball_y = shape["ball_y_mean"]
    zone_label = shape.get("ball_zone_label", f"{zone_x},{zone_y}") \
                  if isinstance(shape, dict) else None

    fig = plt.figure(figsize=(8,12))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax_pitch = fig.add_axes([0, .1,.91, .5])
    pitch, ax = get_pitch(ax_pitch)

    
    if hull is not None :
        hull_path = list(hull.vertices) + [hull.vertices[0]]
        hull_pts = players.iloc[hull_path][["x", "y"]]
        xs = hull_pts["x"].to_numpy()
        ys = hull_pts["y"].to_numpy()

        pitch.plot(xs,ys,
            ax=ax, lw=1, color=HIGHLIGHT_COLOR, alpha=.5, linestyle='--')
        poly = list(zip(xs, ys))
        pitch.polygon([poly],
                ax=ax, facecolor=HIGHLIGHT_COLOR, alpha=0.15, zorder=2)
    
    
    pitch.scatter(
        players["x"],
        players["y"],
        s=40,
        zorder=5,
        ax=ax
    )
    
    
    pitch.scatter(
        ball_x, ball_y,
        marker="football", s=70,
        zorder=7,
        ax=ax
    )


    plt.tight_layout()
    return fig, ax

def plot_team_shape(shape, *,vertical_pitch:bool=False, show_metrics:bool=False, metric_column:str=None, in_possession:bool=True, is_submission: bool=True, show_legend: bool = True, fig: Figure =None, ax_pitch: Axes =None, ax_metrics: Axes =None):
    """
    Plot a single team shape (players, hull, and mean ball position) on a pitch.

    Optionally displays a side panel with key metrics and overlays specific
    metric annotations (e.g., width/depth arrows, block/line height lines,
    centroid marker).

    Parameters
    ----------
    shape : dict
        Shape dictionary containing:
        - players : DataFrame with x, y
        - hull : scipy.spatial.ConvexHull (optional)
        - ball_x_mean, ball_y_mean : float (optional)
        - metrics : dict (required if show_metrics=True)
    vertical_pitch : bool, default=False
        If True, draw the pitch vertically.
    show_metrics : bool, default=False
        If True, show a metrics panel via `_show_metrics`.
    metric_column : str, optional
        Metric name to annotate on the pitch (e.g., "width", "depth",
        "block_height", "line_height", "team_centroid").
    in_possession : bool, default=True
        Used to filter which metrics are shown in the panel.
    is_submission : bool, default=True
        If True, hides some metrics to match submission presentation choices.
    show_legend : bool, default=True
        Whether to display legend for annotations that add labeled artists.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into. If None, a new figure is created.
    ax_pitch : matplotlib.axes.Axes, optional
        Existing axes for the pitch. If None, axes are created.
    ax_metrics : matplotlib.axes.Axes, optional
        Existing axes for the metrics panel. If None, axes are created.

    Returns
    -------
    tuple
        (pitch, fig, ax) where pitch is an mplsoccer Pitch/VerticalPitch,
        fig is the matplotlib Figure, and ax is the pitch Axes.
    """
    players = shape["players"]
    hull = shape["hull"]
    ball_x = shape["ball_x_mean"]
    ball_y = shape["ball_y_mean"]

    if fig is None:
        fig = plt.figure(figsize=(8,12))
        fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    if ax_pitch is None:
        ax_pitch = fig.add_axes([0, .1,.91, .5])
    
    if ax_metrics is None:
        if vertical_pitch:
            ax_metrics = fig.add_axes([.7,.1,.15,.5], sharey=ax_pitch)
        else:
            ax_metrics = fig.add_axes([.91,.17,.15,.36], sharey=ax_pitch)
    
    pitch, ax = get_pitch(ax_pitch,vertical_pitch)

    
    if hull is not None :
        hull_path = list(hull.vertices) + [hull.vertices[0]]
        hull_pts = players.iloc[hull_path][["x", "y"]]
        xs = hull_pts["x"].to_numpy()
        ys = hull_pts["y"].to_numpy()

        pitch.plot(xs,ys,
            ax=ax, lw=1, color=HIGHLIGHT_COLOR, alpha=.5, linestyle='--')
        poly = list(zip(xs, ys))
        pitch.polygon([poly],
                ax=ax, facecolor=HIGHLIGHT_COLOR, alpha=0.15, zorder=2)

    pitch.scatter(
        players["x"],
        players["y"],
        s=40,
        zorder=5,
        color=HIGHLIGHT_COLOR,
        ax=ax
    )
    
    
    pitch.scatter(
        ball_x, ball_y,
        marker="football", s=70,
        zorder=7,
        ax=ax
    )

    if show_metrics:
        _show_metrics(fig, ax, ax_metrics, shape['metrics'], vertical_pitch, in_possession, is_submission=is_submission)
        if metric_column and 'metrics' in shape:
            plot_metric(fig,pitch,ax, shape['metrics'], metric_column, players,vertical_pitch, show_legend=show_legend)

    plt.tight_layout()
    return pitch, fig, ax

def plot_metric(fig,pitch, ax, metrics, metric_name, player_positions, is_vertical=True, show_legend: bool = True):
    """
    Overlay a specific metric annotation on the pitch.

    Supported annotations:
    - "width": double-headed arrow showing lateral spread (max y - min y)
    - "depth": double-headed arrow showing longitudinal spread (max x - min x)
    - "block_height": vertical line at the block height percentile
    - "line_height": vertical line at the line height percentile
    - "team_centroid": star marker at (team_centroid_x, team_centroid_y)

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the plot (not always used directly).
    pitch : mplsoccer.Pitch or mplsoccer.VerticalPitch
        Pitch instance used to draw and convert dimensions.
    ax : matplotlib.axes.Axes
        Axes to annotate.
    metrics : dict
        Dictionary of computed metrics (e.g., from `get_team_metrics`).
    metric_name : str
        Name of the metric annotation to draw.
    player_positions : pandas.DataFrame
        Player positions used to compute ranges, must include "x" and "y".
    is_vertical : bool, default=True
        Whether the pitch is vertical (affects coordinate mapping).
    show_legend : bool, default=True
        Whether to show legend for labeled annotations (e.g., centroid).

    Returns
    -------
    None
    """
    players = player_positions

    def P(x, y):
        return (y, x) if is_vertical else (x, y)

    
    x_len = (pitch.dim.right - pitch.dim.left)
    y_len = (pitch.dim.top - pitch.dim.bottom)
    x_pad = 0.03 * x_len if is_vertical else 0.05 * x_len
    y_pad = 0.05 * y_len if is_vertical else 0.03 * y_len

    def draw_arrow(p1, p2, text_xy, text):
        ax.annotate(
            "", xy=p1, xytext=p2,
            arrowprops=dict(arrowstyle="<->", lw=1, color=SECONDARY_TEXT_COLOR),
            zorder=9,
        )
        ax.text(
            text_xy[0], text_xy[1], text,
            va="center", ha="center", fontsize=8,
            zorder=9, color=SECONDARY_TEXT_COLOR
        )

    match metric_name:
        case "width":
            y_left = float(players["y"].min())
            y_right = float(players["y"].max())
            y_mid = (y_left + y_right) / 2

            x_anchor = float(players["x"].min()) - x_pad  
            
            p1 = P(x_anchor, y_left)
            p2 = P(x_anchor, y_right)
            txy = P(x_anchor - x_pad*0.6, y_mid)
            draw_arrow(p1, p2, txy, f"{(y_right - y_left):.1f} m")

        case "depth":
            x_min = float(players["x"].min())
            x_max = float(players["x"].max())
            x_mid = (x_min + x_max) / 2

            y_anchor = float(players["y"].min()) - y_pad  
            
            p1 = P(x_min, y_anchor)
            p2 = P(x_max, y_anchor)
            txy = P(x_mid, y_anchor - y_pad*0.8)
            draw_arrow(p1, p2, txy, f"{(x_max - x_min):.1f} m")

        case "block_height":
            x_val = float(metrics["block_height"])
            y_min, y_max = pitch.dim.bottom, pitch.dim.top

            pitch.plot([x_val, x_val], [y_min, y_max], ax=ax, ls="--", lw=1, color=SECONDARY_TEXT_COLOR)

            
            label = f"{height_from_own_goal(x_val):.1f} m"
            
            txy = P(x_val + x_pad*0.6, y_max - y_pad)
            ax.text(txy[0], txy[1], label, fontsize=8, color=SECONDARY_TEXT_COLOR, ha="center", va="center", zorder=9)

        case "line_height":
            x_val = float(metrics["line_height"])
            y_min, y_max = pitch.dim.bottom, pitch.dim.top

            pitch.plot([x_val, x_val], [y_min, y_max], ax=ax, ls="--", lw=1, color=SECONDARY_TEXT_COLOR)

            label = f"{height_from_own_goal(x_val):.1f} m"
            txy = P(x_val - x_pad*0.6, y_max - y_pad)
            ax.text(txy[0], txy[1], label, fontsize=8, color=SECONDARY_TEXT_COLOR, ha="center", va="center", zorder=9)

        case 'team_centroid':
            pitch.scatter(
                metrics['team_centroid_x'], metrics['team_centroid_y'],
                marker="*", s=70,
                zorder=7,
                ax=ax,
                color=HIGHLIGHT_COLOR,
                label='Team centroid'
            )
            if show_legend:
                legend = ax.legend(facecolor=BACKGROUND_COLOR, edgecolor='None',loc=(1,.032) if is_vertical else (1,.05), markerscale=.5, fontsize=8)
            
        case _:
            return

def _show_metrics(fig: Figure,ax_pitch: Axes,ax_metrics: Axes, metrics: dict, is_vertical: bool, in_possession: bool, is_submission: bool=True):
    """
    Render a compact metrics panel next to the pitch.

    Filters and formats a subset of metrics and prints them as text on a
    dedicated axes (with axis turned off).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Parent figure.
    ax_pitch : matplotlib.axes.Axes
        Pitch axes (used for alignment/sharing).
    ax_metrics : matplotlib.axes.Axes
        Axes where the metrics panel is drawn.
    metrics : dict
        Dictionary of metric values.
    is_vertical : bool
        If True, uses a vertical layout for the metrics panel.
    in_possession : bool
        Controls filtering of block/line height depending on possession state.
    is_submission : bool, default=True
        If True, hides additional metrics to match submission presentation.

    Returns
    -------
    None
    """
    not_shown_metrics = ['team_centroid_x','team_centroid_y',"n_players_used","centroid_ball_dist"]
    if is_submission:
        not_shown_metrics += ['block_height','line_height']
    metrics = {k: v for k, v in metrics.items() if k not in not_shown_metrics}

    if in_possession:
        metrics = {k: v for k, v in metrics.items() if k != 'block_height'}
    else:
        metrics = {k: v for k, v in metrics.items() if k != 'line_height'}

    def fmt(k, v):
        if v is None:
            return "—"
        if k == "compactness":
            return f"{float(v):.0f} m²"
        if k == "line_height" or k == "block_height":
            return f"{float(height_from_own_goal(v)):.1f} m"
        return f"{float(v):.1f} m"

    if is_vertical:
        if ax_metrics is None:
            ax_metrics = fig.add_axes([.7,.1,.15,.5], sharey=ax_pitch)
        index = 5
        for k in metrics.keys():
            val = metrics.get(k, None)
            label = k.capitalize().replace("_", " ")
            value = fmt(k, val)
            y = 10 * index
            ax_metrics.text(x=.5, y=y, s=label, fontsize=10, color=SECONDARY_TEXT_COLOR, ha='center')
            ax_metrics.text(x=.5, y=y-4, s=value, fontsize=12, color=PRIMARY_TEXT_COLOR, ha='center')
            index -= 1
    else:
        if ax_metrics is None:
            ax_metrics = fig.add_axes([.91,.17,.15,.36], sharey=ax_pitch)
        index = 2
        for k in metrics.keys():
            val = metrics.get(k, None)
            label = k.capitalize().replace("_", " ")
            value = fmt(k, val)
            y = 10 * index
            ax_metrics.text(x=.5, y=y, s=label, fontsize=10, color=SECONDARY_TEXT_COLOR, ha='center')
            ax_metrics.text(x=.5, y=y-4, s=value, fontsize=12, color=PRIMARY_TEXT_COLOR, ha='center')
            index -= 1

    ax_metrics.axis('off')
        
def plot_comparission_shapes(shape_left, shape_right,*,vertical_pitch:bool=False, show_metrics:bool=False, metric_column:str=None, titles:list=None, in_possession:bool=True, is_submission: bool=True):
    """
    Plot two team shapes side-by-side for comparison.

    Draws two pitches and optionally displays metric panels and pitch
    annotations, enabling visual comparison between two average shapes
    (e.g., different possession states, phases, or ball zones).

    Parameters
    ----------
    shape_left : dict
        Shape dictionary for the left plot.
    shape_right : dict
        Shape dictionary for the right plot.
    vertical_pitch : bool, default=False
        If True, draw pitches vertically.
    show_metrics : bool, default=False
        If True, show metrics panels for both shapes.
    metric_column : str, optional
        Metric annotation to overlay on the pitch (see `plot_metric`).
    titles : list of str, optional
        Titles for [left, right] plots.
    in_possession : bool, default=True
        Possession context passed to metric filtering/display.
    is_submission : bool, default=True
        Controls which metrics are shown for submission-friendly visuals.

    Returns
    -------
    tuple
        (pitch_left, pitch_right, fig, ax_left, ax_right)
    """
    fig = plt.figure(figsize=(8,12))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax_pitch_left = fig.add_axes([0, .1,.91, .5])

    ax_metrics_left = fig.add_axes([.7,.1,.15,.5], sharey=ax_pitch_left)

    ax_pitch_right = fig.add_axes([.67, .1,.91, .5])
    ax_metrics_right = fig.add_axes([1.35,.1,.15,.5], sharey=ax_pitch_right)

    pitch_left, fig, ax_pitch_left = plot_team_shape(shape_left, fig=fig, ax_pitch=ax_pitch_left, ax_metrics=ax_metrics_left, show_metrics=show_metrics, metric_column=metric_column, vertical_pitch=vertical_pitch, in_possession=in_possession, is_submission=is_submission, show_legend=True)
    pitch_right, fig, ax_pitch_right = plot_team_shape(shape_right, fig=fig, ax_pitch=ax_pitch_right, ax_metrics=ax_metrics_right,  show_metrics=show_metrics, metric_column=metric_column, vertical_pitch=vertical_pitch, in_possession=in_possession, is_submission=is_submission, show_legend=False)
    ax_pitch_left.set_title(titles[0], color=PRIMARY_TEXT_COLOR, fontsize=12, pad=10)
    ax_pitch_right.set_title(titles[1], color=PRIMARY_TEXT_COLOR, fontsize=12, pad=10)

    return pitch_left, pitch_right, fig, ax_pitch_left, ax_pitch_right