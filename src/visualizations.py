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
    here = Path(__file__).resolve().parent
    root = here.parent                               # .../analytics_cup_research
    fonts_dir = here / "assets" / "fonts" 

    regular = fonts_dir / "Manrope-Regular.ttf"
    bold = fonts_dir / "Manrope-Bold.ttf"

    # Validación útil (para debug)
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
    Convierte coordenada SkillCorner x (-52.5..52.5)
    a metros desde el arco propio (0..105)
    """
    return float(x + PITCH_HALF_LENGTH)

def get_pitch(ax:Axes, vertical: bool = True, draw: bool = True) -> tuple:
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
    Construye una matriz 3x3 [zone_x, zone_y] a partir del dict
    devuelto por build_metrics_by_zone_possession, filtrando por
    métrica e in/out possession.
    """
    m = np.full((ZONE_X, ZONE_Y), np.nan)

    for (zx, zy, pos), info in metrics_by_zone.items():
        if pos != possession:
            continue
        if metric in info:
            m[zx, zy] = info[metric]

    return m

def print_zone_matrix_with_labels(matrix):
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
    Dibuja el heatmap 3x3 de una métrica para toda la cancha,
    usando el dict de métricas por zona/posesión.

    metrics_by_zone: dict de build_metrics_by_zone_possession
    metric_name: "compactness", "width", etc.
    possession: "in" o "out"
    """
 
    # 1) construir la matriz 3x3 desde el dict
    matrix = build_zone_matrix_from_metrics(
        metrics_by_zone,
        metric_name,
        possession
    )

    unit = METRIC_UNITS.get(metric_name, "")

    if metric_name in {"line_height", "block_height"}:
        matrix = np.where(np.isnan(matrix), np.nan, matrix + PITCH_HALF_LENGTH)
        unit = "m (from own goal)"

    # 2) construir el dict `stats` que espera pitch.heatmap
    #    OJO: pcolormesh espera statistic[y, x], así que transponemos
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

    # hm = pitch.heatmap(stats, ax=ax, vmin=vmin, vmax=vmax,cmap=cmap)
    # pitch.label_heatmap(
    #     stats,
    #     ax=ax,
    #     str_format="{:.1f}",
    #     color="green",
    #     fontsize=8
    # )

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
    Dibuja la forma promedio del equipo en una zona concreta y estado de posesión.
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

    # Convex hull (asumiendo mismo sistema de coordenadas que el pitch)
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
    
    # Balón promedio
    pitch.scatter(
        ball_x, ball_y,
        marker="football", s=70,
        zorder=7,
        ax=ax
    )


    plt.tight_layout()
    return fig, ax

def plot_team_shape(shape, *,vertical_pitch:bool=False, show_metrics:bool=False, metric_column:str=None, in_possession:bool=True, is_submission: bool=True, show_legend: bool = True, fig: Figure =None, ax_pitch: Axes =None, ax_metrics: Axes =None):

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

    # Convex hull (asumiendo mismo sistema de coordenadas que el pitch)
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
    
    # Balón promedio
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
    players = player_positions

    def P(x, y):
        # mapea coords (x,y) reales a coords de dibujo según orientación
        return (y, x) if is_vertical else (x, y)

    # offsets relativos
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

            x_anchor = float(players["x"].min()) - x_pad  # a la izquierda del bloque
            # flecha en y, a x constante
            p1 = P(x_anchor, y_left)
            p2 = P(x_anchor, y_right)
            txy = P(x_anchor - x_pad*0.6, y_mid)
            draw_arrow(p1, p2, txy, f"{(y_right - y_left):.1f} m")

        case "depth":
            x_min = float(players["x"].min())
            x_max = float(players["x"].max())
            x_mid = (x_min + x_max) / 2

            y_anchor = float(players["y"].min()) - y_pad  # abajo del bloque
            # flecha en x, a y constante
            p1 = P(x_min, y_anchor)
            p2 = P(x_max, y_anchor)
            txy = P(x_mid, y_anchor - y_pad*0.8)
            draw_arrow(p1, p2, txy, f"{(x_max - x_min):.1f} m")

        case "block_height":
            x_val = float(metrics["block_height"])
            y_min, y_max = pitch.dim.bottom, pitch.dim.top

            pitch.plot([x_val, x_val], [y_min, y_max], ax=ax, ls="--", lw=1, color=SECONDARY_TEXT_COLOR)

            # texto cerca de la línea
            label = f"{height_from_own_goal(x_val):.1f} m"
            # ubicarlo un poquito hacia el lado “fuera” del pitch
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