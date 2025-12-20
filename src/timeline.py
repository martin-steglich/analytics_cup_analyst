from . import preprocessing as pre
from . import metrics as metrics
import pandas as pd

def build_timeline(df_frame_metrics, window_seconds=5.0):
    df = df_frame_metrics.copy()
    df["time_bin"] = (df["time"] // window_seconds) * window_seconds

    metric_cols = [
        "compactness",
        "width",
        "depth",
        "team_centroid_x",
        "team_centroid_y",
        "block_height",
        "line_height",
        "team_spread",
    ]

    agg_dict = {col: "mean" for col in metric_cols}
    agg_dict["frame_id"] = "count"
    agg_dict["period_id"] = "first"
    agg_dict["in_possession"] = "mean"

    timeline = (
        df.groupby("time_bin")
          .agg(agg_dict)
          .reset_index()
          .rename(columns={"frame_id": "n_frames", "in_possession": "in_possession_share"})
          .sort_values("time_bin")
    )

    # renombrar a *_mean si querés ser explícito
    timeline = timeline.rename(columns={
        col: f"{col}_mean" for col in metric_cols
    })

    timeline["in_possession_majority"] = timeline["in_possession_share"] >= 0.6

    return timeline

def build_half_timelines(df_frame_metrics, window_seconds=5.0):
    df = df_frame_metrics.copy()
    first_half  = df[df["period_id"] == 1]
    second_half = df[df["period_id"] == 2]

    t1_start = first_half["time"].min()
    t2_start = second_half["time"].min()

    timeline_1 = build_timeline(first_half, window_seconds)
    timeline_2 = build_timeline(second_half, window_seconds)

    # tiempo relativo
    timeline_1["time_bin_relative"] = (timeline_1["time_bin"] - t1_start) / 60
    timeline_2["time_bin_relative"] = (timeline_2["time_bin"] - t2_start) / 60

    return timeline_1, timeline_2

def build_timeline_smoothed(df_frame_metrics, window_seconds=5.0, smooth_window=9):
    timeline = build_timeline(df_frame_metrics, window_seconds)

    metrics_long_trend = [
        "compactness_mean",
        "width_mean",
        "depth_mean",
        "team_spread_mean",
        "block_height_mean",
        "line_height_mean",
        "team_centroid_x_mean",
    ]

    metrics_short_trend = [
        "team_centroid_y_mean",
    ]

    for col in metrics_long_trend:
        timeline[f"{col}_smooth"] = (
            timeline[col]
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
        )

    for col in metrics_short_trend:
        timeline[f"{col}_smooth"] = (
            timeline[col]
            .rolling(window=max(3, smooth_window // 3), center=True, min_periods=1)
            .mean()
        )

    return timeline

def build_timeline_by_possession_and_third(df_frame_metrics, window_seconds=5.0):
    """
    Genera timelines agrupados por:
      - time_bin
      - in_possession (True/False)
      - ball_zone_x (0,1,2)  --> tercio longitudinal del balón

    Si luego querés ignorar el tercio, bastaría hacer:
        timeline.drop(columns=["ball_zone_x"])
        timeline.groupby(["time_bin", "in_possession"]).mean()

    df_frame_metrics debe tener:
        time, in_possession, ball_zone_x
        y todas las métricas numéricas del equipo.
    """

    df = df_frame_metrics.copy()

    # Asegurar columna time_bin
    df["time_bin"] = (df["time"] // window_seconds) * window_seconds

    metric_cols = [
        "compactness",
        "width",
        "depth",
        "team_centroid_x",
        "team_centroid_y",
        "block_height",
        "line_height",
        "team_spread",
    ]

    # Definir cómo agregar
    agg_dict = {col: "mean" for col in metric_cols}
    agg_dict["frame_id"] = "count"       # cuántos frames caen en ese bin
    agg_dict["period_id"] = "first"      # por si lo necesitás después

    timeline = (
        df.groupby(["time_bin", "in_possession", "ball_zone_x"])
          .agg(agg_dict)
          .reset_index()
          .rename(columns={"frame_id": "n_frames"})
          .sort_values(["in_possession", "ball_zone_x", "time_bin"])
    )

    # Renombrar métricas a *_mean para dejar claro qué son:
    timeline = timeline.rename(columns={
        col: f"{col}_mean" for col in metric_cols
    })

    return timeline

def smooth_timeline_by_possession_and_third(timeline_pt, smooth_window=9):
    """
    Aplica smoothing (rolling mean) sobre un timeline generado por
    build_timeline_by_possession_and_third().

    No altera las columnas de agrupación:
        time_bin, in_possession, ball_zone_x

    El smoothing se aplica por grupo (posesión + tercio).

    timeline_pt debe tener columnas *_mean.
    """

    timeline = timeline_pt.copy()

    # métricas que serán suavizadas (siguen tu convención)
    metrics_long_trend = [
        "compactness_mean",
        "width_mean",
        "depth_mean",
        "team_spread_mean",
        "block_height_mean",
        "line_height_mean",
        "team_centroid_x_mean",
    ]

    metrics_short_trend = [
        "team_centroid_y_mean",
    ]

    # Aplicar smoothing por grupo (in_possession + tercio)
    for (in_pos, third), grp in timeline.groupby(["in_possession", "ball_zone_x"]):
        idx = grp.index

        # long trends
        for col in metrics_long_trend:
            timeline.loc[idx, f"{col}_smooth"] = (
                grp[col]
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
                .values
            )

        # short trends
        for col in metrics_short_trend:
            timeline.loc[idx, f"{col}_smooth"] = (
                grp[col]
                .rolling(window=max(3, smooth_window // 3), center=True, min_periods=1)
                .mean()
                .values
            )

    return timeline

def build_timeline_by_possession_and_third_smoothed(
    df_frame_metrics,
    window_seconds=5.0,
    smooth_window=9
) -> pd.DataFrame:
    """
    Combina build_timeline_by_possession_and_third() y smoothing.
    """
    timeline = build_timeline_by_possession_and_third(
        df_frame_metrics,
        window_seconds
    )

    timeline_smoothed = smooth_timeline_by_possession_and_third(
        timeline,
        smooth_window
    )

    return timeline_smoothed
