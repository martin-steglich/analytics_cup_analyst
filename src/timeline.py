from . import preprocessing as pre
from . import metrics as metrics
import pandas as pd

def build_timeline(df_frame_metrics: pd.DataFrame, window_seconds: float = 5.0) -> pd.DataFrame:
    """
    Aggregate frame-level team metrics into a time-based timeline.

    Frames are grouped into fixed-size time bins and team metrics are averaged
    within each bin. The resulting timeline provides a smoothed, interpretable
    view of how team shape evolves over time.

    Parameters
    ----------
    df_frame_metrics : pandas.DataFrame
        Frame-level metrics DataFrame. Must include at least:
        - time (seconds since match start)
        - frame_id
        - period_id
        - in_possession
        - team shape metrics (e.g., compactness, width, depth, etc.)
    window_seconds : float, default=5.0
        Size of the temporal aggregation window in seconds.

    Returns
    -------
    pandas.DataFrame
        Time-binned DataFrame with one row per time bin, including:
        - *_mean columns for each team metric
        - n_frames : number of frames in the bin
        - in_possession_share : proportion of frames in possession
        - in_possession_majority : boolean flag (>= 60% possession)

    Notes
    -----
    - Metric values are aggregated using the mean.
    - Possession is summarized both as a share and as a majority flag to
      support binary and continuous analyses.
    """
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

    
    timeline = timeline.rename(columns={
        col: f"{col}_mean" for col in metric_cols
    })

    timeline["in_possession_majority"] = timeline["in_possession_share"] >= 0.6

    return timeline

def build_half_timelines(df_frame_metrics: pd.DataFrame, window_seconds: float = 5.0) -> pd.DataFrame:
    """
    Build separate timelines for first and second halves.

    The match is split by `period_id`, and independent timelines are generated
    for each half. Time is additionally expressed relative to the start of
    each half (in minutes).

    Parameters
    ----------
    df_frame_metrics : pandas.DataFrame
        Frame-level metrics DataFrame containing multiple periods.
    window_seconds : float, default=5.0
        Size of the temporal aggregation window in seconds.

    Returns
    -------
    tuple of pandas.DataFrame
        (timeline_first_half, timeline_second_half), each containing:
        - absolute time bins
        - relative time bins (minutes since half start)
        - aggregated team metrics

    Notes
    -----
    - Relative time allows easier visual comparison between halves.
    - Periods are assumed to be labeled as 1 (first half) and 2 (second half).
    """
    df = df_frame_metrics.copy()
    first_half  = df[df["period_id"] == 1]
    second_half = df[df["period_id"] == 2]

    t1_start = first_half["time"].min()
    t2_start = second_half["time"].min()

    timeline_1 = build_timeline(first_half, window_seconds)
    timeline_2 = build_timeline(second_half, window_seconds)

    
    timeline_1["time_bin_relative"] = (timeline_1["time_bin"] - t1_start) / 60
    timeline_2["time_bin_relative"] = (timeline_2["time_bin"] - t2_start) / 60

    return timeline_1, timeline_2

def build_timeline_smoothed(df_frame_metrics: pd.DataFrame, window_seconds: float = 5.0, smooth_window: int = 9):
    """
    Build a time-binned timeline with rolling smoothing applied.

    First aggregates frame-level metrics into time bins using
    `build_timeline`, then applies rolling mean smoothing to selected
    metrics to reduce short-term noise.

    Parameters
    ----------
    df_frame_metrics : pandas.DataFrame
        Frame-level metrics DataFrame.
    window_seconds : float, default=5.0
        Size of the temporal aggregation window in seconds.
    smooth_window : int, default=9
        Window size (in number of bins) for rolling mean smoothing.

    Returns
    -------
    pandas.DataFrame
        Timeline DataFrame with additional *_smooth columns for selected metrics.

    Notes
    -----
    - Metrics representing slower tactical trends (e.g., compactness, width)
      use a longer smoothing window.
    - Lateral centroid movement uses a shorter smoothing window to preserve
      meaningful short-term variation.
    """
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

def build_timeline_by_possession_and_third(df_frame_metrics: pd.DataFrame, window_seconds: float = 5.0):
    """
    Build timelines grouped by possession state and ball longitudinal third.

    Aggregates frame-level metrics by:
    - time bin
    - possession state (in / out of possession)
    - ball_zone_x (longitudinal third of the pitch)

    This enables contextual timeline analysis depending on where the ball is
    and whether the team has possession.

    Parameters
    ----------
    df_frame_metrics : pandas.DataFrame
        Frame-level metrics DataFrame. Must include:
        - time
        - in_possession
        - ball_zone_x
        - team shape metrics
    window_seconds : float, default=5.0
        Size of the temporal aggregation window in seconds.

    Returns
    -------
    pandas.DataFrame
        Aggregated timeline with one row per
        (time_bin, in_possession, ball_zone_x) combination, including:
        - *_mean columns for team metrics
        - n_frames per group

    Notes
    -----
    - To ignore ball thirds after aggregation, drop `ball_zone_x` and
      re-aggregate by (time_bin, in_possession).
    - This representation is particularly useful for contextualized
      tactical analysis.
    """

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

    timeline = (
        df.groupby(["time_bin", "in_possession", "ball_zone_x"])
          .agg(agg_dict)
          .reset_index()
          .rename(columns={"frame_id": "n_frames"})
          .sort_values(["in_possession", "ball_zone_x", "time_bin"])
    )

    
    timeline = timeline.rename(columns={
        col: f"{col}_mean" for col in metric_cols
    })

    return timeline

def smooth_timeline_by_possession_and_third(timeline_pt: pd.DataFrame, smooth_window: int = 9):
    """
    Apply rolling smoothing to a possession- and third-specific timeline.

    Smoothing is applied independently within each
    (in_possession, ball_zone_x) group to preserve contextual separation.

    Parameters
    ----------
    timeline_pt : pandas.DataFrame
        Timeline produced by `build_timeline_by_possession_and_third`.
        Must contain *_mean metric columns.
    smooth_window : int, default=9
        Window size (in number of bins) for rolling mean smoothing.

    Returns
    -------
    pandas.DataFrame
        Timeline DataFrame with additional *_smooth columns for selected metrics.

    Notes
    -----
    - Group-wise smoothing prevents information leakage across possession
      states or pitch thirds.
    - Longitudinal and compactness-related metrics use a longer smoothing
      window than lateral centroid movement.
    """

    timeline = timeline_pt.copy()


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

    
    for (in_pos, third), grp in timeline.groupby(["in_possession", "ball_zone_x"]):
        idx = grp.index

        
        for col in metrics_long_trend:
            timeline.loc[idx, f"{col}_smooth"] = (
                grp[col]
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
                .values
            )

        
        for col in metrics_short_trend:
            timeline.loc[idx, f"{col}_smooth"] = (
                grp[col]
                .rolling(window=max(3, smooth_window // 3), center=True, min_periods=1)
                .mean()
                .values
            )

    return timeline

def build_timeline_by_possession_and_third_smoothed(
    df_frame_metrics: pd.DataFrame,
    window_seconds: float = 5.0,
    smooth_window: int = 9
) -> pd.DataFrame:
    """
    Build and smooth timelines grouped by possession state and ball third.

    Convenience wrapper that combines:
    - `build_timeline_by_possession_and_third`
    - `smooth_timeline_by_possession_and_third`

    Parameters
    ----------
    df_frame_metrics : pandas.DataFrame
        Frame-level metrics DataFrame.
    window_seconds : float, default=5.0
        Size of the temporal aggregation window in seconds.
    smooth_window : int, default=9
        Window size for rolling mean smoothing.

    Returns
    -------
    pandas.DataFrame
        Smoothed timeline grouped by time bin, possession state,
        and ball longitudinal third.
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
