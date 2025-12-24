from . import preprocessing as pre
from .phases import POSSESSION_COLUMNS
from scipy.spatial import ConvexHull, QhullError
import numpy as np
import pandas as pd

ZONE_ROWS = 3
ZONE_COLS = 3
METRIC_UNITS = {
    "compactness": "m²",
    "width": "m",
    "depth": "m",
    "block_height": "m",
    "line_height": "m",
    "team_spread": "m",
    "centroid_ball_dist": "m",
    "team_centroid_x": "m",
    "team_centroid_y": "m",
}

def get_convex_hull(df: pd.DataFrame) -> ConvexHull:
    """
    Compute the 2D convex hull for points provided in a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the point coordinates. This function expects two
        columns named 'x' and 'y' with numeric types. Rows represent points
        (x, y).

    Returns
    -------
    scipy.spatial.ConvexHull
        A ConvexHull object (from scipy.spatial) describing the convex hull of
        the input points. The object contains attributes such as `vertices`
        (indices of hull points), `points` (the input points), and `area`/`volume`
        depending on dimensionality.

    Raises
    ------
    KeyError
        If either the 'x' or 'y' column is missing from `df`.
    ValueError
        If there are fewer than 3 non-collinear points (ConvexHull requires at
        least 3 non-collinear points to define a 2D hull) or if all points are
        identical/degenerate.

    Notes
    -----
    - The function uses scipy.spatial.ConvexHull (Qhull) under the hood; behavior
      (including handling of duplicate points and degenerate cases) follows
      SciPy's implementation.
    - It is recommended to drop NaNs and non-finite values from `df` before
      calling this function.

    Examples
    --------
    >>> # df is a pandas DataFrame with columns 'x' and 'y'
    >>> # hull = get_convex_hull(df)
    >>> # hull.vertices -> indices of points forming the convex hull
    """
    points = df[['x', 'y']].values
    return ConvexHull(points)

def compute_team_compactness(df: pd.DataFrame) -> float:
    """
    Compute the compactness of a team as the area of the 2D convex hull of player positions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns "x" and "y" with player coordinates.
        Rows with NaN in either "x" or "y" are ignored.

    Returns
    -------
    float
        The 2D area (volume attribute of scipy.spatial.ConvexHull) of the convex hull
        formed by the valid (non-NaN) (x, y) points. Returns numpy.nan if:
        - fewer than 3 valid points remain after NaN filtering, or
        - the convex hull computation fails (e.g., raises scipy.spatial.qhull.QhullError).

    Notes
    -----
    - This function treats "compactness" as the geometric area occupied by the team on the field.
    - The implementation relies on scipy.spatial.ConvexHull and its `.volume` attribute for 2D hull area.
    """
    pts = df[["x","y"]].to_numpy()
    pts = pts[~np.isnan(pts).any(axis=1)]
    if pts.shape[0] < 3:
        return np.nan
    try:
        return ConvexHull(pts).volume
    except QhullError:
        return np.nan

def compute_team_width(df: pd.DataFrame) -> float:
    """
    Compute the team width as the range of players' y-coordinates.

    Width is defined as the difference between the maximum and minimum y values
    among all players in the input, representing the lateral spread of the team
    in a given frame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing player positions for one team in one frame.
        Must include a numeric column "y" (lateral coordinate in meters).

    Returns
    -------
    float
        Team width in meters, computed as max(y) - min(y).

    Raises
    ------
    IndexError
        If `df` is empty (no rows), since no minimum/maximum can be computed.

    Notes
    -----
    - This function assumes `df` contains valid numeric y values (no NaNs).
      If NaNs may be present, call `df.dropna(subset=["y"])` before computing.
    - Equivalent to `df["y"].max() - df["y"].min()`.

    Examples
    --------
    >>> width = compute_team_width(df_frame)
    >>> print(width)
    35.5
    """
    top_y = df.nlargest(1, 'y')
    less_y = df.nsmallest(1, 'y')

    min_y = less_y.y.values[0]
    max_y = top_y.y.values[0]
    return max_y - min_y

def compute_team_depth(df: pd.DataFrame) -> float:
    """
    Compute the team depth as the range of players' x-coordinates.

    Depth is defined as the difference between the maximum and minimum x values
    among all players in the input, representing the longitudinal spread of the
    team in a given frame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing player positions for one team in one frame.
        Must include a numeric column "x" (longitudinal coordinate in meters).

    Returns
    -------
    float
        Team depth in meters, computed as max(x) - min(x).

    Raises
    ------
    IndexError
        If `df` is empty (no rows), since no minimum/maximum can be computed.

    Notes
    -----
    - This function assumes `df` contains valid numeric x values (no NaNs).
      If NaNs may be present, call `df.dropna(subset=["x"])` before computing.
    - Equivalent to `df["x"].max() - df["x"].min()`.

    Examples
    --------
    >>> depth = compute_team_depth(df_frame)
    >>> print(depth)
    42.0
    """
    top_x = df.nlargest(1, 'x')
    less_x = df.nsmallest(1, 'x')

    min_x = less_x.x.values[0]
    max_x = top_x.x.values[0]

    return max_x - min_x

def compute_team_block_height(df: pd.DataFrame, percentile: int = 80) -> float:
    """
    Compute the team block height as a percentile of players' x-coordinates.

    This metric is commonly used as a proxy for how high the team is positioned
    along the pitch length at a given frame. By default, the 80th percentile is
    used to approximate the "front" of the block while being less sensitive to
    a single extreme outlier than the maximum.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing player positions for a single team in a single frame.
        Must include column "x" (longitudinal coordinate in meters).
    percentile : int, default=80
        Percentile of the x-coordinate distribution to compute. Typical values:
        - 80: higher line / block front approximation
        - 50: median team height

    Returns
    -------
    float
        The selected percentile of x-coordinates (meters).

    Notes
    -----
    - Assumes the pitch is oriented so that increasing x corresponds to moving
      towards the opponent's goal. If your coordinate system differs, interpret
      accordingly.
    - This function does not remove NaNs; ensure `df["x"]` is clean beforehand
      (e.g., `dropna`).
    """
    return float(np.percentile(df["x"], percentile))

def compute_team_line_height(df: pd.DataFrame, percentile: int = 20) -> float:
    """
    Compute the team line height as a percentile of players' x-coordinates.

    This metric can be used as a proxy for the "back" of the team block
    (e.g., defensive line position) in a given frame. By default, the 20th
    percentile is used to reduce sensitivity to a single deep outlier compared
    to the minimum.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing player positions for a single team in a single frame.
        Must include column "x" (longitudinal coordinate in meters).
    percentile : int, default=20
        Percentile of the x-coordinate distribution to compute. Typical values:
        - 20: defensive line / block back approximation
        - 50: median team height

    Returns
    -------
    float
        The selected percentile of x-coordinates (meters).

    Notes
    -----
    - Assumes the pitch is oriented so that increasing x corresponds to moving
      towards the opponent's goal. If your coordinate system differs, interpret
      accordingly.
    - This function does not remove NaNs; ensure `df["x"]` is clean beforehand.
    """
    return float(np.percentile(df["x"], percentile))

def compute_team_centroid(df: pd.DataFrame):
    """
    Compute the team's centroid (mean position) for a single frame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing player positions for one team in one frame.
        Must include numeric columns "x" and "y" in meters. Typically this
        DataFrame should contain only outfield players (goalkeeper excluded).

    Returns
    -------
    (float, float)
        Tuple (cx, cy) with the mean x and mean y coordinates (meters).

    Notes
    -----
    - The centroid is a basic building block for other shape metrics such as
      spread and centroid-to-ball distance.
    - NaNs should be removed before calling this function.
    """
    # sup: df tiene solo jugadores de campo del equipo en ese frame
    cx = df['x'].mean()
    cy = df['y'].mean()
    return float(cx), float(cy)

def compute_team_spread(df: pd.DataFrame):
    """
    Compute team spread as the root-mean-square distance to the team centroid.

    For a single frame, this measures how dispersed the players are around the
    centroid. It is computed as:

        spread = sqrt(mean((x - cx)^2 + (y - cy)^2))

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing player positions for one team in one frame.
        Must include numeric columns "x" and "y" in meters.

    Returns
    -------
    float
        Root-mean-square distance to the centroid (meters).

    Notes
    -----
    - This implementation always returns a value in meters (RMSD).
    - NaNs should be removed before calling this function.
    """
    cx, cy = compute_team_centroid(df)
    dx = df['x'].to_numpy() - cx
    dy = df['y'].to_numpy() - cy
    msd = np.mean(dx*dx + dy*dy)  # m^2
    return float(np.sqrt(msd))

def compute_centroid_ball_distance(df: pd.DataFrame, ball_x: float, ball_y: float):
    """
    Compute the Euclidean distance between the team centroid and the ball.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing player positions for one team in one frame.
        Must include numeric columns "x" and "y" in meters.
    ball_x : float
        Ball x-coordinate (meters) for the same frame.
    ball_y : float
        Ball y-coordinate (meters) for the same frame.

    Returns
    -------
    float
        Distance from team centroid to ball position (meters).

    Notes
    -----
    - NaNs should be removed from `df` before calling this function.
    - Interpretation depends on the coordinate convention (attacking direction).
    """
    cx, cy = compute_team_centroid(df)
    dx = cx - ball_x
    dy = cy - ball_y
    d2 = dx*dx + dy*dy
    return float(np.sqrt(d2))

def get_team_metrics(df: pd.DataFrame, as_dataframe: bool = True):
    """
    Compute a set of team shape metrics for a single frame.

    The function expects a DataFrame containing player positions for a single
    team and frame, and optionally ball coordinates. It computes:
    - compactness (convex hull area)
    - width (y range)
    - depth (x range)
    - block_height (x percentile, default 80)
    - line_height (x percentile, default 20)
    - team centroid (cx, cy)
    - team spread (RMS distance to centroid)
    - centroid-to-ball distance (if ball_x/ball_y are available)
    - number of players used (after NaN filtering)

    Parameters
    ----------
    df : pandas.DataFrame
        Player-level data for one team in one frame. Must include "x" and "y".
        If "ball_x" and "ball_y" are present, centroid-to-ball distance is added.
        Rows with NaN in "x" or "y" are dropped.
    as_dataframe : bool, default=True
        If True, returns a single-row DataFrame. If False, returns a dict.

    Returns
    -------
    pandas.DataFrame or dict
        A single-row DataFrame (if `as_dataframe=True`) or a dict (otherwise)
        containing the computed metrics. If no valid player points remain after
        filtering, returns an empty DataFrame or empty dict.

    Notes
    -----
    - This function assumes the input corresponds to a single frame. If multiple
      frames are present, results will mix positions and may be meaningless.
    - Units:
        compactness: m²
        width, depth, block_height, line_height, spread, centroid_ball_dist: m
        team_centroid_x, team_centroid_y: m
    """
    df = df.dropna(subset=["x","y"])
    pts = df[["x","y"]].to_numpy()
    pts = pts[~np.isnan(pts).any(axis=1)]
    if pts.shape[0] == 0:
        return pd.DataFrame() if as_dataframe else {}
    
    cx, cy = compute_team_centroid(df)
    metrics_dict= {
        'compactness': compute_team_compactness(df),
        'width': compute_team_width(df),
        'depth': compute_team_depth(df),
        'block_height': compute_team_block_height(df),
        'line_height': compute_team_line_height(df),
        'team_centroid_x':  cx,
        'team_centroid_y':  cy,
        'team_spread': compute_team_spread(df),
        "n_players_used": int(pts.shape[0])
    }

    if 'ball_x' in df.columns and 'ball_y' in df.columns:
        # asumimos valor único por frame; tomamos el primero
        bx = float(df['ball_x'].iloc[0])
        by = float(df['ball_y'].iloc[0])
        metrics_dict['centroid_ball_dist'] = compute_centroid_ball_distance(df, bx, by)

    if as_dataframe:
        return pd.DataFrame([metrics_dict])
    
    return metrics_dict

def compute_frame_metrics(frame_df: pd.DataFrame, frame_id: int) -> dict:
    """
    Compute enriched metrics for a given frame, including context and possession labels.

    This function wraps `get_team_metrics` and appends:
    - frame identifiers and time
    - ball position and ball zone information
    - possession state and additional possession-phase columns (POSSESSION_COLUMNS)

    Parameters
    ----------
    frame_df : pandas.DataFrame
        DataFrame containing player-level rows for a single team in a single frame.
        Expected to include at least:
        - "match_time_s", "period_id"
        - "ball_x", "ball_y"
        - "ball_zone_x", "ball_zone_y", "ball_zone_label"
        - "in_possession"
        - player coordinates "x", "y"
    frame_id : int
        Frame identifier to attach to the output.

    Returns
    -------
    dict
        Dictionary containing team metrics plus contextual fields for that frame.

    Notes
    -----
    - Assumes ball-related values are constant within the frame; the first row
      is used to read them.
    - `POSSESSION_COLUMNS` are added only if present in `frame_df`.
    """
    fr = frame_df.iloc[0]
    time = float(fr["match_time_s"])
    ball_x = float(fr["ball_x"])
    ball_y = float(fr["ball_y"])

    frame_metrics = get_team_metrics(frame_df,False)

    #time
    frame_metrics["frame_id"] = int(frame_id)
    frame_metrics["time"] = time
    frame_metrics["period_id"] = fr["period_id"]

    #ball
    frame_metrics["ball_x"] = ball_x
    frame_metrics["ball_y"] = ball_y
    frame_metrics["ball_zone_x"] = int(fr["ball_zone_x"])
    frame_metrics["ball_zone_y"] = int(fr["ball_zone_y"])
    frame_metrics["ball_zone_label"] = str(fr["ball_zone_label"])

    #possession
    frame_metrics["in_possession"] = bool(fr["in_possession"])

    for col in POSSESSION_COLUMNS:
        if col in fr:
            frame_metrics[col] = fr[col]
    return frame_metrics

def compute_match_metrics_by_frame(df: pd.DataFrame, sort: bool =True, sort_column: str ="time") -> pd.DataFrame:
    """
    Compute frame-level team metrics across a match.

    Groups the input player-level tracking data by `frame_id`, computes metrics
    for each frame using `compute_frame_metrics`, and returns a frame-level
    DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Player-level tracking data for a single team across many frames.
        Must include a "frame_id" column and all fields required by
        `compute_frame_metrics`.
    sort : bool, default=True
        If True, sort the resulting DataFrame by `sort_column`.
    sort_column : str, default="time"
        Column name to sort by when `sort=True` (commonly "time").

    Returns
    -------
    pandas.DataFrame
        Frame-level DataFrame where each row corresponds to one frame and
        includes team metrics, ball context, and possession labels.

    Notes
    -----
    - This function assumes `df` corresponds to a single team. If multiple teams
      are present, compute separately per team before calling.
    """
    rows = []
    for fid, grp in df.groupby('frame_id', sort=False):
            rows.append(compute_frame_metrics(grp, fid))
    
    df_match_metrics = pd.DataFrame(rows)
    
    if sort: 
        df_match_metrics = df_match_metrics.sort_values(sort_column).reset_index(drop=True)

    return df_match_metrics

def compute_zone_metrics(df_frame_metrics: pd.DataFrame, metric_cols: list = None) -> pd.DataFrame:
    """
    Aggregate frame-level metrics by ball zone and possession state.

    Groups `df_frame_metrics` by:
    - ball_zone_x, ball_zone_y, ball_zone_label
    - in_possession

    and computes:
    - n_frames: number of frames in each group
    - mean value for each metric in `metric_cols`

    Parameters
    ----------
    df_frame_metrics : pandas.DataFrame
        Frame-level DataFrame (output of `compute_match_metrics_by_frame`).
        Must include ball zone columns and `in_possession`.
    metric_cols : list of str, optional
        List of metric column names to aggregate. If None, a default set of
        common shape metrics is used.

    Returns
    -------
    pandas.DataFrame
        Aggregated DataFrame with columns:
        - ball_zone_x, ball_zone_y, ball_zone_label, in_possession
        - n_frames
        - one column per metric in `metric_cols` (mean)

    Notes
    -----
    - The ball zones are assumed to be precomputed in preprocessing (e.g., a 3x3 grid).
    - Metrics are aggregated using mean; change `agg_dict` if you want medians or
      dispersion statistics.
    """
    df = df_frame_metrics.copy()

    if metric_cols is None:
        metric_cols = [
            "compactness",
            "width",
            "depth",
            "block_height",
            "line_height",
            "team_centroid_x",
            "team_centroid_y",
            "team_spread",
        ]

    agg_dict = {col: "mean" for col in metric_cols}
    agg_dict["frame_id"] = "count"

    zone_metrics = (
        df.groupby(["ball_zone_x", "ball_zone_y", "ball_zone_label", "in_possession"])
          .agg(agg_dict)
          .reset_index()
          .rename(columns={"frame_id": "n_frames"})
    )

    return zone_metrics

def build_metrics_by_zone_possession(zone_metrics: pd.DataFrame, metric_cols: list = None) -> dict:
    """
    Build a dictionary of aggregated metrics indexed by (zone_x, zone_y, possession).

    Converts the tabular output of `compute_zone_metrics` into a convenient mapping:

        metrics_map[(zone_x, zone_y, "in"|"out")] = {
            "zone_label": str,
            "in_possession": bool,
            "n_frames": int,
            <metric1>: float,
            <metric2>: float,
            ...
        }

    Parameters
    ----------
    zone_metrics : pandas.DataFrame
        Aggregated DataFrame produced by `compute_zone_metrics`, containing one row
        per (ball_zone_x, ball_zone_y, ball_zone_label, in_possession) combination.
    metric_cols : list of str, optional
        Metric columns to include in the dictionary. If None, uses the same
        default list as `compute_zone_metrics`.

    Returns
    -------
    dict
        Dictionary keyed by (zone_x, zone_y, possession_str) where possession_str
        is "in" if `in_possession=True` else "out".

    Notes
    -----
    - This is primarily a convenience structure for plotting heatmaps or
      retrieving per-zone values without further grouping.
    - All metric values are cast to float and `n_frames` to int.
    """
    if metric_cols is None:
        metric_cols = [
            "compactness",
            "width",
            "depth",
            "block_height",
            "line_height",
            "team_centroid_x",
            "team_centroid_y",
            "team_spread",
        ]

    metrics_map = {}

    # zone_metrics viene de compute_zone_metrics, así que
    # ya está agrupado por zona + in_possession (una fila por combinación)
    for _, row in zone_metrics.iterrows():
        zx = int(row["ball_zone_x"])
        zy = int(row["ball_zone_y"])
        in_pos = bool(row["in_possession"])
        key = (zx, zy, "in" if in_pos else "out")

        info = {
            "zone_label": row["ball_zone_label"],
            "in_possession": in_pos,
            "n_frames": int(row["n_frames"]),
        }

        for m in metric_cols:
            info[m] = float(row[m])

        metrics_map[key] = info

    return metrics_map


