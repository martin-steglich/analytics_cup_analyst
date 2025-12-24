from . import metrics as metrics
import pandas as pd

def compute_average_team_position(df: pd.DataFrame, top_n_players: int = 10) -> pd.DataFrame:
    """
    Compute average player positions by ball zone and possession state.

    For each ball zone and possession state, this function:
    1) Selects the `top_n_players` most frequently appearing outfield players.
    2) Computes the mean (x, y) position of each selected player.
    3) Computes the mean ball position for the same subset of frames.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format tracking data already filtered to a single team.
        Must include at least the following columns:
        - frame_id, player_id
        - x, y
        - ball_x, ball_y
        - ball_zone_x, ball_zone_y, ball_zone_label
        - in_possession
    top_n_players : int, default=10
        Number of outfield players to include, selected by number of frames
        played (most frequent players).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per (player_id, ball_zone, possession)
        combination, with:
        - average player positions (x, y)
        - average ball position (ball_x_mean, ball_y_mean)
        - ball zone metadata and possession flag

        Returns an empty DataFrame if no valid groups are found.

    Notes
    -----
    - This function is intended to describe *average team shape* rather than
      instantaneous structure.
    - Player selection by frame count ensures stable shapes and avoids noise
      from substitutes or brief appearances.
    """
    df_team = df.copy()

    
    player_counts = (
        df_team.groupby("player_id")["frame_id"]
               .nunique()
               .sort_values(ascending=False)
    )
    top_players = player_counts.head(top_n_players).index
    df_team = df_team[df_team["player_id"].isin(top_players)]


    results = []
    for (zone_x, zone_y,zone_label, in_pos), grp in df_team.groupby(["ball_zone_x", "ball_zone_y","ball_zone_label", "in_possession"]):
        avg_players = (
            grp.groupby("player_id")[["x", "y"]]
               .mean()
               .reset_index()
        )
        ball_mean = grp[["ball_x", "ball_y"]].mean()

        avg_players["ball_zone_x"] = zone_x
        avg_players["ball_zone_y"] = zone_y
        avg_players["ball_zone_label"] = zone_label
        avg_players["in_possession"] = in_pos
        avg_players["ball_x_mean"] = ball_mean["ball_x"]
        avg_players["ball_y_mean"] = ball_mean["ball_y"]

        results.append(avg_players)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)

def compute_average_team_shape(df: pd.DataFrame, top_n_players: int = 10) -> dict:
    """
    Compute average team shapes by ball zone and possession state.

    This function builds on `compute_average_team_position` to produce an
    average team shape for each combination of ball zone and possession
    (in / out of possession). For each group, it computes a convex hull
    over the averaged player positions.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format tracking data for a single team.
        Must include player positions, ball positions, ball zones, and
        possession state.
    top_n_players : int, default=10
        Number of outfield players to include in the average shape.

    Returns
    -------
    dict
        Dictionary indexed by:
            (ball_zone_x, ball_zone_y, "in" | "out")

        Each value is a dictionary with:
        - zone_label : str
        - in_possession : bool
        - players : pandas.DataFrame with columns [player_id, x, y]
        - ball_x_mean : float
        - ball_y_mean : float
        - hull : scipy.spatial.ConvexHull

        Returns an empty dictionary if no shapes can be computed.

    Notes
    -----
    - Player positions are averaged before computing the convex hull.
    - The resulting hull represents a *mean spatial configuration* rather
      than a physical shape observed at a single frame.
    """
    df_avg = compute_average_team_position(df, top_n_players)
    if df_avg.empty:
        return {}
    
    shapes = {}

    
    for (zone_x, zone_y, in_pos), grp in df_avg.groupby(["ball_zone_x", "ball_zone_y", "in_possession"]):
    
        hull = metrics.get_convex_hull(grp)   

    
        ball_x_mean = grp["ball_x_mean"].iloc[0]
        ball_y_mean = grp["ball_y_mean"].iloc[0]
        zone_label = grp["ball_zone_label"].iloc[0]

        key = (zone_x, zone_y, "in" if in_pos else "out")

        shapes[key] = {
            "zone_label": zone_label,
            "in_possession": in_pos,
            "players": grp[["player_id", "x", "y"]].copy(),
            "ball_x_mean": ball_x_mean,
            "ball_y_mean": ball_y_mean,
            "hull": hull,
        }

    return shapes

def compute_average_team_shape_segment(df: pd.DataFrame,*,
                                       top_n_players: int = 10, include_metrics: bool = False) -> dict:
    """
    Compute the average team shape for a selected subset of frames.

    The input DataFrame is assumed to be pre-filtered (e.g., by team,
    phase of play, ball zone, or possession state). The function:
    1) Selects the most frequently appearing outfield players.
    2) Computes each player's average (x, y) position.
    3) Computes a representative average ball position.
    4) Builds a convex hull over the average player positions.
    5) Optionally computes team shape metrics on the averaged shape.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format tracking data for a single team and a selected
        subset of frames. Must include:
        - frame_id, player_id
        - x, y
        Optional but recommended:
        - ball_x, ball_y
    top_n_players : int, default=10
        Number of outfield players to include, selected by frame count.
    include_metrics : bool, default=False
        If True, compute team shape metrics on the averaged player
        positions and include them in the output.

    Returns
    -------
    dict or None
        Dictionary with:
        - players : pandas.DataFrame of averaged player positions
        - ball_x_mean : float or None
        - ball_y_mean : float or None
        - hull : scipy.spatial.ConvexHull
        - n_frames : int, number of frames used
        - n_players : int, number of players used
        - metrics : dict, optional (only if include_metrics=True)

        Returns None if there are insufficient data or fewer than
        three valid players.

    Notes
    -----
    - The convex hull is computed on *average positions*, not on raw
      frame-level positions.
    - Ball position is aggregated using the median per frame to reduce
      sensitivity to outliers.
    - This function is well suited for segment-level analysis
      (e.g., specific phases, zones, or possessions).
    """
    if df.empty:
        return None

    df_seg = df.dropna(subset=["x","y"]).copy()

    
    player_counts = (
        df_seg.groupby("player_id")["frame_id"]
              .nunique()
              .sort_values(ascending=False)
    )
    top_players = player_counts.head(top_n_players).index
    df_seg = df_seg[df_seg["player_id"].isin(top_players)]

    if df_seg["player_id"].nunique() < 3:
        return None


    
    avg_players = (
        df_seg.groupby("player_id", as_index=False)[["x","y"]]
              .mean()
    )

    
    ball_x_mean = ball_y_mean = None
    if {"ball_x","ball_y","frame_id"}.issubset(df_seg.columns):
        ball_per_frame = df_seg.groupby("frame_id")[["ball_x","ball_y"]].first()
        ball_mean = ball_per_frame.median()  # o mean()
        ball_x_mean = float(ball_mean["ball_x"])
        ball_y_mean = float(ball_mean["ball_y"])

    
    hull = metrics.get_convex_hull(avg_players)

    results = {
        "players": avg_players,
        "ball_x_mean": ball_x_mean,
        "ball_y_mean": ball_y_mean,
        "hull": hull,
        "n_frames": int(df_seg["frame_id"].nunique()),
        "n_players": int(avg_players.shape[0]),
    }

    if include_metrics:
        metrics_dict = metrics.get_team_metrics(avg_players, as_dataframe=False)
        if ball_x_mean is not None:
            metrics_dict["centroid_ball_dist"] = metrics.compute_centroid_ball_distance(
                avg_players, ball_x_mean, ball_y_mean
            )
        results["metrics"] = metrics_dict
    
    return results

    