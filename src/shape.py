from . import metrics as metrics
import pandas as pd

def compute_average_team_position(df, top_n_players=10):
    # df_tracking YA tiene zone_x, zone_y, ball_zone
    df_team = df.copy()

    # 1) elegir top10 jugadores de campo
    player_counts = (
        df_team.groupby("player_id")["frame_id"]
               .nunique()
               .sort_values(ascending=False)
    )
    top_players = player_counts.head(top_n_players).index
    df_team = df_team[df_team["player_id"].isin(top_players)]

    # 2) agrupar por zona + posesión
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

def compute_average_team_shape(df, top_n_players=10):
    df_avg = compute_average_team_position(df, top_n_players)
    if df_avg.empty:
        return {}
    
    shapes = {}

    # agrupamos por zona + posesión sobre el DF de promedios
    for (zone_x, zone_y, in_pos), grp in df_avg.groupby(["ball_zone_x", "ball_zone_y", "in_possession"]):
        # grp: filas = jugadores (promedio en esa zona/posesión)
        hull = metrics.get_convex_hull(grp)   

        # ball_x_mean / ball_y_mean son constantes dentro del grupo
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
                                       top_n_players: int = 10, include_metrics: bool = False):
    """
    Calcula el shape promedio del equipo para el subconjunto de frames
    dado en df (ya filtrado por equipo, fase, zona, etc.).

    df debe tener columnas:
      - player_id, x, y
      - ball_x, ball_y (opcional pero recomendable)
    """
    if df.empty:
        return None

    df_seg = df.dropna(subset=["x","y"]).copy()

    # 1) elegir los jugadores de campo
    player_counts = (
        df_seg.groupby("player_id")["frame_id"]
              .nunique()
              .sort_values(ascending=False)
    )
    top_players = player_counts.head(top_n_players).index
    df_seg = df_seg[df_seg["player_id"].isin(top_players)]

    if df_seg["player_id"].nunique() < 3:
        return None


    # 2) posición promedio de cada jugador
    avg_players = (
        df_seg.groupby("player_id", as_index=False)[["x","y"]]
              .mean()
    )

    # 3) posición promedio de la pelota
    ball_x_mean = ball_y_mean = None
    if {"ball_x","ball_y","frame_id"}.issubset(df_seg.columns):
        ball_per_frame = df_seg.groupby("frame_id")[["ball_x","ball_y"]].first()
        ball_mean = ball_per_frame.median()  # o mean()
        ball_x_mean = float(ball_mean["ball_x"])
        ball_y_mean = float(ball_mean["ball_y"])

    # 4) convex hull sobre las posiciones promedio
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

    