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

def get_convex_hull(df) -> ConvexHull:
    points = df[['x', 'y']].values
    return ConvexHull(points)
    
def compute_team_compactness(df) -> float:
    pts = df[["x","y"]].to_numpy()
    pts = pts[~np.isnan(pts).any(axis=1)]
    if pts.shape[0] < 3:
        return np.nan
    try:
        return ConvexHull(pts).volume
    except QhullError:
        return np.nan

def compute_team_width(df) -> float:
    top_y = df.nlargest(1, 'y')
    less_y = df.nsmallest(1, 'y')

    min_y = less_y.y.values[0]
    max_y = top_y.y.values[0]
    return max_y - min_y

def compute_team_depth(df) -> float:
    top_x = df.nlargest(1, 'x')
    less_x = df.nsmallest(1, 'x')

    min_x = less_x.x.values[0]
    max_x = top_x.x.values[0]

    return max_x - min_x

def compute_team_block_height(df, percentile=80) -> float:
    return float(np.percentile(df["x"], percentile))

def compute_team_line_height(df, percentile=20) -> float:
    return float(np.percentile(df["x"], percentile))

def compute_team_centroid(df):
    """
    Centroide del equipo en ESTE FRAME (promedio de x e y).
    Retorna (cx, cy) en metros.
    """
    # sup: df tiene solo jugadores de campo del equipo en ese frame
    cx = df['x'].mean()
    cy = df['y'].mean()
    return float(cx), float(cy)

def compute_team_spread(df):
    """
    Dispersión respecto al centroide en ESTE FRAME.
    - squared=True  -> devuelve MSD  (m^2)
    - squared=False -> devuelve RMSD (m)
    """
    cx, cy = compute_team_centroid(df)
    dx = df['x'].to_numpy() - cx
    dy = df['y'].to_numpy() - cy
    msd = np.mean(dx*dx + dy*dy)  # m^2
    return float(np.sqrt(msd))

def compute_centroid_ball_distance(df, ball_x: float, ball_y: float):
    """
    Distancia entre el centroide del equipo y el balón en ESTE FRAME.
    """
    cx, cy = compute_team_centroid(df)
    dx = cx - ball_x
    dy = cy - ball_y
    d2 = dx*dx + dy*dy
    return float(np.sqrt(d2))

def get_team_metrics(df, as_dataframe = True):
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

    # if 'in_possession' in df.columns:
    #     in_possession = df['in_possession'].iloc[0]
    #     metrics_dict['in_possession'] = bool(in_possession)
    
    # if 'ball_zone_x' in df.columns:
    #     ball_zone_x = df['ball_zone_x'].iloc[0]
    #     metrics_dict['ball_zone_x'] = str(ball_zone_x)
    #     ball_zone_y = df['ball_zone_y'].iloc[0]
    #     metrics_dict['ball_zone_y'] = str(ball_zone_y)
    #     ball_zone_label = df['ball_zone_label'].iloc[0]
    #     metrics_dict['ball_zone_label'] = str(ball_zone_label)
    
    if as_dataframe:
        return pd.DataFrame([metrics_dict])
    
    return metrics_dict

def compute_frame_metrics(frame_df, frame_id):
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

def compute_match_metrics_by_frame(df, sort=True, sort_column="time"):
    
    rows = []
    for fid, grp in df.groupby('frame_id', sort=False):
            rows.append(compute_frame_metrics(grp, fid))
    
    df_match_metrics = pd.DataFrame(rows)
    
    if sort: 
        df_match_metrics = df_match_metrics.sort_values(sort_column).reset_index(drop=True)

    return df_match_metrics

def compute_zone_metrics(df_frame_metrics: pd.DataFrame, metric_cols=None) -> pd.DataFrame:
    """
    Agrupa métricas por zona del balón e in_possession.
    
    Devuelve un DataFrame con:
      ball_zone_x, ball_zone_y, ball_zone_label, in_possession,
      n_frames, y las métricas agregadas (mean).
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

def build_metrics_by_zone_possession(
    zone_metrics: pd.DataFrame,
    metric_cols=None
) -> dict:
    """
    Construye un diccionario con las métricas agregadas por zona y posesión,
    similar a la estructura de `shapes`.

    Devuelve:
      metrics_map[(zone_x, zone_y, "in"|"out")] = {
          "zone_label": str,
          "in_possession": bool,
          "n_frames": int,
          <métrica1>: float,
          <métrica2>: float,
          ...
      }
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


