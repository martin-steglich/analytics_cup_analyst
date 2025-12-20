from kloppy import skillcorner
import pandas as pd
import numpy as np
import re
import math
import requests
from . import phases
#TODO: translate docstrings to English

PITCH_WIDTH = 68
PITCH_LENGTH = 105
PITCH_MIN_X = -52.5
PITCH_MAX_X = 52.5
PITCH_MIN_Y = -34
PITCH_MAX_Y = 34
ZONE_X = 3
ZONE_Y = 3

def load_available_matches() -> pd.DataFrame:
    return pd.read_json("https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches.json")

def load_metadata(match_id: int | str) -> dict:
    url = (
        "https://raw.githubusercontent.com/SkillCorner/opendata/"
        f"master/data/matches/{match_id}/{match_id}_match.json"
    )
    resp = requests.get(url)
    resp.raise_for_status()   # para que explote claro si el status no es 200
    return resp.json()

def load_tracking(match_id: str):
    """
    Carga datos de tracking de SkillCorner OpenData para un partido dado.
    Devuelve (dataset_kloppy, metadata_kloppy).
    """
    tracking_data_github_url = (
        f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/"
        f"data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    )
    meta_data_github_url = (
        f"https://raw.githubusercontent.com/SkillCorner/opendata/master/"
        f"data/matches/{match_id}/{match_id}_match.json"
        )

    dataset = skillcorner.load(
        meta_data=meta_data_github_url,
        raw_data=tracking_data_github_url,
        coordinates="skillcorner",
        sample_rate=(1 / 2),  # pasa de 10 fps a 5 fps
        # limit=100,  # útil para debug
    )
    return dataset, dataset.metadata

def load_phases_of_play(match_id: str) -> pd.DataFrame:
    return pd.read_csv(f'https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_phases_of_play.csv')

def convert_to_dataframe(dataset, is_home: bool = True) -> pd.DataFrame:
    """
    Convierte el dataset de kloppy a un DataFrame ancho (una fila por frame).
    Alinea la orientación para que el equipo local ataque siempre hacia la derecha.
    """
    to_orientation = "STATIC_HOME_AWAY" if is_home else "STATIC_AWAY_HOME"
    return dataset.transform(to_orientation=to_orientation).to_df(engine="pandas")

def get_players_info_dataframe(match_id: str, teams=None) -> pd.DataFrame:
    """
    Devuelve un DataFrame con info de jugadores:
    player_id, jersey_no, name, team_id, position (si está disponible).
    """
    if teams is None:
        _, metadata = load_tracking(match_id)
        teams = metadata.teams

    home_team, away_team = teams

    players = {}

    def add_team_players(team_obj):
        for player in team_obj.players:
            player_dict = {
                "player_id": int(player.player_id),
                "jersey_no": player.jersey_no,
                "first_name": player.first_name,
                "last_name": player.last_name,
                "name": player.name,
                "team_id": team_obj.team_id,
            }
            if getattr(player, "starting_position", None):
                if player.starting_position:
                    # .code suele ser 'GK', 'ST', 'RCB', etc.
                    player_dict["position"] = player.starting_position.code
            players[player.player_id] = player_dict

    add_team_players(home_team)
    add_team_players(away_team)

    return pd.DataFrame.from_dict(players, orient="index").reset_index(drop=True)

def to_long_format(df: pd.DataFrame, players_info: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Convierte el tracking ancho en formato largo:
    - una fila por (frame_id, player_id)
    - columnas x, y, d, s del jugador
    - incluye ball_x, ball_y del mismo frame
    - si se pasa players_info, agrega team_id, name, etc.
    """
    player_cols = [c for c in df.columns if re.match(r"^\d+_(x|y|d|s)$", c)]

    df_long = df.melt(
        id_vars=[
            "frame_id",
            "timestamp",
            "period_id",
            "ball_state",
            "ball_owning_team_id",
            "ball_x",
            "ball_y",
        ],
        value_vars=player_cols,
        var_name="player_metric",
        value_name="value",
    )

    df_long[["player_id", "metric"]] = df_long["player_metric"].str.extract(r"(\d+)_(x|y|d|s)")
    df_long["player_id"] = df_long["player_id"].astype(int)

    df_players = df_long.pivot_table(
        index=[
            "frame_id",
            "timestamp",
            "period_id",
            "ball_state",
            "ball_owning_team_id",
            "ball_x",
            "ball_y",
            "player_id",
        ],
        columns="metric",
        values="value",
    ).reset_index()

    # asegurar tipos numéricos
    df_players["x"] = df_players["x"].astype(float)
    df_players["y"] = df_players["y"].astype(float)
    df_players["ball_x"] = df_players["ball_x"].astype(float)
    df_players["ball_y"] = df_players["ball_y"].astype(float)

    # opcionalmente agregar metadata de jugadores
    if players_info is not None:
        df_players = df_players.merge(players_info, on="player_id", how="left")

    return df_players

def get_team_goalkeepers_id(df: pd.DataFrame, threshold: float = 0.05) -> list[int]:
    """
    Heurística para identificar IDs de arqueros.
    1. Primero intenta usar la columna 'position' si existe y es 'GK'.
    2. Si no hay 'GK' explícito, intenta inferirlo:
       - ordena jugadores por x (más cerca de su propio arco)
       - mira quién ocupa esa posición la mayoría del tiempo.
    """
    gk_mask = (df.get("position") == "GK") if "position" in df.columns else pd.Series(False, index=df.index)
    goalkeepers = df[gk_mask]["player_id"].unique().tolist()

    if len(goalkeepers) == 0:
        df_sort = df.sort_values(["frame_id", "x"], ascending=[True, True])
        df_last = df_sort.groupby("frame_id").head(1)
        gk_candidates = df_last.player_id.value_counts(normalize=True)
        goalkeepers = gk_candidates[gk_candidates > threshold].index.tolist()

    return list(map(int, goalkeepers))

def get_goalkeeper_by_frame(
    df: pd.DataFrame,
    gk_ids: list[int],
    fps: int = 5,
    min_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Para cada frame_id, devuelve cuál player_id actuó como arquero estable.
    Suaviza microcambios de 1-2 frames.
    """
    def pick_gk_for_frame(frame_df: pd.DataFrame) -> int:
        in_frame_gks = frame_df[frame_df["player_id"].isin(gk_ids)]
        if len(in_frame_gks) > 0:
            gk_row = in_frame_gks.nsmallest(1, "x")
        else:
            gk_row = frame_df.nsmallest(1, "x")
        return int(gk_row["player_id"].values[0])

    gk_raw = (
        df[["frame_id", "player_id", "x"]]
        .groupby("frame_id", group_keys=False)
        .apply(pick_gk_for_frame)
        .rename("gk_player_id_raw")
        .reset_index()
    )

    min_frames = int(min_seconds * fps)

    gk_raw = gk_raw.sort_values("frame_id").reset_index(drop=True)
    gk_raw["gk_shifted"] = gk_raw["gk_player_id_raw"].shift(1)
    gk_raw["new_segment"] = (gk_raw["gk_player_id_raw"] != gk_raw["gk_shifted"]).astype(int)
    gk_raw["segment_id"] = gk_raw["new_segment"].cumsum()

    seg_info = (
        gk_raw.groupby("segment_id")
        .agg(
            gk_player_id=("gk_player_id_raw", "first"),
            n_frames=("frame_id", "count"),
        )
        .reset_index()
    )

    seg_info["is_noise"] = seg_info["n_frames"] < min_frames

    stable_ids = {}
    for idx, row in seg_info.iterrows():
        seg_id = row["segment_id"]
        if not row["is_noise"]:
            stable_ids[seg_id] = row["gk_player_id"]
        else:
            prev_id = seg_info.loc[idx - 1, "segment_id"] if (idx - 1) in seg_info.index else None
            next_id = seg_info.loc[idx + 1, "segment_id"] if (idx + 1) in seg_info.index else None

            if prev_id is not None and not seg_info.loc[idx - 1, "is_noise"]:
                stable_ids[seg_id] = stable_ids[prev_id]
            elif next_id is not None and not seg_info.loc[idx + 1, "is_noise"]:
                stable_ids[seg_id] = seg_info.loc[idx + 1, "gk_player_id"]
            else:
                stable_ids[seg_id] = row["gk_player_id"]

    gk_raw["gk_player_id_stable"] = gk_raw["segment_id"].map(stable_ids)

    return (
        gk_raw[["frame_id", "gk_player_id_stable"]]
        .rename(columns={"gk_player_id_stable": "gk_player_id"})
    )

def exclude_goalkeepers_for_match(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Devuelve el df sin el arquero (por frame), usando las heurísticas anteriores.
    Útil para calcular bloque defensivo/compactness de campo sin que el GK arruine la profundidad.
    """
    team_gk = get_team_goalkeepers_id(df, threshold=threshold)
    gk_by_frame = get_goalkeeper_by_frame(df, gk_ids=team_gk, fps=5, min_seconds=2.0)

    df_no_gk = (
        df.merge(gk_by_frame, on="frame_id")
          .query("player_id != gk_player_id")
          .drop(columns="gk_player_id")
          .reset_index(drop=True)
    )
    return df_no_gk

def load_tracking_as_long_dataframe(match_id: str, is_home: bool = True):
    """
    Carga todo junto:
    - tracking ancho desde kloppy
    - lo convierte a largo
    - agrega metadata de jugadores
    Devuelve (df_tracking_long, metadata_kloppy)
    """
    dataset, metadata = load_tracking(match_id)
    df_tracking = convert_to_dataframe(dataset, is_home=is_home)
    df_players_info = get_players_info_dataframe(match_id, metadata.teams)
    df_tracking_long = to_long_format(df_tracking, players_info=df_players_info)
    return df_tracking_long, metadata

def add_in_possession_column(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """
    Adds a boolean column 'in_possession' indicating if the specified team has ball possession.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking data containing 'ball_owning_team_id' column.
    team_id : int
        The team ID to check for possession.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with an additional 'in_possession' boolean column.
    """
    df = df.copy()
    df["in_possession"] = df["ball_owning_team_id"] == team_id
    return df

def add_match_time(df: pd.DataFrame, period_col="period_id", timestamp_col="timestamp") -> pd.DataFrame:
    """
    Adds continuous match time columns (seconds and timedelta) combining period and timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking data containing 'period_id' and 'timestamp' columns.
    period_col : str, optional
        Name of the period column (default='period_id').
    timestamp_col : str, optional
        Name of the timestamp column (default='timestamp').

    Returns
    -------
    pd.DataFrame
        Same DataFrame with two new columns:
        - 'match_time_s'  : float seconds since match start
        - 'match_time_td' : pandas Timedelta for readability
    """
    df = df.copy()
    
    # Convert timestamp to seconds (handles Timedelta or string)
    if np.issubdtype(df[timestamp_col].dtype, np.timedelta64):
        ts_seconds = df[timestamp_col].dt.total_seconds()
    else:
        ts_seconds = pd.to_timedelta(df[timestamp_col]).dt.total_seconds()

    # Compute max timestamp per period
    period_offsets = (
        df.groupby(period_col)[timestamp_col]
        .max()
        .pipe(pd.to_timedelta)
        .dt.total_seconds()
        .sort_index()
        .cumsum()
        .shift(fill_value=0)
    )
    # Map offset to each row
    df["match_time_s"] = ts_seconds + df[period_col].map(period_offsets)
    df["match_time_td"] = pd.to_timedelta(df["match_time_s"], unit="s")
    return df

def get_pitch_zone(x, y):
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        raise ValueError("x and y must be numeric")

    zone_length = PITCH_LENGTH / ZONE_X
    zone_width = PITCH_WIDTH / ZONE_Y

    # índice de tercio horizontal (defensivo–medio–ofensivo)
    rel_x = (xf - PITCH_MIN_X) / zone_length
    zx = int(min(max(math.floor(rel_x), 0), ZONE_X - 1))

    # índice de carril bruto (bottom–middle–top en coords SkillCorner)
    rel_y = (yf - PITCH_MIN_Y) / zone_width
    zy_raw = int(min(max(math.floor(rel_y), 0), ZONE_Y - 1))

    # invertir para que 0 sea "arriba" (tu carril izquierdo)
    zy = (ZONE_Y - 1) - zy_raw

    return zx, zy

def get_zone_label(zx,zy):
    third = str(zx + 1)
    channel = ["L", "C", "R"][zy]

    return third + channel

def add_ball_zones(df):
    # df: tracking de un partido, con ball_x, ball_y, frame_id, etc.
    df = df.copy()
    df["ball_zone_x"], df["ball_zone_y"] = zip(*df.apply(
        lambda r: get_pitch_zone(r["ball_x"], r["ball_y"]), axis=1
    ))
    df["ball_zone_label"] = df.apply(
        lambda r: get_zone_label(r["ball_zone_x"], r["ball_zone_y"]), axis=1
    )
    return df

def add_phases_of_play_info(
    df: pd.DataFrame,
    match_id: str,
    team_id: int,
) -> pd.DataFrame:
    """
    Carga phases_of_play y las mergea al tracking frame a frame.
    """
    phases_df = load_phases_of_play(match_id)
    phases_per_frame = phases.expand_phases_to_frames(phases_df)
    df_merged = phases.merge_phases_into_tracking(df, phases_per_frame)
       
    return phases.add_team_phase_of_play_info(df_merged, team_id)

def prepare_team_tracking(match_id, team_id, is_home_team,*, 
                          include_phases_of_play = True,
                          include_match_time = True, 
                          include_in_possession = True,  
                          include_ball_zones = True,
                          exclude_goalkeeper=True):
    df, _ = load_tracking_as_long_dataframe(match_id, is_home_team)
    df = df[df["team_id"] == team_id]

    if include_phases_of_play:
        df = add_phases_of_play_info(df, match_id, team_id)

    if include_match_time:
        df = add_match_time(df)

    if include_in_possession:
        df = add_in_possession_column(df, team_id)
    
    if include_ball_zones:
        df = add_ball_zones(df)

    if exclude_goalkeeper:
        df = exclude_goalkeepers_for_match(df)
        
    return df

def prepare_team_tracking_from_raw_data(df_tracking_raw, team_id,*, 
                          include_match_time = True, 
                          include_in_possession = True,  
                          include_ball_zones = True,
                          exclude_goalkeeper=True):
    df = df_tracking_raw.copy()
    df = df[df["team_id"] == team_id]

    if include_match_time:
        df = add_match_time(df)

    if include_in_possession:
        df = add_in_possession_column(df, team_id)
    
    if include_ball_zones:
        df = add_ball_zones(df)

    if exclude_goalkeeper:
        df = exclude_goalkeepers_for_match(df)
        
    return df

