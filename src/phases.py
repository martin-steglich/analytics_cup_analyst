import pandas as pd
import numpy as np

POSSESSION_COLUMNS  = [
            "possession_id",
            "possession_duration_s",
            "team_phase_type",
            "opponent_phase_type",
            'possession_start_team_third',
            'possession_end_team_third', 
            'possession_start_team_channel', 
            'possession_end_team_channel',
            "possession_lead_to_shot",
            "possession_lead_to_goal",
            "team_loss_in_possession",
            "team_recovery_in_possession",
            "possession_ends_in_opponent_box",
            "possession_ends_in_team_box",
            "possession_starts_in_opponent_box",
            "possession_starts_in_team_box",
        ]

def expand_phases_to_frames(phases_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expande cada fase de juego a todos los frames comprendidos entre
    frame_start y frame_end (inclusive).

    Devuelve un DataFrame con UNA fila por frame_id.
    """
    rows = []

    phases_df = phases_df.rename(columns={
        "index":"possession_id",
        "channel_start":"possession_channel_start",
        "duration": "possession_duration_s",
        "third_start": "possession_third_start",
        "penalty_area_start": "possession_penalty_area_start",
        "channel_end": "possession_channel_end",
        "third_end": "possession_third_end",
        "penalty_area_end" : "possession_penalty_area_end",
        "team_possession_lead_to_shot": "possession_lead_to_shot",
        "team_possession_lead_to_goal": "possession_lead_to_goal"
        })

    # Elegí las columnas que querés “arrastrar” a nivel frame
    cols_keep = [
        "possession_id",
        "match_id",
        "frame_start",
        "frame_end",
        "possession_duration_s",
        'possession_channel_start', 
        'possession_third_start',
        'possession_penalty_area_start',
        'possession_channel_end',
        'possession_third_end', 
        'possession_penalty_area_end',
        "team_in_possession_id",
        "team_in_possession_phase_type",
        "team_in_possession_phase_type_id",
        "team_out_of_possession_phase_type",
        "team_out_of_possession_phase_type_id",
        "possession_lead_to_shot",
        "possession_lead_to_goal",
        "team_possession_loss_in_phase",
        "n_player_possessions_in_phase",
    ]

    
    phases_df = phases_df[cols_keep].copy()


    for _, r in phases_df.iterrows():
        frames = np.arange(r.frame_start, r.frame_end + 1, dtype=int)
        base = {col: r[col] for col in cols_keep if col not in ("frame_start", "frame_end")}
        df_phase = pd.DataFrame(base, index=frames)
        df_phase["frame_id"] = frames
        rows.append(df_phase)

    phases_per_frame = pd.concat(rows, ignore_index=True)

    phases_per_frame = phases_per_frame.drop_duplicates(subset=["frame_id"])
    phases_per_frame = phases_per_frame.sort_values("frame_id").reset_index(drop=True)

    return phases_per_frame

def merge_phases_into_tracking(
    tracking_df: pd.DataFrame,
    phases_per_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Hace un left-join por frame_id para añadir info de phases_of_play
    al tracking frame a frame.
    """
    merged = tracking_df.merge(
        phases_per_frame,
        on="frame_id",
        how="left",
        validate="many_to_one", 
    )
    return merged

def invert_third(third: str) -> str:
    if pd.isna(third):
        return third
    if third == "defensive_third":
        return "attacking_third"
    if third == "attacking_third":
        return "defensive_third"
    return "middle_third"

def invert_channel(channel: str) -> str:
    if pd.isna(channel):
        return channel 
    swap = {
        "wide_left": "wide_right",
        "wide_right": "wide_left",
        "half_space_left": "half_space_right",
        "half_space_right": "half_space_left",
        "central": "central",
    }
    return swap.get(channel, channel)

def add_team_phase_of_play_info(
    df: pd.DataFrame,
    my_team_id: int,
) -> pd.DataFrame:
    """
    Añade una columna con la fase relevante para mi equipo:
    - Si mi equipo está en posesión en ese frame -> phase = team_in_possession_phase_type
    - Si no -> phase = team_out_of_possession_phase_type
    """
    df = df.copy()

    
    df["team_in_possession"] = df["team_in_possession_id"] == my_team_id


    df["team_phase_type"] = np.where(
        df["team_in_possession"],
        df["team_in_possession_phase_type"],
        df["team_out_of_possession_phase_type"],
    )

    df["opponent_phase_type"] = np.where(
        df["team_in_possession"],
        df["team_out_of_possession_phase_type"],
        df["team_in_possession_phase_type"],
    )


    df["possession_start_team_third"] = np.where(
        df["team_in_possession"],
        df["possession_third_start"],
        df["possession_third_start"].map(invert_third),
    )

    df["possession_end_team_third"] = np.where(
        df["team_in_possession"],
        df["possession_third_end"],
        df["possession_third_end"].map(invert_third),
    )

    df["possession_start_team_channel"] = np.where(
        df["team_in_possession"],
        df["possession_channel_start"],
        df["possession_channel_start"].map(invert_channel)
    )

    df["possession_end_team_channel"] = np.where(
        df["team_in_possession"],
        df["possession_channel_end"],
        df["possession_channel_end"].map(invert_channel)
    )

    df["team_loss_in_possession"] = (
        df["team_in_possession"] &
        df["team_possession_loss_in_phase"]
    )

    df["team_recovery_in_possession"] = (
        (~df["team_in_possession"]) &
        df["team_possession_loss_in_phase"]
    )

    df["possession_ends_in_opponent_box"] = (
        df["team_in_possession"] & df["possession_penalty_area_end"]
    )

    df["possession_ends_in_team_box"] = (
        (~df["team_in_possession"]) & df["possession_penalty_area_end"]
    )

    df["possession_starts_in_opponent_box"] = (
        df["team_in_possession"] & df["possession_penalty_area_start"]
    )

    df["possession_starts_in_team_box"] = (
        (~df["team_in_possession"]) & df["possession_penalty_area_start"]
    )

    return df
