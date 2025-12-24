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
    Expand phases-of-play intervals into a frame-level table.

    Each row in `phases_df` defines a possession/phase interval via `frame_start`
    and `frame_end`. This function expands every interval to all integer frame
    ids in the inclusive range [frame_start, frame_end], producing a DataFrame
    with one row per `frame_id` and the phase/possession metadata repeated for
    each frame.

    Parameters
    ----------
    phases_df : pandas.DataFrame
        DataFrame containing phase/possession segments. Must include at least
        `frame_start` and `frame_end` and the columns referenced in `cols_keep`
        after renaming. The function renames several columns to standardized
        `possession_*` names.

    Returns
    -------
    pandas.DataFrame
        Frame-level DataFrame with exactly one row per `frame_id`, sorted by
        `frame_id`. If multiple phase rows cover the same frame, duplicates are
        removed keeping the first occurrence (after concatenation).

    Notes
    -----
    - The output is suitable for a many-to-one merge into tracking data
      (`tracking_df` has many player rows per frame, while this has one).
    - Duplicate handling: `drop_duplicates(subset=["frame_id"])` is applied.
      If overlaps exist, the retained row depends on the concatenation order.
    - `frame_start` and `frame_end` are treated as inclusive bounds.

    Raises
    ------
    KeyError
        If required columns are missing from `phases_df` (before or after renaming).
    ValueError
        If `phases_df` is empty and `pd.concat(rows, ...)` is called with no rows.

    Examples
    --------
    >>> phases_per_frame = expand_phases_to_frames(phases_df)
    >>> phases_per_frame.head()
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
        frames = np.arange(int(r.frame_start), int(r.frame_end), dtype=int)
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
    Merge frame-level phases-of-play information into tracking data.

    Performs a left join on `frame_id` to attach the (single-row-per-frame)
    phase/possession metadata from `phases_per_frame` to the player-level
    tracking table.

    Parameters
    ----------
    tracking_df : pandas.DataFrame
        Player-level tracking DataFrame. Must contain a `frame_id` column.
        Typically contains multiple rows per frame (one per player).
    phases_per_frame : pandas.DataFrame
        Frame-level DataFrame produced by `expand_phases_to_frames`, with one
        row per `frame_id`.

    Returns
    -------
    pandas.DataFrame
        Tracking DataFrame enriched with phases-of-play columns. The number of
        rows equals `len(tracking_df)` (left join).

    Notes
    -----
    - Uses `validate="many_to_one"` to ensure `phases_per_frame` has at most one
      row per `frame_id`. If that condition is violated, pandas raises an error.
    """
    merged = tracking_df.merge(
        phases_per_frame,
        on="frame_id",
        how="left",
        validate="many_to_one", 
    )
    return merged

def invert_third(third: str) -> str:
    """
    Convert an opponent-relative third label into a team-relative label.

    This helper swaps defensive and attacking thirds and keeps the middle third
    unchanged. It is useful when a possession start/end third is defined from
    the perspective of the team in possession, and you want the equivalent
    label from the perspective of a specific team.

    Parameters
    ----------
    third : str
        Third label. Expected values are:
        - "defensive_third"
        - "middle_third"
        - "attacking_third"
        Can also be NaN.

    Returns
    -------
    str
        Inverted third label:
        - defensive_third -> attacking_third
        - attacking_third -> defensive_third
        - middle_third -> middle_third
        Returns the input unchanged if it is NaN or an unknown value.
    """
    if pd.isna(third):
        return third
    if third == "defensive_third":
        return "attacking_third"
    if third == "attacking_third":
        return "defensive_third"
    return "middle_third"

def invert_channel(channel: str) -> str:
    """
    Convert an opponent-relative channel label into a team-relative label.

    This helper swaps left/right channels and keeps central unchanged. It is
    useful when channel labels are defined from the perspective of the team in
    possession, and you want the equivalent label from the perspective of a
    specific team.

    Parameters
    ----------
    channel : str
        Channel label. Expected values include:
        - "wide_left", "wide_right"
        - "half_space_left", "half_space_right"
        - "central"
        Can also be NaN.

    Returns
    -------
    str
        Inverted channel label:
        - wide_left <-> wide_right
        - half_space_left <-> half_space_right
        - central -> central
        Returns the input unchanged if it is NaN or an unknown value.
    """
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
    Add team-relative phases-of-play and possession-context columns.

    The phases-of-play data is typically defined relative to the team in
    possession (e.g., `team_in_possession_phase_type`). This function adds
    derived columns that are relative to `my_team_id`, so that "team phase" and
    "opponent phase" are consistent regardless of whether the team is in or out
    of possession in a given frame.

    It also derives team-relative possession start/end locations (third/channel)
    and several boolean flags describing turnovers and penalty-area context.

    Parameters
    ----------
    df : pandas.DataFrame
        Tracking (or tracking+phases) DataFrame containing, at minimum:
        - `team_in_possession_id`
        - `team_in_possession_phase_type`, `team_out_of_possession_phase_type`
        - `possession_third_start`, `possession_third_end`
        - `possession_channel_start`, `possession_channel_end`
        - `possession_penalty_area_start`, `possession_penalty_area_end`
        - `team_possession_loss_in_phase`
        Typically this is the output of `merge_phases_into_tracking`.
    my_team_id : int
        Team identifier for the team of interest.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with additional columns, including:
        - `team_in_possession` (bool)
        - `team_phase_type`, `opponent_phase_type`
        - `possession_start_team_third`, `possession_end_team_third`
        - `possession_start_team_channel`, `possession_end_team_channel`
        - `team_loss_in_possession`, `team_recovery_in_possession`
        - `possession_ends_in_opponent_box`, `possession_ends_in_team_box`
        - `possession_starts_in_opponent_box`, `possession_starts_in_team_box`

    Notes
    -----
    - Third/channel inversion is applied when `my_team_id` is NOT the team in
      possession for the frame, so that the start/end locations remain
      team-relative.
    - `team_possession_loss_in_phase` is interpreted as "the team in possession
      lost the ball during the phase". Therefore:
        - `team_loss_in_possession` is True when `my_team_id` was in possession
          and loss occurred.
        - `team_recovery_in_possession` is True when `my_team_id` was out of
          possession and the in-possession team lost the ball (i.e., my team
          recovered possession).
    """
    df = df.copy()

    
    df["in_possession"] = df["team_in_possession_id"] == my_team_id
    

    # print(df["team_in_possession"] == df["in_possession"])

    df["team_phase_type"] = np.where(
        df["in_possession"],
        df["team_in_possession_phase_type"],
        df["team_out_of_possession_phase_type"],
    )

    df["opponent_phase_type"] = np.where(
        df["in_possession"],
        df["team_out_of_possession_phase_type"],
        df["team_in_possession_phase_type"],
    )


    df["possession_start_team_third"] = np.where(
        df["in_possession"],
        df["possession_third_start"],
        df["possession_third_start"].map(invert_third),
    )

    df["possession_end_team_third"] = np.where(
        df["in_possession"],
        df["possession_third_end"],
        df["possession_third_end"].map(invert_third),
    )

    df["possession_start_team_channel"] = np.where(
        df["in_possession"],
        df["possession_channel_start"],
        df["possession_channel_start"].map(invert_channel)
    )

    df["possession_end_team_channel"] = np.where(
        df["in_possession"],
        df["possession_channel_end"],
        df["possession_channel_end"].map(invert_channel)
    )

    df["team_loss_in_possession"] = (
        df["in_possession"] &
        df["team_possession_loss_in_phase"]
    )

    df["team_recovery_in_possession"] = (
        (~df["in_possession"]) &
        df["team_possession_loss_in_phase"]
    )

    df["possession_ends_in_opponent_box"] = (
        df["in_possession"] & df["possession_penalty_area_end"]
    )

    df["possession_ends_in_team_box"] = (
        (~df["in_possession"]) & df["possession_penalty_area_end"]
    )

    df["possession_starts_in_opponent_box"] = (
        df["in_possession"] & df["possession_penalty_area_start"]
    )

    df["possession_starts_in_team_box"] = (
        (~df["in_possession"]) & df["possession_penalty_area_start"]
    )

    return df
