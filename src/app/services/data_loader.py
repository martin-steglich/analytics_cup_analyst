# app/services/data_loader.py
import streamlit as st
from src import preprocessing as pre
from src import metrics as met
import pandas as pd

@st.cache_data(show_spinner="Loading available matches...")
def load_available_matches():
    return pre.load_available_matches() 

@st.cache_data(show_spinner="Loading match data...")
def load_match_data(*,match_id:str, team_id:str, is_home_team:bool):
    return pre.prepare_team_tracking(match_id, team_id, is_home_team=is_home_team, 
                          include_phases_of_play = True,
                          include_match_time = True, 
                          include_in_possession = True,  
                          include_ball_zones = True,
                          exclude_goalkeeper=True)

@st.cache_data(show_spinner="Filtering data...")
def exclude_goalkeepers(df:pd.DataFrame) -> pd.DataFrame:
    df = pre.exclude_goalkeepers_for_match(df)
    return df

@st.cache_data(show_spinner="Filtering data...")
def filter_by_in_out_possession(df:pd.DataFrame, in_possession:bool=None) -> pd.DataFrame:
    if in_possession is not None:
        df = df[df["in_possession"] == in_possession].copy()
    return df

@st.cache_data(show_spinner="Filtering data...")
def filter_by_zones(df:pd.DataFrame, zones:list=None) -> pd.DataFrame:
    if zones and len(zones) >0:
        df = df[df["ball_zone_label"].isin(zones)].copy()
    return df

@st.cache_data(show_spinner="Filtering data...")
def filter_by_phase_of_play(df:pd.DataFrame, phases_of_play:list=None) -> pd.DataFrame:
    if phases_of_play and len(phases_of_play) >0:
        df = df[df["team_phase_type"].isin(phases_of_play)].copy()
    return df

@st.cache_data(show_spinner="Filtering data...")
def filter_by_period(df:pd.DataFrame, periods:list=None) -> pd.DataFrame:
    if periods and len(periods) > 0:
        df = df[df["period_id"].isin(periods)].copy()
    return df

@st.cache_data(show_spinner="Computing match metrics...")
def compute_match_metrics(df:pd.DataFrame) -> pd.DataFrame:
    df_metrics = met.compute_match_metrics_by_frame(df, sort=True, sort_column="time")
    return df_metrics

# @st.cache_data(show_spinner="Cargando métricas del partido...")
# def load_match_metrics(match_id: str, team_id: str) -> pd.DataFrame:
#     df_tracking = pre.load_tracking(match_id)
#     df_team = df_tracking[df_tracking.team_id == team_id].copy()
#     df_team = pre.exclude_goalkeepers_for_match(df_team)
#     df_metrics = met.compute_match_metrics_by_frame(df_team, team_id)
#     return df_metrics
