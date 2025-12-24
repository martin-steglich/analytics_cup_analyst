# app/tabs_container.py
import streamlit as st
from src import visualizations as viz
import src.shape as sh   
import pandas as pd

TRI = [None, True, False]
def tri_label(v):
    if v is None:
        return "Don't care"
    return "Yes" if v else "No"

def filter_by_outcomes(df, lead_to_shot, lead_to_goal):
    out = df

    if lead_to_shot is not None:
        out = out[out["possession_lead_to_shot"] == lead_to_shot]

    if lead_to_goal is not None:
        out = out[out["possession_lead_to_goal"] == lead_to_goal]

    return out

def pretty_phase(k: str) -> str:
    return k.replace("_", " ").capitalize()

def filter_by_outcomes(df, lead_to_shot, lead_to_goal):
    out = df
    if lead_to_shot is not None:
        out = out[out["possession_lead_to_shot"] == lead_to_shot]
    if lead_to_goal is not None:
        out = out[out["possession_lead_to_goal"] == lead_to_goal]
    return out

def build_title(base: str, phase_key, lead_to_shot, lead_to_goal) -> str:
    tags = [base]
    if phase_key is not None:
        tags.append(phase_key.replace("_", " ").title())
    if lead_to_shot is not None:
        tags.append(f"Shot: {tri_label(lead_to_shot)}")
    if lead_to_goal is not None:
        tags.append(f"Goal: {tri_label(lead_to_goal)}")
    return " • ".join(tags)

def render_phase_panel(
    df_tracking: pd.DataFrame,
    phase_values: list[str],
    show_metrics: bool,
    primary_metric: str | None,
    vertical_pitch: bool,
    base_title: str,
    panel_key: str,
    in_pos: bool,  # ✅ nuevo
):
    phase_options = [None] + phase_values  # ✅ All phases

    phase_key = st.selectbox(
        "Phase",
        options=phase_options,
        index=0,
        format_func=lambda k: "All phases" if k is None else pretty_phase(k),
        key=f"phase_{panel_key}",
    )

    lead_to_shot = st.selectbox(
        "Lead to shot",
        options=TRI,
        format_func=tri_label,
        index=0,
        key=f"lead_to_shot_{panel_key}",
    )

    lead_to_goal = st.selectbox(
        "Lead to goal",
        options=TRI,
        format_func=tri_label,
        index=0,
        key=f"lead_to_goal_{panel_key}",
    )

    df = df_tracking
    if phase_key is not None:
        df = df[df["team_phase_type"] == phase_key]

    df = filter_by_outcomes(df, lead_to_shot=lead_to_shot, lead_to_goal=lead_to_goal)

    panel_title = build_title(base_title, phase_key, lead_to_shot, lead_to_goal)

    if df.empty:
        st.warning("No data matches the selected filters.")
        return

    render_side(df, show_metrics, primary_metric, vertical_pitch, panel_title, in_pos=in_pos)

def render_side(df_side, show_metrics: bool, primary_metric: str | None, vertical_pitch: bool, label: str, in_pos: bool):
    st.subheader(label)

    team_shape = sh.compute_average_team_shape_segment(
        df_side,
        top_n_players=10,
        include_metrics=show_metrics,
    )
    if not team_shape:
        st.warning(f"No data available for {label.lower()} with current filters.")
        return

    _, fig, _ = viz.plot_team_shape(
        team_shape,
        show_metrics=show_metrics,
        metric_column=primary_metric,
        vertical_pitch=vertical_pitch,
        in_possession=in_pos,
    )
    st.pyplot(fig, use_container_width=True)


def render_tabs(df_tracking_team, filters: dict):
    """
    
    """
    tab_in_out, tab_in, tab_out = st.tabs(["In vs Out Possession", "In possession - Phase comparison", "Out possession - Phase comparison"])

    with tab_in_out:
        render_in_out_tab(
            df_tracking_team=df_tracking_team,
            filters=filters,
            
            
        )

    with tab_in:
        render_in_tab(
            df_tracking_team=df_tracking_team,
            filters=filters,
            
            
        )
    
    with tab_out:
        render_out_tab(
            df_tracking_team=df_tracking_team,
            filters=filters,
            
            
        )

def render_in_out_tab(df_tracking_team, filters: dict):
    col_in, col_out = st.columns(2)

    top_n = 10
    show_metrics = filters.get("show_metrics", True)
    primary_metric = filters.get("primary_metric", None)
    vertical_pitch = filters.get("vertical_pitch", True)

    with col_in:
        df_in = df_tracking_team[df_tracking_team["in_possession"] == True]
        render_side(df_in, show_metrics, primary_metric, vertical_pitch,  "In possession", in_pos=True)
    with col_out:
        df_out = df_tracking_team[df_tracking_team["in_possession"] == False]
        render_side(df_out, show_metrics, primary_metric, vertical_pitch, "Out of possession", in_pos=False)

def render_in_tab(df_tracking_team, filters: dict):
    col_left, col_right = st.columns(2)

    show_metrics = filters.get("show_metrics", True)
    primary_metric = filters.get("primary_metric", None)
    vertical_pitch = filters.get("vertical_pitch", True)

    df_in = df_tracking_team[df_tracking_team["in_possession"] == True]
    phase_values = sorted(df_in["team_phase_type"].dropna().unique().tolist())

    with col_left:
        render_phase_panel(df_in, phase_values, show_metrics, primary_metric, vertical_pitch,
                        base_title="In Possession", panel_key="in_left", in_pos=True)
    with col_right:
        render_phase_panel(df_in, phase_values, show_metrics, primary_metric, vertical_pitch,
                        base_title="In Possession", panel_key="in_right", in_pos=True)

def render_out_tab(df_tracking_team, filters: dict):
    col_left, col_right = st.columns(2)

    show_metrics = filters.get("show_metrics", True)
    primary_metric = filters.get("primary_metric", None)
    vertical_pitch = filters.get("vertical_pitch", True)

    df_out = df_tracking_team[df_tracking_team["in_possession"] == False]
    phase_values = sorted(df_out["team_phase_type"].dropna().unique().tolist())

    with col_left:
        render_phase_panel(df_out, phase_values, show_metrics, primary_metric, vertical_pitch,
                        base_title="Out of Possession", panel_key="out_left", in_pos=False)
    with col_right:
        render_phase_panel(df_out, phase_values, show_metrics, primary_metric, vertical_pitch,
                        base_title="Out of Possession", panel_key="out_right", in_pos=False)



def render_side(df_side, show_metrics: bool, primary_metric:str, vertical_pitch:bool, label: str, in_pos: bool):
        st.subheader(f"{label}".strip())

        shape = sh.compute_average_team_shape_segment(
            df_side,
            top_n_players=10,
            include_metrics=show_metrics,
        )
        if not shape:
            st.warning(f"No data available for {label.lower()} with current filters.")
            return

        _, fig, _ = viz.plot_team_shape(
            shape,
            show_metrics=show_metrics,
            metric_column=primary_metric,
            vertical_pitch=vertical_pitch,
            in_possession=in_pos,
        )
        st.pyplot(fig, use_container_width=True)



