# app/tabs_container.py
import streamlit as st
from src import visualizations as viz
from src import shape as sh
# from app.tabs_phase import render_phase_tab  # lo creamos abajo


def render_tabs(df_tracking_team, filters: dict):
    """
    Crea las dos tabs y delega la lógica a render_phase_tab.
    """
    tab_out, tab_in = st.tabs(["Out of possession", "In possession"])

    with tab_out:
        render_phase_tab(
            df_tracking_team=df_tracking_team,
            filters=filters,
            in_possession=False,
            title_suffix="(Out of possession)",
        )

    with tab_in:
        render_phase_tab(
            df_tracking_team=df_tracking_team,
            filters=filters,
            in_possession=True,
            title_suffix="(In possession)",
        )

def render_phase_tab(df_tracking_team, filters: dict, in_possession: bool, title_suffix: str):
    """
    Renderiza el contenido de una tab (in_possession = True/False).
    """
    df = df_tracking_team[
        df_tracking_team["in_possession"] == in_possession
    ]

    col_pitch, col_timelines = st.columns([2, 1])

    with col_pitch:
        shape = sh.compute_average_team_shape_segment(df, top_n_players=10, include_metrics=True)
        if not shape:
            st.warning("No average team shape data available for this phase.")
            return
        _,fig, _ = viz.plot_team_shape(shape, show_metrics=filters['show_metrics'], metric_column='width', vertical_pitch=False, in_possession=in_possession)
        # fig, ax = viz.plot_team_shape(shape)
        st.pyplot(fig)
    with col_timelines:
        st.write("Timelines placeholder")