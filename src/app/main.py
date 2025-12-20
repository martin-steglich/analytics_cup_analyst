# app/app.py
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import streamlit as st

from app.layout.tabs_container import render_tabs
from app.layout.sidebar import render_sidebar_match_team_selection, render_sidebar_filters
from app.services.data_loader import load_match_data, filter_by_zones, compute_match_metrics,exclude_goalkeepers, filter_by_in_out_possession, filter_by_phase_of_play, filter_by_period
from src.visualizations import setup_fonts
# from app.tabs_container import render_tabs


def inject_fonts():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');

        /* Base text */
        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
            font-size: 13px;
        }

        .element-container:has(#button-after) + div button {
            font-size: 10px !important;
            padding: 0px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def init_fonts():
    inject_fonts()
    setup_fonts()

st.set_page_config(
    page_title="Team Shape Analyzer",
    layout="wide",
    page_icon="⚽️"
)

def main():

    st.title("Team Shape Analyzer")

    init_fonts()

    # 1) Sidebar → el usuario elige partido, equipo, métrica, etc.
    config = render_sidebar_match_team_selection()
    

    # # 2) Datos → cargamos métricas según match + team
    if config:
        df = load_match_data(match_id=config['match_id'], 
                             team_id=config['team_id'], 
                             is_home_team=config['is_home_team'],
                             )
        
        df_metrics = compute_match_metrics(df)

        filters = render_sidebar_filters()
        if filters:
            # if filters.get('exclude_goalkeeper', False):
            #     df = exclude_goalkeepers(df)

            df = filter_by_zones(df, zones=filters['zones'])
        
            render_tabs(df, filters)
        
    # df_metrics = load_match_metrics(
    #     match_id=config["match_id"],
    #     team_id=config["team_id"],
    # )

    # # 3) Tabs → defensiva / ofensiva / transiciones
    # render_tabs(
    #     df_metrics=df_metrics,
    #     config=config,
    # )

if __name__ == "__main__":
    main()
