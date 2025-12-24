# app/sidebar.py
import streamlit as st
from services.data_loader import load_available_matches
from src.metrics import METRIC_UNITS

def render_sidebar_match_team_selection():
    st.sidebar.header("Setup")
     # 1) Cargar índice de partidos desde tu función
    matches = load_available_matches()


    match_labels = []
    label_to_id = {}

    for _, row in matches.iterrows():
        date_str = str(row.get("date_time", ""))[:10]  # yyyy-mm-dd
        home_team = row['home_team']['short_name']
        away_team = row['away_team']['short_name']
        label = f'{home_team} vs {away_team} – {date_str}'
        match_labels.append(label)
        label_to_id[label] = row["id"]

    # 2) Select de partidos (el usuario ve los nombres reales)
    selected_label = st.sidebar.selectbox(
        "Match",
        match_labels,
        index=None,
        placeholder="Select a match"
        )
    
    if selected_label:
        match_id = label_to_id[selected_label] if selected_label != None else None

        # 3) Buscar la fila del partido elegido
        match_row = matches[matches["id"] == match_id].iloc[0] if selected_label != None else None

        # 4) Equipos
        teams = {
            "home": match_row["home_team"],
            "away": match_row["away_team"],
        }

        team_key = st.sidebar.selectbox(
            "Team",
            list(teams.keys()),
            index=None,
            placeholder='Select a Team',
            format_func=lambda k: teams[k]['short_name'] if k is not None else ""
        )

        if team_key:
            is_home_team = team_key == 'home'
        
            return dict(
                match_id=match_id,
                team_id=teams[team_key]['id'] if team_key is not None else None,
                team_name=teams[team_key]['short_name'] if team_key is not None else "",
                is_home_team = is_home_team
            )
        
    return None

def render_sidebar_filters():
    st.sidebar.header("Filters")

    # exclude_goalkeeper = st.sidebar.checkbox(
    #     "Exclude GK",
    #     value=True,
    # )

    show_metrics = st.sidebar.checkbox(
        "Show Metrics on Pitch",
        value=True,
    )

    show_metrics = [None, "width", "depth", "block_height","line_height", "team_centroid" ]
    primary_metric = st.sidebar.selectbox(
        "Primary metric",
        options=show_metrics,
        index=0,
        format_func=lambda k: k.replace("_", " ").capitalize() if k != None else "None",
        key=f"primary_metric",
    )

    zones = render_zone_selector()

    

    return dict(
        # exclude_goalkeeper=exclude_goalkeeper,
        zones = zones,
        show_metrics = show_metrics,
        primary_metric = primary_metric
    )
        
    return None

# ZONES_GRID = [
#     ["1L", "2L", "3L"],
#     ["1C", "2C", "3C"],
#     ["1R", "2R", "3R"],
# ]
ZONES_GRID = [
    ["3L", "3C", "3R"],
    ["2L", "2C", "2R"],
    ["1L", "1C", "1R"],
]
def toggle_zone(zone_code):
    """Callback que se ejecuta en cada on_click."""
    selected = set(st.session_state.get("zones_selected", []))
    if zone_code in selected:
        selected.remove(zone_code)
    else:
        selected.add(zone_code)
    st.session_state["zones_selected"] = list(selected)
    st.session_state['all_selected'] = len(selected) == 9

def toggle_all_zones():
    """Selecciona o deselecciona TODAS las zonas."""
    current = set(st.session_state.get("zones_selected", []))
    if len(current) == 9:
        # Si ya están todas → deseleccionar todas
        st.session_state["zones_selected"] = []
        st.session_state['all_selected'] = False
    else:
        # Si hay algunas o ninguna → seleccionar todas
        all_zones = [z for row in ZONES_GRID for z in row]
        st.session_state["zones_selected"] = all_zones
        st.session_state['all_selected'] = True


def render_zone_selector():
    # Inicializar estado
    if "zones_selected" not in st.session_state:
        st.session_state["zones_selected"] = ["1L", "1C", "1R","2L", "2C", "2R","3L", "3C", "3R"]
        st.session_state['all_selected'] = True

    selected = set(st.session_state["zones_selected"])

    title_col, button_col = st.sidebar.columns([1.5, 1])
    with title_col:
        st.markdown("### Zones")

    with button_col:
        # st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
        st.button(
            "Deselect All" if st.session_state['all_selected'] else "Select All",
            key="toggle_all",
            on_click=toggle_all_zones,
            use_container_width=True,
            type='tertiary',
            width="stretch",
            
        )

    # Grilla 3×3
    for row in ZONES_GRID:
        cols = st.sidebar.columns(3, gap="small")
        for col, z in zip(cols, row):
            with col:
                is_selected = z in selected

                st.button(
                    z,
                    key=f"zone_btn_{z}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                    on_click=toggle_zone,
                    args=(z,),
                )

    
    return list(st.session_state["zones_selected"])