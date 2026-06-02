import streamlit as st
from PIL import Image as _PILImage
from pathlib import Path as _Path

_icon_path = _Path(__file__).parent / "assets" / "logo_icon_white.png"
_page_icon = _PILImage.open(_icon_path) if _icon_path.exists() else "🛡️"

st.set_page_config(
    page_title="Specter — Analytic",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.sidebar import load_css, render_sidebar
from views import overview, eda, preprocessing, modeling, evaluation

# Read URL query param before render_sidebar() initializes session_state.page
if "page" not in st.session_state:
    param = st.query_params.get("page", "Overview")
    st.session_state.page = param

# Inject CSS from main context so it applies globally
load_css()
render_sidebar()

page = st.session_state.get("page", "Overview")

if page == "Overview":
    overview.render()
elif page == "EDA":
    eda.render()
elif page == "Preprocessing":
    preprocessing.render()
elif page == "Modeling":
    modeling.render()
elif page == "Evaluation":
    evaluation.render()
