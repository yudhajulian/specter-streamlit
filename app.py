import streamlit as st

st.set_page_config(
    page_title="Spectre — Anti-Spoofing Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.sidebar import load_css, render_sidebar
from views import overview, eda, preprocessing, modeling, evaluation

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
