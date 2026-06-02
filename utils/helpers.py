import streamlit as st


def bi_icon(name: str, color: str = "#040404", size: str = "1rem") -> str:
    return f'<i class="bi bi-{name}" style="color:{color};font-size:{size};line-height:1;vertical-align:middle;"></i>'


def metric_card(label, value, delta=None, delta_neg=False):
    delta_html = ""
    if delta:
        cls = "neg" if delta_neg else ""
        delta_html = f'<div class="delta {cls}">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title, subtitle=""):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def insight_box(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


def badge(text, color="indigo"):
    st.markdown(f'<span class="badge badge-{color}">{text}</span>', unsafe_allow_html=True)
