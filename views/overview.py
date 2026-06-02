import streamlit as st
import plotly.graph_objects as go
from utils.helpers import section_header, insight_box, bi_icon


def render():
    section_header(
        "SPECTER — Face Anti-Spoofing Dashboard",
        "Face Authentication Platform · Anti-Spoofing API · Coding Camp 2026"
    )

    # ── KPI Cards ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Total Dataset</div>
            <div class="value">11,819</div>
            <div class="delta">Train + Val + Test</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Jumlah Kelas</div>
            <div class="value">6</div>
            <div class="delta">Real + 5 Spoof Type</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Best Val Accuracy</div>
            <div class="value">93.59%</div>
            <div class="delta">↑ IlhamCaesar ResNet50</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Model Utama</div>
            <div class="value" style="font-size:0.95rem;">AntiSpoofNetV4</div>
            <div class="delta">ConvNeXtSmall + FFT + CDC</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Project Pipeline ───────────────────────────────────
    st.markdown('<h3 style="display:flex;align-items:center;gap:8px;">' + bi_icon("signpost-split-fill", "#040404", "1.2rem") + ' Alur Project</h3>', unsafe_allow_html=True)

    pipeline_steps = [
        ("database-fill",  "Data Collection", "Kaggle tf-fas dataset\nFace spoofing images",        "#240CF6"),
        ("search",         "EDA",             "Distribusi kelas\nAnalisis ukuran gambar",            "#06b6d4"),
        ("gear-fill",      "Preprocessing",   "Deduplication\nStratified split 70/15/15",            "#F59E0B"),
        ("cpu-fill",       "Modeling",        "AntiSpoofNetV4 + ResNet50\nHingeACER Focal Loss",     "#8b5cf6"),
        ("graph-up-arrow", "Evaluation",      "Val Acc ~94% · Macro F1 ~90%\nClassification Report",  "#10b981"),
    ]

    cols = st.columns(5)
    for col, (icon, title, desc, color) in zip(cols, pipeline_steps):
        with col:
            st.markdown(f"""
            <div style="
                background:#FFFFFF;
                border:1px solid #E9E9E9;
                border-top:3px solid {color};
                border-radius:12px;
                padding:1rem 0.75rem;
                text-align:center;
                height:140px;
                box-shadow:0 1px 3px rgba(0,0,0,0.04);
            ">
                <div style="margin-bottom:6px;">{bi_icon(icon, color, '1.8rem')}</div>
                <div style="font-weight:700;font-size:0.8rem;color:#040404;margin:6px 0 4px;">{title}</div>
                <div style="font-size:0.7rem;color:#52525B;line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column: Business Understanding + Dataset Split ─
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<h3 style="display:flex;align-items:center;gap:8px;">' + bi_icon("bullseye", "#040404", "1.1rem") + ' Business Understanding</h3>', unsafe_allow_html=True)
        goals = [
            ("Face Liveness Detection API", "SPECTER menyediakan API untuk memverifikasi apakah wajah yang di-capture adalah orang nyata, bukan foto, video, atau topeng."),
            ("Multi-class Anti-Spoofing", "Sistem mampu mendeteksi 5 jenis serangan: mannequin, mask, papercut, printed photo, dan screen replay attack."),
            ("Face Authentication Platform", "Mendukung alur register & authenticate wajah dengan liveness check + identity matching via InsightFace ArcFace."),
        ]
        for title, desc in goals:
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:14px;align-items:flex-start;">
                <div style="width:6px;height:6px;background:#040404;border-radius:50%;margin-top:7px;flex-shrink:0;"></div>
                <div>
                    <div style="font-weight:600;font-size:0.85rem;color:#040404;">{title}</div>
                    <div style="font-size:0.78rem;color:#52525B;line-height:1.5;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<h3 style="display:flex;align-items:center;gap:8px;">' + bi_icon("folder2-open", "#040404", "1.1rem") + ' Pembagian Dataset</h3>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Train (70%)", "Validation (15%)", "Test (15%)"],
            values=[70, 15, 15],
            hole=0.52,
            marker=dict(
                colors=["#240CF6", "#F59E0B", "#10b981"],
                line=dict(color="white", width=2),
            ),
            textinfo="percent",
            textposition="inside",
            insidetextfont=dict(family="Inter", size=13, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>",
            pull=[0.03, 0.03, 0.03],
        ))
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                font=dict(family="Inter", size=12, color="#040404"),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(t=20, b=20, l=20, r=160),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text="<b>Split</b>",
                x=0.42, y=0.5,
                font=dict(size=14, family="Inter", color="#040404"),
                showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        for col, label, val, color in zip(
            [col1, col2, col3],
            ["Train", "Validation", "Test"],
            ["8211", "1760", "1760"],
            ["#040404", "#F59E0B", "#10b981"]
        ):
            with col:
                st.markdown(f"""
                <div style="text-align:center;padding:0.5rem;background:#FAFAFA;border-radius:8px;border-left:3px solid {color};border:1px solid #E9E9E9;">
                    <div style="font-size:0.7rem;color:#A1A1AA;font-weight:600;">{label}</div>
                    <div style="font-size:1rem;font-weight:700;color:{color};">{val}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Insight ────────────────────────────────────────────
    insight_box(
        "Dataset <b>berrymilkyss/tf-fas</b> dari Kaggle digunakan sebagai basis training. Dataset terdiri dari 6 kelas spoofing dengan distribusi tidak seimbang, sehingga diterapkan <b>dynamic oversampling</b> ke 2.500 gambar per kelas untuk training. Hash-based deduplication digunakan untuk menghapus duplikat dan mencegah data leakage antar split train/val/test."
    )

    # ── Spoof types ────────────────────────────────────────
    st.markdown('<h3 style="display:flex;align-items:center;gap:8px;">' + bi_icon("shield-fill-check", "#040404", "1.1rem") + ' Jenis Spoofing yang Dideteksi</h3>', unsafe_allow_html=True)
    spoof_types = [
        ("person-fill",         "fake_mannequin", "Spoof menggunakan mannequin atau boneka wajah.", "#e0e7ff", "#3730a3"),
        ("eye-slash-fill",      "fake_mask",      "Serangan spoofing menggunakan masker wajah palsu.", "#fef3c7", "#92400e"),
        ("scissors",            "fake_papercut",  "Foto wajah dipotong dan dimanipulasi.", "#fee2e2", "#991b1b"),
        ("image-fill",          "fake_printed",   "Printed photo attack menggunakan foto cetak.", "#d1fae5", "#065f46"),
        ("display-fill",        "fake_screen",    "Replay attack menggunakan layar digital.", "#f3e8ff", "#6b21a8"),
        ("person-check-fill",   "realperson",     "Wajah asli pengguna.", "#ecfdf5", "#065f46"),
    ]
    cols = st.columns(3)
    for i, (icon, name, desc, bg, text) in enumerate(spoof_types):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="
                background:{bg};
                border-radius:10px;
                padding:0.85rem 1rem;
                margin-bottom:10px;
                display:flex;
                gap:10px;
                align-items:center;
            ">
                <div style="font-size:1.4rem;flex-shrink:0;">{bi_icon(icon, text, '1.4rem')}</div>
                <div>
                    <div style="font-weight:700;font-size:0.82rem;color:{text};">{name}</div>
                    <div style="font-size:0.72rem;color:#4b5563;line-height:1.4;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
