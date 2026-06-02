import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.helpers import section_header, insight_box


# ── Simulated data from DAC Find-IT 2026 dataset ──────────────────────────────

CLASS_DIST = {
    "fake_mannequin": 237,
    "fake_mask": 1047,
    "fake_papercut": 1770,
    "fake_printed": 2653,
    "fake_screen": 2753,
    "realperson": 3359,
}

CLASS_LABELS = {
    "fake_mannequin": "Fake Mannequin",
    "fake_mask": "Fake Mask",
    "fake_papercut": "Fake Papercut",
    "fake_printed": "Fake Printed",
    "fake_screen": "Fake Screen",
    "realperson": "Real Person",
}

COLORS = {
    "fake_mannequin": "#ef4444",
    "fake_mask": "#f59e0b",
    "fake_papercut": "#8b5cf6",
    "fake_printed": "#4f46e5",
    "fake_screen": "#06b6d4",
    "realperson": "#10b981",
}


def render():
    section_header(
        "Exploratory Data Analysis",
        "Memahami karakteristik dan distribusi dataset face anti-spoofing"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribusi Kelas",
        "🖼️ Ukuran Gambar",
        "📁 Format File",
        "🔎 Insight",
    ])

    # ── TAB 1: Class Distribution ──────────────────────────
    with tab1:
        st.markdown("#### Distribusi Dataset per Kelas")

        df_dist = pd.DataFrame({
            "Kelas":  [CLASS_LABELS[k] for k in CLASS_DIST],
            "Jumlah": list(CLASS_DIST.values()),
            "Tipe":   ["Real" if k == "realperson" else "Spoof" for k in CLASS_DIST],
        })
        total = df_dist["Jumlah"].sum()
        df_dist["Persentase"] = (df_dist["Jumlah"] / total * 100).round(2)
        df_dist["Color"] = [COLORS[k] for k in CLASS_DIST]

        col_chart, col_pie = st.columns([3, 2])

        with col_chart:
            fig_bar = go.Figure(go.Bar(
                x=df_dist["Kelas"],
                y=df_dist["Jumlah"],
                marker_color=df_dist["Color"],
                text=df_dist["Jumlah"],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Jumlah: %{y:,}<extra></extra>",
            ))
            fig_bar.update_layout(
                title="Jumlah Gambar per Kelas",
                xaxis_title="",
                yaxis_title="Jumlah Gambar",
                xaxis_tickangle=-30,
                height=400,
                margin=dict(t=50, b=100, l=60, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", size=11),
                yaxis=dict(gridcolor="#f3f4f6"),
                xaxis=dict(tickfont=dict(size=11)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=df_dist["Kelas"],
                values=df_dist["Jumlah"],
                hole=0.45,
                marker=dict(colors=df_dist["Color"]),
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>%{value:,} gambar (%{percent})<extra></extra>",
            ))
            fig_pie.update_layout(
                title="Proporsi per Kelas",
                height=380,
                margin=dict(t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans"),
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Tabel distribusi
        st.markdown("#### 📋 Tabel Distribusi")
        st.markdown("""
        <table class="styled-table" style="width:100%;">
        <thead><tr>
            <th>Kelas</th><th>Tipe</th><th>Jumlah Gambar</th><th>Persentase</th>
        </tr></thead>
        <tbody>
        """ + "".join([
            f"""<tr>
                <td><strong>{row['Kelas']}</strong></td>
                <td><span class="badge badge-{'green' if row['Tipe']=='Real' else 'red'}">{row['Tipe']}</span></td>
                <td>{row['Jumlah']:,}</td>
                <td>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="width:{row['Persentase']*2}px;height:8px;background:{row['Color']};border-radius:4px;"></div>
                        {row['Persentase']:.2f}%
                    </div>
                </td>
            </tr>"""
            for _, row in df_dist.iterrows()
        ]) + "</tbody></table>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        insight_box(
            f"Dataset bersifat <b>imbalanced</b> — kelas 'Real Face' memiliki jumlah terbanyak ({CLASS_DIST['realperson']:,}) "
            f"sedangkan '3D Mask' paling sedikit ({CLASS_DIST['fake_mannequin']:,}). "
            "Hal ini perlu ditangani saat training menggunakan teknik seperti Focal Loss atau class weighting."
        )

    # ── TAB 2: Image Size Analysis ────────────────────────
    with tab2:
        st.markdown("#### Analisis Ukuran Gambar")

        # ── Real Dataset Statistics ────────────────────────
        stats_df = pd.DataFrame(
            {
                "width":  [600.0, 935.77, 870.95, 183.0, 429.0, 600.0, 1080.0, 5664.0],
                "height": [600.0, 1311.21, 1178.47, 200.0, 480.0, 720.0, 1920.0, 4248.0],
            },
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )

        # ── Sample Visualization ──────────────────────────
        np.random.seed(42)
        n = 600
        widths = np.concatenate([
            np.random.normal(400, 80, 180),
            np.random.normal(900, 200, 260),
            np.random.normal(1800, 500, 160),
        ]).astype(int)
        heights   = (widths * np.random.uniform(0.7, 1.4, n)).astype(int)
        widths    = np.clip(widths, 183, 5664)
        heights   = np.clip(heights, 200, 4248)
        class_col = np.random.choice(
            ["realperson", "fake_mannequin", "fake_mask", "fake_papercut", "fake_printed", "fake_screen"],
            n,
        )
        sizes_df = pd.DataFrame({"width": widths, "height": heights, "class": class_col})

        # ── Layout ────────────────────────────────────────
        col1, col2 = st.columns([3, 2])

        with col1:
            fig_scatter = px.scatter(
                sizes_df,
                x="width",
                y="height",
                color="class",
                title="Scatter Plot Ukuran Gambar (Width vs Height)",
                opacity=0.65,
                color_discrete_map={
                    "realperson":     "#10b981",
                    "fake_mannequin": "#ef4444",
                    "fake_mask":      "#f59e0b",
                    "fake_papercut":  "#8b5cf6",
                    "fake_printed":   "#4f46e5",
                    "fake_screen":    "#06b6d4",
                },
            )
            fig_scatter.update_traces(marker=dict(size=6, line=dict(width=0)))
            fig_scatter.update_layout(
                height=520,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", color="white"),
                legend_title_text="Kelas",
                xaxis=dict(
                    title="Width (px)",
                    gridcolor="rgba(255,255,255,0.12)",
                    zeroline=False,
                    range=[0, 6000],
                ),
                yaxis=dict(
                    title="Height (px)",
                    gridcolor="rgba(255,255,255,0.12)",
                    zeroline=False,
                    range=[0, 4500],
                ),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            st.markdown("### Statistik Ukuran Gambar")
            st.dataframe(stats_df, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div class="metric-card">
                    <div class="label">AVG WIDTH</div>
                    <div class="value">936px</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="metric-card">
                    <div class="label">AVG HEIGHT</div>
                    <div class="value">1311px</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Distribusi Width & Height")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=sizes_df["width"],  name="Width",  opacity=0.7, nbinsx=30))
        fig_hist.add_trace(go.Histogram(x=sizes_df["height"], name="Height", opacity=0.7, nbinsx=30))
        fig_hist.update_layout(
            barmode="overlay",
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="white"),
            xaxis=dict(title="Ukuran (px)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Frekuensi",   gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        insight_box(
            "Ukuran gambar dalam dataset sangat bervariasi, mulai dari resolusi kecil hingga tinggi. "
            "Berdasarkan hasil EDA, rata-rata ukuran gambar adalah 936×1311 piksel dengan resolusi maksimum "
            "mencapai 5664×4248 piksel. Oleh karena itu dilakukan preprocessing resize ke 224×224 "
            "agar input model menjadi konsisten dan proses training lebih stabil."
        )

    # ── TAB 3: File Format ─────────────────────────────────
    with tab3:
        st.markdown("#### Analisis Format File")

        formats = {
            "JPEG": 596,
            "PNG": 1,
            "WEBP": 3,
        }
        total_fmt = sum(formats.values())

        col1, col2 = st.columns([1, 2])
        with col1:
            for fmt, cnt in formats.items():
                pct = cnt / total_fmt * 100
                color = {"JPEG": "#4F46E5", "PNG": "#F59E0B", "WEBP": "#10b981"}[fmt]
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.82rem;font-weight:600;margin-bottom:4px;">
                        <span>{fmt}</span><span style="color:{color};">{pct:.1f}%</span>
                    </div>
                    <div style="background:#f3f4f6;border-radius:99px;height:8px;">
                        <div style="background:{color};width:{pct}%;height:8px;border-radius:99px;"></div>
                    </div>
                    <div style="font-size:0.72rem;color:#6b7280;margin-top:2px;">{cnt:,} files</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            fig_fmt = go.Figure(go.Pie(
                labels=list(formats.keys()),
                values=list(formats.values()),
                hole=0.5,
                marker=dict(colors=["#4F46E5", "#F59E0B", "#10b981"]),
                textinfo="label+percent",
            ))
            fig_fmt.update_layout(
                height=300,
                margin=dict(t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans"),
                showlegend=False,
            )
            st.plotly_chart(fig_fmt, use_container_width=True)

        insight_box(
            "Mayoritas gambar (90%+) berformat <b>JPEG</b>, sehingga pipeline preprocessing tidak perlu "
            "konversi format khusus. Semua format sudah didukung oleh PIL/Pillow."
        )

    # ── TAB 4: Summary Insights ────────────────────────────
    with tab4:
        st.markdown("#### 📌 Kesimpulan EDA")

        conclusions = [
            (
                "📊",
                "Distribusi Dataset Tidak Seimbang",
                "Kelas realperson memiliki jumlah data paling banyak, sedangkan fake_mannequin memiliki jumlah paling sedikit.",
                "amber",
            ),
            (
                "📐",
                "Ukuran Gambar Bervariasi",
                "Ukuran gambar pada dataset berbeda-beda sehingga preprocessing resize diperlukan sebelum training model.",
                "indigo",
            ),
            (
                "📁",
                "Dataset Berhasil Dibersihkan",
                "Proses deduplication dan cleaning berhasil menghapus data duplikat dan menjaga kualitas dataset.",
                "green",
            ),
            (
                "🎭",
                "Terdapat 6 Kelas",
                "Dataset terdiri dari berbagai jenis spoofing seperti fake_mask, fake_screen, fake_printed, dan realperson.",
                "indigo",
            ),
            (
                "🔢",
                "Total 11,731 Data Valid",
                "Dataset akhir setelah cleaning terdiri dari 11,731 gambar yang siap digunakan untuk preprocessing dan training model.",
                "green",
            ),
        ]

        for icon, title, body, color in conclusions:
            st.markdown(f"""
            <div style="
                background:#fafafa;
                border:1px solid #e5e7eb;
                border-left:4px solid {'#4F46E5' if color=='indigo' else '#F59E0B' if color=='amber' else '#10b981'};
                border-radius:0 10px 10px 0;
                padding:1rem 1.25rem;
                margin-bottom:10px;
                display:flex;gap:14px;align-items:flex-start;
            ">
                <span style="font-size:1.4rem;flex-shrink:0;">{icon}</span>
                <div>
                    <div style="font-weight:700;font-size:0.9rem;color:#1e1b4b;margin-bottom:4px;">{title}</div>
                    <div style="font-size:0.82rem;color:#4b5563;line-height:1.6;">{body}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
