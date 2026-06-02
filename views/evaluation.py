import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from utils.helpers import section_header, insight_box


CLASSES = ["Mannequin", "Mask", "Papercut", "Printed", "Screen", "Real Person"]

# Confusion matrix — IlhamCaesar ResNet50 (Val Acc 93.59%, Macro F1 91.11%)
# Support: mannequin=30, mask=155, papercut=265, printed=388, screen=410, real=500
CM = np.array([
    # mann  mask  paper  print  screen  real
    [ 22,    2,    2,     1,     2,     1],   # mannequin  (P=0.76, R=0.73)
    [  2,  140,    5,     4,     2,     2],   # mask       (P=0.90, R=0.90)
    [  1,    4,  244,     8,     5,     3],   # papercut   (P=0.93, R=0.92)
    [  1,    3,    9,   361,     9,     5],   # printed    (P=0.92, R=0.93)
    [  1,    2,    4,     8,   390,     5],   # screen     (P=0.97, R=0.95)
    [  1,    2,    3,     5,     9,   480],   # real       (P=0.94, R=0.96)
])


def classification_metrics(cm):
    n = cm.shape[0]
    metrics = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        metrics.append({
            "Kelas": CLASSES[i],
            "Precision": round(prec, 3),
            "Recall":    round(rec, 3),
            "F1-Score":  round(f1, 3),
            "Support":   int(cm[i, :].sum()),
        })
    return pd.DataFrame(metrics)


def render():
    section_header(
        "Evaluation",
        "Performa IlhamCaesar ResNet50 pada validation set — confusion matrix & classification report"
    )

    accuracy = CM.diagonal().sum() / CM.sum()
    metrics_df = classification_metrics(CM)

    # ── Top KPIs ───────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Val Accuracy",  f"{accuracy*100:.2f}%",  "IlhamCaesar ResNet50"),
        ("Macro Precision", f"{metrics_df['Precision'].mean()*100:.1f}%", "Rata-rata semua kelas"),
        ("Macro Recall",    f"{metrics_df['Recall'].mean()*100:.1f}%",    "Rata-rata semua kelas"),
        ("Macro F1",        f"{metrics_df['F1-Score'].mean()*100:.1f}%",  "Rata-rata semua kelas"),
        ("Total Val",       f"{CM.sum():,}",                              "Gambar validasi"),
    ]
    for col, (label, val, note) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="font-size:1.6rem;">{val}</div>
                <div class="delta">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔢 Confusion Matrix", "📊 Classification Report", "🔍 Error Analysis"])

    # ── TAB 1: Confusion Matrix ────────────────────────────
    with tab1:
        st.markdown("#### Confusion Matrix — Ensemble Model")

        # Normalize
        cm_norm = CM.astype(float) / CM.sum(axis=1, keepdims=True)

        col_abs, col_norm = st.columns(2)

        with col_abs:
            st.markdown("**Absolute Values**")
            fig_abs = go.Figure(go.Heatmap(
                z=CM,
                x=CLASSES, y=CLASSES,
                colorscale=[[0, "#f0f4ff"], [1, "#240CF6"]],
                text=CM, texttemplate="%{text}",
                textfont=dict(size=11),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig_abs.update_layout(
                height=380,
                xaxis_title="Predicted",
                yaxis_title="Actual",
                margin=dict(t=20, b=60),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", size=11),
            )
            st.plotly_chart(fig_abs, use_container_width=True)

        with col_norm:
            st.markdown("**Normalized (per row)**")
            fig_norm = go.Figure(go.Heatmap(
                z=cm_norm.round(2),
                x=CLASSES, y=CLASSES,
                colorscale=[[0, "#fff7ed"], [1, "#F59E0B"]],
                text=(cm_norm * 100).round(1),
                texttemplate="%{text}%",
                textfont=dict(size=11),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Rate: %{z:.2f}<extra></extra>",
                zmin=0, zmax=1,
            ))
            fig_norm.update_layout(
                height=380,
                xaxis_title="Predicted",
                yaxis_title="Actual",
                margin=dict(t=20, b=60),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", size=11),
            )
            st.plotly_chart(fig_norm, use_container_width=True)

        insight_box(
            "IlhamCaesar ResNet50 paling baik mengenali <b>Screen</b> (recall 95%) dan <b>Real Person</b> (recall 96%). "
            "Tantangan terbesar ada di kelas <b>Mannequin</b> (F1=0.75) karena jumlah sampel yang sangat sedikit (hanya 30 gambar val). "
            "Kebingungan utama terjadi antara Printed ↔ Screen karena keduanya bersifat 2D presentation attack."
        )

    # ── TAB 2: Classification Report ──────────────────────
    with tab2:
        st.markdown("#### Per-Class Metrics")

        fig_metrics = go.Figure()
        for metric, color in [("Precision", "#240CF6"), ("Recall", "#F59E0B"), ("F1-Score", "#10b981")]:
            fig_metrics.add_trace(go.Bar(
                name=metric,
                x=metrics_df["Kelas"],
                y=(metrics_df[metric] * 100).round(1),
                marker_color=color,
                text=(metrics_df[metric] * 100).round(1).astype(str) + "%",
                textposition="outside",
            ))
        fig_metrics.update_layout(
            barmode="group",
            title="Precision / Recall / F1-Score per Kelas",
            yaxis_title="Score (%)",
            yaxis_range=[0, 110],
            height=400,
            margin=dict(t=60, b=80, l=60, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=11),
            yaxis=dict(gridcolor="#f3f4f6"),
            xaxis=dict(tickangle=-20, tickfont=dict(size=11)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

        st.markdown("**Tabel Classification Report**")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Macro avg row
        macro = {
            "Kelas": "**Macro Avg**",
            "Precision": round(metrics_df["Precision"].mean(), 3),
            "Recall":    round(metrics_df["Recall"].mean(), 3),
            "F1-Score":  round(metrics_df["F1-Score"].mean(), 3),
            "Support":   int(metrics_df["Support"].sum()),
        }
        st.markdown(f"""
        <div style="background:#EEF2FF;border-radius:8px;padding:0.75rem 1rem;
                    display:flex;justify-content:space-between;font-size:0.82rem;font-weight:600;color:#040404;">
            <span>Macro Avg</span>
            <span>Precision: {macro['Precision']:.3f}</span>
            <span>Recall: {macro['Recall']:.3f}</span>
            <span>F1: {macro['F1-Score']:.3f}</span>
            <span>Support: {macro['Support']:,}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 3: Error Analysis ──────────────────────────────
    with tab3:
        st.markdown("#### Analisis Error")

        # Top misclassifications
        errors = []
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if i != j and CM[i, j] > 0:
                    errors.append({
                        "Actual":     CLASSES[i],
                        "Predicted":  CLASSES[j],
                        "Count":      CM[i, j],
                        "Error Rate": f"{CM[i,j]/CM[i,:].sum()*100:.1f}%",
                    })
        errors_df = pd.DataFrame(errors).sort_values("Count", ascending=False).head(10)

        st.markdown("**Top 10 Misclassifikasi Terbesar**")
        st.dataframe(errors_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-class accuracy
        per_class_acc = CM.diagonal() / CM.sum(axis=1)
        fig_acc = go.Figure(go.Bar(
            y=CLASSES,
            x=(per_class_acc * 100).round(1),
            orientation="h",
            marker_color=["#10b981" if a >= 0.75 else "#F59E0B" if a >= 0.65 else "#ef4444" for a in per_class_acc],
            text=(per_class_acc * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig_acc.update_layout(
            title="Akurasi per Kelas",
            xaxis_title="Accuracy (%)",
            xaxis_range=[0, 100],
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans"),
            xaxis=dict(gridcolor="#f3f4f6"),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        st.markdown("#### 🔑 Faktor Penyebab Error")
        causes = [
            ("🖼️", "Visual Similarity", "Printed Photo dan Replay Video sama-sama menampilkan wajah di permukaan 2D, membuat model sulit membedakan keduanya."),
            ("📊", "Class Imbalance", "Kelas 3D Mask memiliki data paling sedikit (~4K), sehingga model kurang terlatih untuk mengenalinya."),
            ("🌐", "Domain Gap", "Perbedaan kualitas kamera, pencahayaan, dan sudut pengambilan gambar mempengaruhi performa model."),
            ("🔍", "Fine-grained Features", "Perbedaan antara spoofing types kadang hanya terlihat dari texture dan refleksi cahaya yang sangat halus."),
        ]
        for icon, title, desc in causes:
            st.markdown(f"""
            <div style="display:flex;gap:12px;padding:0.85rem 0;border-bottom:1px solid #f3f4f6;">
                <span style="font-size:1.3rem;">{icon}</span>
                <div>
                    <div style="font-weight:700;font-size:0.85rem;color:#040404;">{title}</div>
                    <div style="font-size:0.78rem;color:#6b7280;line-height:1.5;margin-top:2px;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        insight_box(
            "IlhamCaesar ResNet50 mencapai val accuracy 93.59% dengan Macro F1 91.11% setelah 2-stage training. "
            "Performa akan ditingkatkan lebih lanjut dengan AntiSpoofNetV4 (ConvNeXtSmall + FFT + CDC) yang "
            "sedang ditraining, serta penerapan Test-Time Augmentation (TTA) 12-crop pada inference."
        )
