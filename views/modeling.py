import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils.helpers import section_header, insight_box, bi_icon


def render():
    section_header(
        "Modeling",
        "Arsitektur model, strategi training, dan eksperimen yang dilakukan"
    )

    tab1, tab2, tab3 = st.tabs(["Arsitektur", "Training Strategy", "Eksperimen"])

    # ── TAB 1: Architecture ────────────────────────────────
    with tab1:
        st.markdown("#### Model yang Digunakan")

        models = [
            {
                "name": "AntiSpoofNetV4",
                "icon": "cpu-fill",
                "params": "~28M",
                "input": "256×256",
                "accuracy": "—",
                "role": "Model Utama",
                "desc": "Multi-branch architecture: ConvNeXtSmall backbone + AttentionPooling (768-dim), FFT frequency branch (256-dim), dan Contextual Difference Convolution (CDC) texture branch (256-dim). Fusion 1280-dim → FSFM embedder 512-dim.",
                "color": "#040404",
                "badge": "indigo",
            },
            {
                "name": "IlhamCaesar ResNet50",
                "icon": "diagram-3-fill",
                "params": "~25M",
                "input": "224×224",
                "accuracy": "93.59%",
                "role": "Model Alternatif",
                "desc": "Transfer learning dari ResNet50 (ImageNet pretrained). Dua tahap: head-only training (10 epoch) dilanjutkan fine-tuning 30 layer backbone terakhir (5 epoch). Val Acc 93.59%, Macro F1 91.11%.",
                "color": "#10b981",
                "badge": "green",
            },
        ]

        cols = st.columns(2)
        for i, m in enumerate(models):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="
                    background:#FFFFFF;border:1px solid #E9E9E9;
                    border-left:4px solid {m['color']};
                    border-radius:0 12px 12px 0;padding:1.1rem 1.2rem;margin-bottom:12px;
                ">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div style="display:flex;gap:10px;align-items:center;">
                            <div style="font-size:1.5rem;">{bi_icon(m['icon'], m['color'], '1.5rem')}</div>
                            <div>
                                <div style="font-weight:800;font-size:0.95rem;color:#040404;">{m['name']}</div>
                                <span class="badge badge-{m['badge']}">{m['role']}</span>
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:1.2rem;font-weight:800;color:{m['color']};">{m['accuracy']}</div>
                            <div style="font-size:0.68rem;color:#A1A1AA;">val acc.</div>
                        </div>
                    </div>
                    <div style="font-size:0.78rem;color:#52525B;margin-top:8px;line-height:1.5;">{m['desc']}</div>
                    <div style="display:flex;gap:16px;margin-top:8px;">
                        <div style="font-size:0.72rem;color:#A1A1AA;">Params: <b style="color:#040404;">{m['params']}</b></div>
                        <div style="font-size:0.72rem;color:#A1A1AA;">Input: <b style="color:#040404;">{m['input']}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<h4 style="display:flex;align-items:center;gap:6px;">{bi_icon("boxes", "#040404", "1rem")} AntiSpoofNetV4 — Detail Arsitektur</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#FAFAFA;border:1px solid #E9E9E9;border-radius:12px;padding:1.5rem;">
            <div style="font-size:0.82rem;color:#52525B;line-height:1.8;">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                    <div>
                        <div style="font-weight:700;color:#040404;margin-bottom:6px;">Backbone Branch</div>
                        <div>ConvNeXtSmall (ImageNet pretrained)</div>
                        <div style="color:#A1A1AA;">→ AttentionPooling → 768-dim</div>
                    </div>
                    <div>
                        <div style="font-weight:700;color:#040404;margin-bottom:6px;">Frequency Branch</div>
                        <div>FFT 2D → Conv Stack</div>
                        <div style="color:#A1A1AA;">→ 256-dim features</div>
                    </div>
                    <div>
                        <div style="font-weight:700;color:#040404;margin-bottom:6px;">Texture Branch</div>
                        <div>Central Difference Conv (CDC)</div>
                        <div style="color:#A1A1AA;">→ 256-dim features</div>
                    </div>
                    <div>
                        <div style="font-weight:700;color:#040404;margin-bottom:6px;">Fusion + Head</div>
                        <div>Concat (768+256+256=1280)</div>
                        <div style="color:#A1A1AA;">→ Dense(512) L2-norm → Dense(6)</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<h4 style="display:flex;align-items:center;gap:6px;">{bi_icon("person-bounding-box", "#040404", "1rem")} Face Identity — InsightFace ArcFace</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#FAFAFA;border:1px solid #E9E9E9;border-radius:12px;padding:1rem 1.5rem;">
            <div style="font-size:0.82rem;color:#52525B;line-height:1.7;">
                Model <b style="color:#040404;">buffalo_l</b> (ONNX) dari InsightFace digunakan untuk mengekstrak
                512-dim identity embedding bagi face matching. Output L2-normalized dengan similarity threshold
                <b style="color:#240CF6;">0.40</b>. Fallback: embedding dari FSFM layer AntiSpoofNetV4.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: Training Strategy ───────────────────────────
    with tab2:
        st.markdown("#### Konfigurasi Training — AntiSpoofNetV4 (5-Stage Progressive)")

        stages = [
            ("Stage 1", "Epoch 0–10",  "5e-4",  "Head only (backbone frozen)",         "Warmup 5 epoch"),
            ("Stage 2", "Epoch 10–25", "2e-4",  "Partial backbone unfreezing",         "Warmup 1 epoch"),
            ("Stage 3", "Epoch 25–42", "8e-5",  "Lebih banyak backbone dibuka",        "Cosine decay"),
            ("Stage 4", "Epoch 42–92", "1e-5",  "Hampir full backbone, batch size=8",  "Warmup 3 epoch"),
            ("Stage 5", "Epoch 92–150","5e-6",  "Full fine-tuning semua layer",        "Final refinement"),
        ]

        for stage, epoch, lr, desc, note in stages:
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:8px;align-items:flex-start;">
                <div style="
                    min-width:72px;background:#EEF2FF;color:#240CF6;
                    border-radius:6px;padding:4px 8px;font-size:0.72rem;font-weight:700;text-align:center;
                ">{stage}</div>
                <div style="flex:1;background:#FAFAFA;border:1px solid #E9E9E9;border-radius:8px;padding:0.6rem 1rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-weight:600;font-size:0.82rem;color:#040404;">{epoch} — {desc}</div>
                        <div style="font-size:0.72rem;color:#A1A1AA;">LR={lr} · {note}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Konfigurasi Utama**")
            configs = [
                ("Loss Function",  "HingeACER Focal Loss (γ=2.5)",      "APCER + BPCER penalty, label smooth=0.08"),
                ("Optimizer",      "AdamW + weight decay (5e-4)",        "EMA decay=0.9995 untuk stabilitas val"),
                ("Batch Size",     "32 (Stage 1-3) → 8 (Stage 4-5)",    "Disesuaikan GPU memory"),
                ("Early Stopping", "Patience=15 epoch (val F1)",         "Best checkpoint disimpan"),
                ("Image Size",     "256×256 (AntiSpoofNetV4)",           "224×224 (ResNet50)"),
                ("Oversampling",   "2,500 gambar per kelas",             "Dynamic balancing saat training"),
            ]
            for param, value, note in configs:
                st.markdown(f"""
                <div style="padding:0.55rem 0;border-bottom:1px solid #f3f4f6;">
                    <div style="display:flex;justify-content:space-between;gap:8px;">
                        <span style="font-size:0.78rem;color:#52525B;min-width:110px;">{param}</span>
                        <div style="text-align:right;">
                            <div style="font-size:0.8rem;font-weight:600;color:#040404;">{value}</div>
                            <div style="font-size:0.7rem;color:#A1A1AA;">{note}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("**Augmentasi Data**")
            augs = [
                ("shuffle",          "MixUp (α=0.4)",             "Interpolasi antar gambar & label"),
                ("scissors",         "CutMix (α=1.0)",            "Patch swap antar gambar"),
                ("arrow-repeat",     "Flip & Rotation (±18°)",    "Geometric augmentation dasar"),
                ("sliders",          "Brightness/Contrast/Color", "Photometric jitter"),
                ("camera-fill",      "JPEG Simulation",           "Mensimulasikan kompresi print"),
                ("water",            "Moiré Overlay",             "Artefak replay/screen attack"),
                ("soundwave",        "Gaussian Noise",            "Robustness terhadap sensor noise"),
                ("zoom-in",          "Elastic Distortion",        "Deformasi spasial realistis"),
            ]
            for icon, name, desc in augs:
                st.markdown(f"""
                <div style="display:flex;gap:8px;padding:0.5rem 0;border-bottom:1px solid #f9fafb;align-items:center;">
                    <div style="flex-shrink:0;">{bi_icon(icon, '#240CF6', '1rem')}</div>
                    <div>
                        <span style="font-size:0.8rem;font-weight:600;color:#040404;">{name}</span>
                        <span style="font-size:0.7rem;color:#52525B;"> — {desc}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        insight_box(
            "HingeACER Focal Loss menggabungkan Focal Loss (γ=2.5) dengan ACER penalty yang secara eksplisit "
            "meminimalkan Attack Presentation Classification Error Rate (APCER + BPCER). Ini lebih efektif "
            "dari standard Focal Loss untuk FAS karena langsung mengoptimalkan metrik keamanan yang relevan."
        )

    # ── TAB 3: Experiments ────────────────────────────────
    with tab3:
        st.markdown("#### Riwayat Eksperimen")

        experiments = [
            {"Iterasi": "v1", "Model": "IlhamCaesar ResNet50 (Stage 1)", "Loss": "CrossEntropy",       "Epoch": "10", "Val Acc": 0.9045, "Note": "Head-only baseline"},
            {"Iterasi": "v2", "Model": "IlhamCaesar ResNet50 (Stage 2)", "Loss": "CrossEntropy",       "Epoch": "15", "Val Acc": 0.9359, "Note": "🏆 Best (ResNet50)"},
            {"Iterasi": "v3", "Model": "AntiSpoofNetV4 (Stage 1-3)",     "Loss": "HingeACER Focal",    "Epoch": "42", "Val Acc": None,   "Note": "In progress"},
            {"Iterasi": "v4", "Model": "AntiSpoofNetV4 (Full)",          "Loss": "HingeACER Focal",    "Epoch": "150","Val Acc": None,   "Note": "Target deployment"},
        ]

        df_exp = pd.DataFrame(experiments)
        df_plot = df_exp[df_exp["Val Acc"].notna()].copy()

        colors = ["#040404" if v == max(df_plot["Val Acc"]) else "#c7d2fe" for v in df_plot["Val Acc"]]
        fig = go.Figure(go.Bar(
            x=df_plot["Iterasi"],
            y=(df_plot["Val Acc"] * 100).round(2),
            marker_color=colors,
            text=(df_plot["Val Acc"] * 100).round(2).astype(str) + "%",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Val Accuracy: %{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            title="Progres Akurasi per Iterasi (Hasil yang Tersedia)",
            yaxis_title="Validation Accuracy (%)",
            yaxis_range=[85, 100],
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            yaxis=dict(gridcolor="#f3f4f6"),
        )
        st.plotly_chart(fig, use_container_width=True)

        df_display = df_exp.copy()
        df_display["Val Acc"] = df_display["Val Acc"].apply(
            lambda v: f"{v*100:.2f}%" if v is not None else "In Progress"
        )
        st.dataframe(df_display, use_container_width=True, hide_index=True)
