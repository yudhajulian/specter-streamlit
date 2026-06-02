import streamlit as st
import numpy as np
import time
from utils.helpers import section_header, insight_box


# ── Simulated inference ────────────────────────────────────────────────────────

CLASSES = ["Real Face", "Printed Photo", "Replay Video", "3D Mask", "Cut Photo", "Digital Spoof"]
COLORS  = {
    "Real Face":     "#10b981",
    "Printed Photo": "#4F46E5",
    "Replay Video":  "#F59E0B",
    "3D Mask":       "#ef4444",
    "Cut Photo":     "#8b5cf6",
    "Digital Spoof": "#06b6d4",
}

def simulate_prediction(seed=None):
    """Simulate ensemble prediction probabilities."""
    rng = np.random.default_rng(seed)
    dominant = rng.integers(0, len(CLASSES))
    probs = rng.dirichlet(np.ones(len(CLASSES)) * 0.3)
    probs[dominant] += 0.4
    probs = probs / probs.sum()
    return dominant, probs


def render():
    section_header(
        "Prediction Demo",
        "Simulasi inferensi model face anti-spoofing secara interaktif"
    )

    st.info(
        "ℹ️ **Mode Simulasi** — Demo ini menggunakan simulasi probabilitas model. "
        "Pada deployment nyata, gambar akan diproses melalui ensemble EfficientNet yang telah dilatih.",
        icon=None
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### 📤 Upload Gambar")
        uploaded = st.file_uploader(
            "Pilih gambar wajah (JPG/PNG/WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            st.image(uploaded, caption="Gambar yang diupload", use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### ⚙️ Konfigurasi Model")
            models_used = st.multiselect(
                "Model yang digunakan dalam ensemble:",
                ["EfficientNet-B3", "ConvNeXt-Base", "EfficientNetV2-M", "EfficientNet-B4"],
                default=["EfficientNet-B3", "ConvNeXt-Base", "EfficientNetV2-M", "EfficientNet-B4"],
            )
            threshold = st.slider("Confidence threshold (%)", 0, 100, 50, 5)

            predict_btn = st.button("🔍 Analisis Gambar", use_container_width=True)

    with col_result:
        st.markdown("#### 📊 Hasil Prediksi")

        if not uploaded:
            st.markdown("""
            <div style="
                background:#f9fafb;border:2px dashed #d1d5db;
                border-radius:14px;padding:3rem;text-align:center;
                color:#9ca3af;
            ">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">🖼️</div>
                <div style="font-weight:600;font-size:0.9rem;">Upload gambar terlebih dahulu</div>
                <div style="font-size:0.78rem;margin-top:4px;">Hasil prediksi akan muncul di sini</div>
            </div>
            """, unsafe_allow_html=True)

        elif predict_btn:
            with st.spinner("Menjalankan inferensi ensemble..."):
                time.sleep(1.2)

            seed = hash(uploaded.name) % 9999
            dominant, probs = simulate_prediction(seed)
            pred_class = CLASSES[dominant]
            pred_conf  = probs[dominant] * 100
            is_real    = pred_class == "Real Face"
            color      = COLORS[pred_class]

            # Main result
            result_class = "pred-real" if is_real else "pred-fake"
            icon = "✅" if is_real else "⚠️"
            st.markdown(f"""
            <div class="pred-result {result_class}">
                <div style="font-size:2.5rem;">{icon}</div>
                <div class="pred-label" style="color:{'#065f46' if is_real else '#991b1b'};">
                    {pred_class}
                </div>
                <div class="pred-conf">
                    Confidence: <strong>{pred_conf:.1f}%</strong>
                </div>
                <div style="font-size:0.75rem;margin-top:4px;color:#4b5563;">
                    Threshold: {threshold}% · {'✅ Di atas threshold' if pred_conf >= threshold else '⚠️ Di bawah threshold'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Distribusi Probabilitas per Kelas**")

            sorted_idx = np.argsort(probs)[::-1]
            for idx in sorted_idx:
                cls   = CLASSES[idx]
                prob  = probs[idx] * 100
                col_c = COLORS[cls]
                is_top = idx == dominant
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:3px;">
                        <span style="font-weight:{'700' if is_top else '400'};color:{'#1e1b4b' if is_top else '#4b5563'};">{cls}</span>
                        <span style="font-weight:{'700' if is_top else '400'};color:{col_c};">{prob:.1f}%</span>
                    </div>
                    <div style="background:#f3f4f6;border-radius:99px;height:{'10px' if is_top else '6px'};">
                        <div style="background:{col_c};width:{prob}%;height:100%;border-radius:99px;transition:width 0.5s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Model Ensemble Details**")
            model_names = models_used if models_used else ["EfficientNet-B3"]
            for m in model_names:
                m_prob = (probs[dominant] + np.random.uniform(-0.05, 0.05)) * 100
                m_prob = min(max(m_prob, 30), 95)
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:4px 0;
                            font-size:0.78rem;border-bottom:1px solid #f3f4f6;">
                    <span style="color:#4b5563;">{m}</span>
                    <span style="color:{COLORS[pred_class]};font-weight:600;">{pred_class} ({m_prob:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="
                background:#f0f4ff;border:1px solid #c7d2fe;
                border-radius:14px;padding:2rem;text-align:center;
                color:#4b5563;
            ">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div>
                <div style="font-weight:600;font-size:0.9rem;color:#1e1b4b;">Siap untuk dianalisis</div>
                <div style="font-size:0.78rem;margin-top:4px;">Tekan tombol "Analisis Gambar" untuk memulai</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Try sample images ─────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎯 Coba dengan Skenario Simulasi")

    scenario_cols = st.columns(3)
    scenarios = [
        ("👤 Wajah Asli",    "Simulasi gambar wajah nyata di depan kamera",        0),
        ("📸 Printed Photo", "Simulasi foto dicetak dan ditunjukkan ke kamera",     1),
        ("📱 Replay Attack", "Simulasi video wajah diputar ulang di layar HP",     2),
    ]
    for col, (label, desc, dominant_cls) in zip(scenario_cols, scenarios):
        with col:
            if st.button(label, use_container_width=True, key=f"scenario_{dominant_cls}"):
                probs = np.random.dirichlet(np.ones(6) * 0.2)
                probs[dominant_cls] += 0.5
                probs = probs / probs.sum()
                pred_class = CLASSES[dominant_cls]
                pred_conf  = probs[dominant_cls] * 100
                is_real    = dominant_cls == 0

                st.session_state[f"scenario_result_{dominant_cls}"] = {
                    "pred_class": pred_class,
                    "pred_conf": pred_conf,
                    "probs": probs,
                    "is_real": is_real,
                }

            result_key = f"scenario_result_{dominant_cls}"
            if result_key in st.session_state:
                r = st.session_state[result_key]
                color   = "#10b981" if r["is_real"] else "#ef4444"
                bg      = "#d1fae5" if r["is_real"] else "#fee2e2"
                icon    = "✅" if r["is_real"] else "⚠️"
                st.markdown(f"""
                <div style="background:{bg};border-radius:10px;padding:0.75rem;text-align:center;margin-top:6px;">
                    <div>{icon} <strong style="color:{color};">{r['pred_class']}</strong></div>
                    <div style="font-size:0.78rem;color:#4b5563;">{r['pred_conf']:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#f9fafb;border-radius:10px;padding:0.75rem;
                            text-align:center;margin-top:6px;color:#9ca3af;font-size:0.78rem;">
                    {desc}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    insight_box(
        "Pada deployment nyata, pipeline inferensi mencakup: face detection → crop & resize ke 224×224 → "
        "normalisasi ImageNet → forward pass melalui 4 model secara paralel → soft voting ensemble → "
        "output prediksi dengan confidence score. Latency estimasi: ~200ms per gambar pada GPU T4."
    )
