import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils.helpers import section_header, insight_box, bi_icon


def render():
    section_header(
        "Data Cleaning & Preprocessing",
        "Pipeline pembersihan data dan persiapan dataset untuk training model"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Pipeline Overview",
        "Deduplication",
        "Data Cleaning",
        "Dataset Split",
    ])

    # ── TAB 1: Pipeline ────────────────────────────────────
    with tab1:
        st.markdown("#### Alur Preprocessing")

        steps = [
    (
        "1",
        "Load Dataset",
        "Membaca seluruh dataset face anti-spoofing dari 6 kategori dengan total 11,819 gambar.",
        "#040404"
    ),

    (
        "2",
        "Buat Dataframe",
        "Setiap gambar dicatat ke dalam DataFrame beserta label kelas dan MD5 hash untuk proses identifikasi duplikat.",
        "#7c3aed"
    ),

    (
        "3",
        "Deduplication",
        "Menghapus 88 gambar duplikat menggunakan hash MD5 sehingga tersisa 11,731 gambar unik.",
        "#F59E0B"
    ),

    (
        "4",
        "Data Cleaning",
        "Verifikasi integritas gambar menggunakan PIL untuk memastikan seluruh file valid dan siap diproses.",
        "#ef4444"
    ),

    (
        "5",
        "Stratified Split",
        "Dataset dibagi menjadi Train (70%), Validation (15%), dan Test (15%) dengan stratifikasi kelas.",
        "#10b981"
    ),

    (
        "6",
        "Copy & Export",
        "Melakukan pengecekan duplicate antar split, copy dataset ke folder output, dan export processed_dataset.zip.",
        "#06b6d4"
    ),
]

        for num, title, desc, color in steps:
            st.markdown(f"""
            <div style="display:flex;gap:16px;margin-bottom:12px;align-items:flex-start;">
                <div style="
                    width:36px;height:36px;background:{color};color:white;
                    border-radius:50%;display:flex;align-items:center;justify-content:center;
                    font-weight:800;font-size:0.9rem;flex-shrink:0;
                ">{num}</div>
                <div style="
                    flex:1;background:#FFFFFF;border:1px solid #E9E9E9;
                    border-radius:10px;padding:0.75rem 1rem;
                ">
                    <div style="font-weight:700;font-size:0.88rem;color:#040404;">{title}</div>
                    <div style="font-size:0.78rem;color:#52525B;margin-top:3px;line-height:1.5;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<h4 style="display:flex;align-items:center;gap:6px;">{bi_icon("code-slash", "#040404", "1rem")} Kode: Membuat Dataframe & Hash</h4>', unsafe_allow_html=True)
        st.code("""
import hashlib
import pandas as pd

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

data = []
for cls in classes:
    cls_path = os.path.join(base_path, cls)
    for img in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img)
        try:
            img_hash = get_hash(img_path)
            data.append([img, cls, img_hash])
        except:
            continue

df = pd.DataFrame(data, columns=["image", "label", "hash"])
        """, language="python")

    # ── TAB 2: Deduplication ───────────────────────────────
    with tab2:
        st.markdown("#### Hash-Based Deduplication")
        st.markdown("""
        <div class="insight-box">
            Setiap gambar dihitung nilai <code>MD5 hash</code>-nya. Gambar dengan hash identik 
            dianggap duplikat meskipun nama filenya berbeda. Metode ini lebih andal 
            dibanding pengecekan nama file saja.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="label">Sebelum Dedup</div>
                <div class="value">11,819</div>
                <div class="delta">Total gambar raw</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="label">Duplikat Ditemukan</div>
                <div class="value" style="color:#ef4444;">88</div>
                <div class="delta neg">1.9% dari total</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="label">Sesudah Dedup</div>
                <div class="value" style="color:#10b981;">11,731</div>
                <div class="delta">Gambar unik</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Duplikat dihapus per kelas (selisih distribusi awal − split totals, total = 88)
        dup_per_class = {
            "fake_mannequin": 33,
            "fake_mask":       6,
            "fake_papercut":   0,
            "fake_printed":   17,
            "fake_screen":     4,
            "realperson":     28,
        }
        fig = go.Figure(go.Bar(
            x=list(dup_per_class.keys()),
            y=list(dup_per_class.values()),
            marker_color="#ef4444",
            text=list(dup_per_class.values()),
            textposition="outside",
        ))
        fig.update_layout(
            title="Duplikat Dihapus per Kelas (total = 88)",
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans"),
            yaxis=dict(gridcolor="#f3f4f6"),
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.code("""
# Hapus duplikat berdasarkan hash
df_unique = df.drop_duplicates(subset=['hash'])

print("Sebelum Deduplikasi:", len(df))
print("Sesudah Deduplikasi:", len(df_unique))
        """, language="python")

    # ── TAB 3: Data Cleaning ───────────────────────────────
    with tab3:
        st.markdown("#### Verifikasi Integritas Gambar")
        st.markdown("""
        <div class="insight-box">
            Setelah deduplication, setiap gambar diverifikasi menggunakan 
            <code>img.verify()</code> dari PIL untuk memastikan tidak ada file corrupt 
            atau tidak dapat dibuka.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sebelum Cleaning**")
            st.markdown("""
            <div class="metric-card">
                <div class="label">Total Unik (post-dedup)</div>
                <div class="value">11,731</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("**Sesudah Cleaning**")
            st.markdown("""
            <div class="metric-card">
                <div class="label">Data Valid</div>
                <div class="value" style="color:#10b981;">11,731</div>
                <div class="delta">0 file corrupt ditemukan</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Checklist Kualitas Data")

        checks = [
            ("Tidak ada file corrupt",         "PIL verify() berhasil pada semua gambar yang tersisa"),
            ("Tidak ada gambar kosong",         "Semua file memiliki ukuran > 0 bytes"),
            ("Format valid (JPEG/PNG/WEBP)",    "Semua format yang didukung oleh model"),
            ("Label konsisten",                 "Setiap gambar memiliki label kelas yang valid"),
            ("No NaN/null values",              "DataFrame bebas dari nilai kosong"),
        ]
        for title, desc in checks:
            st.markdown(f"""
            <div style="display:flex;gap:10px;padding:0.6rem 0;border-bottom:1px solid #f3f4f6;align-items:center;">
                <div style="flex-shrink:0;">{bi_icon("check-circle-fill", "#10b981", "1.1rem")}</div>
                <div>
                    <span style="font-weight:600;font-size:0.85rem;color:#040404;">{title}</span>
                    <span style="font-size:0.78rem;color:#52525B;"> — {desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.code("""
valid_data = []
for _, row in df_unique.iterrows():
    img_path = os.path.join(base_path, row['label'], row['image'])
    try:
        img = Image.open(img_path)
        img.verify()  # raises exception if corrupt
        valid_data.append(row)
    except:
        continue  # skip corrupt files

df_clean = pd.DataFrame(valid_data)
print("Data valid:", len(df_clean))
        """, language="python")

    # ── TAB 4: Dataset Split ───────────────────────────────
    with tab4:
        st.markdown("#### Stratified Train/Val/Test Split")
        st.markdown("""
        <div class="insight-box">
            Dataset dibagi dengan <b>Stratified Split</b> — memastikan setiap kelas 
            terwakili secara proporsional di Train, Validation, dan Test set. 
            Duplikat antar split juga dicek ulang untuk menghindari data leakage.
        </div>
        """, unsafe_allow_html=True)

        # Split visualization
        split_data = {
            "Train":8211,
            "Validation ":1760,
            "Test ":1760,
        }

        col1, col2, col3 = st.columns(3)
        for col, (label, count), color in zip(
            [col1, col2, col3],
            split_data.items(),
            ["#040404", "#F59E0B", "#10b981"]
        ):
            with col:
                st.markdown(f"""
                <div style="
                    background:{color}15;border:2px solid {color};
                    border-radius:12px;padding:1.2rem;text-align:center;
                ">
                    <div style="font-size:0.8rem;font-weight:700;color:{color};">{label}</div>
                    <div style="font-size:2rem;font-weight:800;color:{color};margin:4px 0;">{count:,}</div>
                    <div style="font-size:0.72rem;color:#52525B;">gambar</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-class distribution per split
        df_split = pd.DataFrame({

    "Kelas": [
        "fake_mannequin",
        "fake_mask",
        "fake_papercut",
        "fake_printed",
        "fake_screen",
        "realperson"
    ],

    "Train": [
        143,
        729,
        1239,
        1845,
        1924,
        2331
    ],

    "Validation": [
        30,
        156,
        266,
        395,
        413,
        500
    ],

    "Test": [
        31,
        156,
        265,
        396,
        412,
        500
    ]
})
        st.markdown("**Distribusi per Kelas per Split**")
        st.dataframe(df_split, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Duplicate Check Antar Split")
        dup_checks = [
            ("Train ↔ Validation", 0),
            ("Train ↔ Test",       0),
            ("Validation ↔ Test",  0),
        ]
        for check, count in dup_checks:
            color = "#10b981" if count == 0 else "#ef4444"
            status_icon = bi_icon("check-circle-fill", color, "0.95rem")
            status_text = "Aman" if count == 0 else "Ada duplikat"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.6rem 1rem;
                        background:#FAFAFA;border-radius:8px;margin-bottom:6px;align-items:center;">
                <span style="font-size:0.85rem;font-weight:500;">{check}</span>
                <div style="display:flex;gap:12px;align-items:center;">
                    <span style="font-size:0.82rem;color:#52525B;">{count} duplikat</span>
                    <span style="display:flex;align-items:center;gap:4px;color:{color};font-weight:700;font-size:0.85rem;">{status_icon} {status_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.code("""
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df_clean,
    test_size=0.3,
    stratify=df_clean['label'],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['label'],
    random_state=42
)

print("Train:", len(train_df))
print("Val:  ", len(val_df))
print("Test: ", len(test_df))
        """, language="python")

        insight_box(
            "Tidak ditemukan data leakage antar split. Hash-based duplicate check memastikan "
            "tidak ada gambar yang sama muncul di Train, Validation, dan Test secara bersamaan. "
            "Dataset siap digunakan untuk training model."
        )
