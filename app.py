"""
Streamlit App — Prediksi Harga Mobil Bekas
Dataset: Kaggle Playground Series S4E9 (Used Car Price Prediction)
Model  : XGBoost (tuned) — hasil terbaik dari notebook Final_Project_DS59
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="Kaggle Playground Series | Prediksi Harga Mobil Bekas",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# THEME — palet "showroom dealer mobil"
RED = "#E63946"
GOLD = "#F4A300"
ASPHALT = "#11141B"
CARD = "#1B2030"
CARD_ALT = "#222842"
MUTED = "#9AA0AD"
LIGHT = "#F2F3F5"
GREEN = "#3DDC97"

PLOTLY_TEMPLATE = "plotly_dark"
SEQ_SCALE = ["#1B2030", "#3A3360", "#7A2E45", "#C23B4B", "#E63946", "#F4A300"]
CATEGORICAL_SCALE = [RED, GOLD, "#4D96FF", GREEN, "#9D4EDD", "#FF6B6B", "#6C757D"]

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at 15% 0%, rgba(230,57,70,0.10), transparent 35%),
            radial-gradient(circle at 85% 10%, rgba(244,163,0,0.08), transparent 40%),
            {ASPHALT};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {CARD};
        border-right: 1px solid #2E3548;
    }}
    h1, h2, h3 {{
        font-family: 'Trebuchet MS', sans-serif;
        letter-spacing: 0.3px;
    }}
    .hero {{
        background: linear-gradient(120deg, #1B2030 0%, #2A1622 55%, #2A1A0E 100%);
        border: 1px solid #33394d;
        border-radius: 18px;
        padding: 34px 38px;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }}
    .hero::after {{
        content: "";
        position: absolute;
        right: -40px; top: -40px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(244,163,0,0.18), transparent 70%);
    }}
    .hero h1 {{
        font-size: 2.1rem;
        margin-bottom: 6px;
        background: linear-gradient(90deg, {LIGHT}, {GOLD});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero p {{
        color: {MUTED};
        font-size: 1.02rem;
        max-width: 720px;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }}
    .badge-red {{ background: rgba(230,57,70,0.18); color: #ff8a92; border: 1px solid rgba(230,57,70,0.4); }}
    .badge-gold {{ background: rgba(244,163,0,0.18); color: {GOLD}; border: 1px solid rgba(244,163,0,0.4); }}
    .badge-green {{ background: rgba(61,220,151,0.15); color: {GREEN}; border: 1px solid rgba(61,220,151,0.4); }}

    .metric-card {{
        background: linear-gradient(160deg, {CARD} 0%, {CARD_ALT} 100%);
        border: 1px solid #2E3548;
        border-radius: 16px;
        padding: 18px 20px;
        text-align: left;
    }}
    .metric-card .label {{
        color: {MUTED};
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }}
    .metric-card .value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: {LIGHT};
    }}
    .price-result {{
        background: linear-gradient(135deg, #2A1622 0%, #1B2030 60%, #2A1A0E 100%);
        border: 1px solid {GOLD};
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 10px;
    }}
    .price-result .price {{
        font-size: 3rem;
        font-weight: 800;
        color: {GOLD};
        margin: 6px 0;
    }}
    .price-result .range {{
        color: {MUTED};
        font-size: 0.95rem;
    }}
    .section-divider {{
        border-top: 1px solid #2E3548;
        margin: 18px 0 22px 0;
    }}
    .stButton > button {{
        background: linear-gradient(90deg, {RED}, #c8313d);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }}
    .stButton > button:hover {{
        background: linear-gradient(90deg, #ff5158, {RED});
        color: white;
    }}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# LOAD ARTIFACTS
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/model.joblib")
    prep = joblib.load("artifacts/preprocessing.joblib")
    with open("artifacts/metadata.json") as f:
        meta = json.load(f)
    return model, prep, meta


@st.cache_data
def load_eda_data():
    return pd.read_csv("artifacts/eda_data.csv")


@st.cache_data
def load_val_predictions():
    return pd.read_csv("artifacts/val_predictions.csv")


@st.cache_data
def load_feature_importance():
    fi = pd.read_csv("artifacts/feature_importance.csv", index_col=0)
    return fi


model, prep, meta = load_artifacts()
options = meta["options"]
ranges = meta["ranges"]
metrics = meta["metrics"]
medians = meta["medians"]

FRIENDLY_NAMES = {
    "model_year": "Tahun Produksi",
    "milage": "Jarak Tempuh (mil)",
    "milage_per_year": "Jarak Tempuh / Tahun",
    "horsepower": "Tenaga Mesin (HP)",
    "engine_size": "Kapasitas Mesin (L)",
    "cylinder": "Jumlah Silinder",
    "cylinder_missing": "Silinder Tidak Diketahui",
    "turbo": "Turbo",
    "hybrid": "Hybrid / Electric",
    "hp_per_liter": "HP per Liter",
    "hp_per_cylinder": "HP per Silinder",
    "transmission_speed": "Jumlah Percepatan Transmisi",
    "accident": "Riwayat Kecelakaan",
    "clean_title": "Status Clean Title",
    "brand_te": "Pengaruh Brand (encoded)",
    "model_te": "Pengaruh Model (encoded)",
    "ext_col_freq": "Popularitas Warna Eksterior",
    "int_col_freq": "Popularitas Warna Interior",
}


def friendly(col: str) -> str:
    if col in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[col]
    if col.startswith("fuel_type_"):
        return f"Bahan Bakar: {col.replace('fuel_type_', '')}"
    if col.startswith("transmission_type_"):
        return f"Transmisi: {col.replace('transmission_type_', '')}"
    return col


# INFERENCE
def predict_price(inputs: dict):
    current_year = meta["current_year"]
    car_age = current_year - inputs["model_year"]
    milage_per_year = inputs["milage"] / (car_age + 1)
    hp_per_liter = inputs["horsepower"] / (inputs["engine_size"] + 1e-5)
    hp_per_cylinder = inputs["horsepower"] / (inputs["cylinder"] + 1e-5)

    accident_enc = prep["le_accident"].transform([inputs["accident"]])[0]
    clean_title_enc = prep["le_clean_title"].transform([inputs["clean_title"]])[0]

    ohe_input = pd.DataFrame(
        {"fuel_type": [inputs["fuel_type"]], "transmission_type": [inputs["transmission_type"]]}
    )
    ohe_out = prep["ohe_encoder"].transform(ohe_input)
    ohe_cols_out = prep["ohe_encoder"].get_feature_names_out(prep["ohe_cols"])
    ohe_dict = dict(zip(ohe_cols_out, ohe_out[0]))

    brand_info = prep["target_encode_maps"]["brand"]
    model_info = prep["target_encode_maps"]["model"]
    brand_te = brand_info["map"].get(inputs["brand"], brand_info["global_mean"])
    model_te = model_info["map"].get(inputs["model"], model_info["global_mean"])

    ext_freq = prep["ext_freq_map"].get(inputs["ext_col"], 0)
    int_freq = prep["int_freq_map"].get(inputs["int_col"], 0)

    row = {
        "model_year": inputs["model_year"],
        "milage": inputs["milage"],
        "accident": accident_enc,
        "clean_title": clean_title_enc,
        "milage_per_year": milage_per_year,
        "horsepower": inputs["horsepower"],
        "engine_size": inputs["engine_size"],
        "cylinder": inputs["cylinder"],
        "cylinder_missing": 0,
        "turbo": int(inputs["turbo"]),
        "hybrid": int(inputs["hybrid"]),
        "hp_per_liter": hp_per_liter,
        "hp_per_cylinder": hp_per_cylinder,
        "transmission_speed": inputs["transmission_speed"],
        "brand_te": brand_te,
        "model_te": model_te,
        "ext_col_freq": ext_freq,
        "int_col_freq": int_freq,
    }
    row.update(ohe_dict)

    feature_columns = meta["feature_columns"]
    X_row = pd.DataFrame([[row.get(c, 0) for c in feature_columns]], columns=feature_columns)

    log_pred = model.predict(X_row)[0]
    price = float(np.expm1(log_pred))
    return price, X_row


# SIDEBAR NAV
st.sidebar.markdown(
    f"""
    <div style="text-align:center; padding: 8px 0 18px 0;">
        <div style="font-size:2.2rem;">🚗</div>
        <div style="font-weight:800; font-size:1.15rem; color:{LIGHT};">Kaggle Playground Series S4E9</div>
        <div style="color:{MUTED}; font-size:0.78rem;">Regression of Used Car Prices</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigasi",
    [
        "🏠  Beranda",
        "📊  Jelajahi Data",
        "🧮  Prediksi Harga",
        "📈  Performa Model",
        "ℹ️  Tentang Proyek",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown(
    f"""
    <div class="metric-card">
        <div class="label">Model Aktif</div>
        <div class="value" style="font-size:1.1rem;">XGBoost (Tuned)</div>
    </div>
    <br>
    <div class="metric-card">
        <div class="label">Validasi R²</div>
        <div class="value">{metrics['r2_val']:.3f}</div>
    </div>
    <br>
    <div class="metric-card">
        <div class="label">MAE Validasi</div>
        <div class="value">${metrics['mae_val']:,.0f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# PAGE: BERANDA
if page.startswith("🏠"):
    st.markdown(
        f"""
        <div class="hero">
            <span class="badge badge-red">Kaggle Playground S4E9</span>
            <span class="badge badge-gold">Dataset Bernoise Tinggi</span>
            <span class="badge badge-green">XGBoost Tuned</span>
            <h1>Prediksi Harga Mobil Bekas</h1>
            <p>
                Dashboard ini membungkus pipeline data science lengkap, mulai dari pembersihan
                data yang noise (brand model salah label, harga placeholder), rekayasa fitur
                mesin & transmisi, hingga model XGBoost yang sudah di tuning ke dalam satu
                aplikasi interaktif untuk mengestimasi harga pasar mobil bekas
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Baris Data Terpakai", f"{metrics['n_rows_used']:,}", "setelah pembersihan noise"),
        ("Baris Noise Dibuang", f"{metrics['n_outlier_dropped']:,}", "mislabel + harga placeholder"),
        ("R² Validasi", f"{metrics['r2_val']:.3f}", "XGBoost tuned"),
        ("MAE Validasi", f"${metrics['mae_val']:,.0f}", "rata-rata selisih prediksi"),
    ]
    for col, (label, val, sub) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div style="color:{MUTED}; font-size:0.78rem; margin-top:4px;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("### 🛠️ Alur Pipeline")
        steps = [
            ("1. Cleaning Noise", "Hapus baris brand model salah label & harga placeholder (price ratio vs median grup brand+model)"),
            ("2. Feature Engineering", "Ekstraksi horsepower, kapasitas mesin, silinder, turbo, tipe & kecepatan transmisi dari teks mentah"),
            ("3. Encoding", "Target encoding (OOF) untuk brand & model, frequency encoding untuk warna, one-hot untuk fuel & transmisi"),
            ("4. Modeling", "Linear/Ridge → XGBoost, LightGBM, CatBoost dibandingkan via 5 fold CV, lalu di tuning dengan GridSearchCV"),
        ]
        for title, desc in steps:
            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom:10px;">
                    <div style="color:{GOLD}; font-weight:700; margin-bottom:4px;">{title}</div>
                    <div style="color:{MUTED}; font-size:0.88rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        eda = load_eda_data()
        top_brands = eda["brand"].value_counts().head(8).reset_index()
        top_brands.columns = ["brand", "count"]
        fig = px.bar(
            top_brands.sort_values("count"),
            x="count", y="brand", orientation="h",
            color="count", color_continuous_scale=SEQ_SCALE,
            template=PLOTLY_TEMPLATE,
            title="Brand Paling Banyak di Dataset",
        )
        fig.update_layout(
            coloraxis_showscale=False, height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.info("👉 Buka tab **Prediksi Harga** di sidebar untuk mengestimasi harga mobil bekas secara interaktif.")

# PAGE: JELAJAHI DATA (EDA)
elif page.startswith("📊"):
    st.markdown("## 📊 Jelajahi Data")
    st.caption("Eksplorasi data setelah proses cleaning noise (sampel 30.000 baris untuk performa render)")

    eda = load_eda_data()

    t1, t2, t3, t4 = st.tabs(["Distribusi", "Korelasi Numerik", "Kategorikal", "Brand Explorer"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                eda, x="price", nbins=60, color_discrete_sequence=[RED],
                template=PLOTLY_TEMPLATE, title="Distribusi Harga Mobil ($)",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.histogram(
                eda, x="milage", nbins=60, color_discrete_sequence=[GOLD],
                template=PLOTLY_TEMPLATE, title="Distribusi Jarak Tempuh (mil)",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            fig = px.histogram(
                eda, x="model_year", nbins=40, color_discrete_sequence=["#4D96FF"],
                template=PLOTLY_TEMPLATE, title="Distribusi Tahun Produksi",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, width="stretch")
        with c4:
            fig = px.histogram(
                eda, x="horsepower", nbins=50, color_discrete_sequence=[GREEN],
                template=PLOTLY_TEMPLATE, title="Distribusi Tenaga Mesin (HP)",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, width="stretch")

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                eda.sample(min(6000, len(eda)), random_state=1),
                x="milage", y="price", opacity=0.35,
                color_discrete_sequence=[RED],
                template=PLOTLY_TEMPLATE, title="Harga vs Jarak Tempuh",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.scatter(
                eda.sample(min(6000, len(eda)), random_state=1),
                x="model_year", y="price", opacity=0.35,
                color_discrete_sequence=[GOLD],
                template=PLOTLY_TEMPLATE, title="Harga vs Tahun Produksi",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
            st.plotly_chart(fig, width="stretch")

        num_cols = ["model_year", "milage", "horsepower", "engine_size", "cylinder", "transmission_speed", "price"]
        corr = eda[num_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale=SEQ_SCALE,
            template=PLOTLY_TEMPLATE, title="Matriks Korelasi Fitur Numerik",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=480)
        st.plotly_chart(fig, width="stretch")

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            vc = eda["fuel_type"].value_counts().reset_index()
            vc.columns = ["fuel_type", "count"]
            fig = px.pie(
                vc, names="fuel_type", values="count", hole=0.5,
                color_discrete_sequence=CATEGORICAL_SCALE,
                template=PLOTLY_TEMPLATE, title="Komposisi Jenis Bahan Bakar",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, width="stretch")
        with c2:
            vc = eda["accident"].value_counts().reset_index()
            vc.columns = ["accident", "count"]
            fig = px.bar(
                vc, x="accident", y="count", color="accident",
                color_discrete_sequence=[GREEN, RED],
                template=PLOTLY_TEMPLATE, title="Riwayat Kecelakaan",
            )
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, width="stretch")

        avg_price_fuel = eda.groupby("fuel_type")["price"].median().sort_values(ascending=False).reset_index()
        fig = px.bar(
            avg_price_fuel, x="fuel_type", y="price",
            color="price", color_continuous_scale=SEQ_SCALE,
            template=PLOTLY_TEMPLATE, title="Median Harga per Jenis Bahan Bakar",
        )
        fig.update_layout(coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, width="stretch")

    with t4:
        sel_brand = st.selectbox("Pilih Brand", sorted(eda["brand"].unique()), index=0)
        sub = eda[eda["brand"] == sel_brand]
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="label">Jumlah Unit</div><div class="value">{len(sub):,}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="label">Median Harga</div><div class="value">${sub["price"].median():,.0f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="label">Median Tahun</div><div class="value">{int(sub["model_year"].median())}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        top_models = sub["model"].value_counts().head(10).reset_index()
        top_models.columns = ["model", "count"]
        fig = px.bar(
            top_models.sort_values("count"), x="count", y="model", orientation="h",
            color="count", color_continuous_scale=SEQ_SCALE,
            template=PLOTLY_TEMPLATE, title=f"Model {sel_brand} Terbanyak",
        )
        fig.update_layout(coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420)
        st.plotly_chart(fig, width="stretch")

# PAGE: PREDIKSI HARGA
elif page.startswith("🧮"):
    st.markdown("## 🧮 Prediksi Harga Mobil Bekas")
    st.caption("Isi spesifikasi mobil di bawah ini. Geser slider untuk parameter numerik, lalu klik **Prediksi**")

    form_col, result_col = st.columns([1.3, 1])

    with form_col:
        st.markdown("#### Identitas Mobil")
        c1, c2 = st.columns(2)
        with c1:
            brand = st.selectbox("Brand", options["brand_list"], index=options["brand_list"].index("Toyota") if "Toyota" in options["brand_list"] else 0)
        with c2:
            model_list = options["brand_model_map"].get(brand, [])
            model_name = st.selectbox("Model", model_list, index=0)

        c3, c4 = st.columns(2)
        with c3:
            fuel_type = st.selectbox("Jenis Bahan Bakar", options["fuel_type_options"],
                                      index=options["fuel_type_options"].index("Gasoline") if "Gasoline" in options["fuel_type_options"] else 0)
        with c4:
            transmission_type = st.selectbox("Tipe Transmisi", options["transmission_type_options"],
                                              index=options["transmission_type_options"].index("Automatic") if "Automatic" in options["transmission_type_options"] else 0)

        st.markdown("#### Spesifikasi Teknis")
        model_year = st.slider(
            "Tahun Produksi", min_value=ranges["model_year"][0], max_value=meta["current_year"],
            value=2018, step=1,
        )
        milage = st.slider(
            "Jarak Tempuh (mil)", min_value=0, max_value=int(ranges["milage"][1]),
            value=45000, step=500,
        )
        horsepower = st.slider(
            "Tenaga Mesin (HP)", min_value=float(ranges["horsepower"][0]), max_value=float(ranges["horsepower"][1]),
            value=float(medians["horsepower"]), step=5.0,
        )
        engine_size = st.slider(
            "Kapasitas Mesin (Liter)", min_value=float(ranges["engine_size"][0]), max_value=float(ranges["engine_size"][1]),
            value=float(medians["engine_size"]), step=0.1,
        )
        cylinder = st.slider(
            "Jumlah Silinder", min_value=int(ranges["cylinder"][0]), max_value=int(ranges["cylinder"][1]),
            value=int(medians["cylinder"]), step=1,
        )
        transmission_speed = st.select_slider(
            "Jumlah Percepatan Transmisi (0 = tidak diketahui/CVT)",
            options=[int(x) for x in options["transmission_speed_options"]],
            value=6,
        )

        c5, c6 = st.columns(2)
        with c5:
            turbo = st.checkbox("Dilengkapi Turbo", value=False)
        with c6:
            hybrid_default = fuel_type in ("Hybrid", "Plug-In Hybrid")
            hybrid = st.checkbox("Hybrid / Electric", value=hybrid_default)

        st.markdown("#### Kondisi & Tampilan")
        c7, c8 = st.columns(2)
        with c7:
            accident = st.selectbox("Riwayat Kecelakaan", options["accident_options"])
        with c8:
            clean_title = st.selectbox("Status Clean Title", options["clean_title_options"],
                                        index=options["clean_title_options"].index("Yes") if "Yes" in options["clean_title_options"] else 0)

        c9, c10 = st.columns(2)
        with c9:
            ext_col = st.selectbox("Warna Eksterior", options["ext_col_options"])
        with c10:
            int_col = st.selectbox("Warna Interior", options["int_col_options"])

        predict_clicked = st.button("🔮 Prediksi Harga", width="stretch")

    with result_col:
        if predict_clicked:
            inputs = dict(
                brand=brand, model=model_name, fuel_type=fuel_type, transmission_type=transmission_type,
                model_year=model_year, milage=milage, horsepower=horsepower, engine_size=engine_size,
                cylinder=cylinder, transmission_speed=transmission_speed, turbo=turbo, hybrid=hybrid,
                accident=accident, clean_title=clean_title, ext_col=ext_col, int_col=int_col,
            )
            price, X_row = predict_price(inputs)
            mae = metrics["mae_val"]
            low, high = max(0, price - mae), price + mae

            st.markdown(
                f"""
                <div class="price-result">
                    <div style="color:{MUTED}; text-transform:uppercase; letter-spacing:1px; font-size:0.8rem;">Estimasi Harga Pasar</div>
                    <div class="price">${price:,.0f}</div>
                    <div class="range">Kisaran wajar: ${low:,.0f} – ${high:,.0f} (± MAE model)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**{brand} {model_name}** · {int(model_year)} · {milage:,.0f} mil")

            eda = load_eda_data()
            comp = eda[(eda["brand"] == brand)]
            if len(comp) > 0:
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=comp["price"], name=f"Harga pasar {brand}",
                    marker_color=GOLD, boxmean=True,
                ))
                fig.add_trace(go.Scatter(
                    x=[f"Harga pasar {brand}"], y=[price],
                    mode="markers", marker=dict(size=16, color=RED, symbol="diamond"),
                    name="Prediksi Anda",
                ))
                fig.update_layout(
                    template=PLOTLY_TEMPLATE, title=f"Posisi Prediksi vs Sebaran Harga {brand}",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360,
                    showlegend=True,
                )
                st.plotly_chart(fig, width="stretch")

            similar = eda[(eda["brand"] == brand) & (eda["model"] == model_name)]
            if len(similar) > 0:
                similar = similar.assign(diff_year=(similar["model_year"] - model_year).abs())
                similar = similar.sort_values("diff_year").head(5)
                show_cols = ["model_year", "milage", "fuel_type", "transmission_type", "accident", "price"]
                st.markdown("##### 🔎 Listing Mirip di Dataset")
                st.dataframe(
                    similar[show_cols].rename(columns={
                        "model_year": "Tahun", "milage": "Mil", "fuel_type": "Bahan Bakar",
                        "transmission_type": "Transmisi", "accident": "Kecelakaan", "price": "Harga ($)",
                    }),
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("Tidak ada listing identik (brand+model) pada sampel EDA untuk dibandingkan.")
        else:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center; padding:40px 20px;">
                    <div style="font-size:2.4rem;">🚘</div>
                    <div style="color:{MUTED}; margin-top:10px;">
                        Lengkapi spesifikasi mobil di sebelah kiri, lalu klik tombol
                        <b>Prediksi Harga</b> untuk melihat estimasi.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# PAGE: PERFORMA MODEL
elif page.startswith("📈"):
    st.markdown("## 📈 Performa Model")
    st.caption("Model final: XGBoost (tuned) dipilih dari 4 kandidat algoritma berdasarkan 5 fold cross validation.")

    c1, c2, c3, c4 = st.columns(4)
    perf_cards = [
        ("R² Training", f"{metrics['r2_train']:.4f}"),
        ("R² Validasi", f"{metrics['r2_val']:.4f}"),
        ("RMSE Validasi", f"${metrics['rmse_val']:,.0f}"),
        ("MAE Validasi", f"${metrics['mae_val']:,.0f}"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], perf_cards):
        col.markdown(
            f'<div class="metric-card"><div class="label">{label}</div><div class="value">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Actual vs Predicted")
        val_pred = load_val_predictions()
        fig = px.scatter(
            val_pred, x="actual", y="predicted", opacity=0.35,
            color_discrete_sequence=[RED], template=PLOTLY_TEMPLATE,
        )
        max_val = max(val_pred["actual"].max(), val_pred["predicted"].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            line=dict(color=GOLD, dash="dash"), name="Prediksi Sempurna",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420,
            xaxis_title="Harga Aktual ($)", yaxis_title="Harga Prediksi ($)",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("#### Feature Importance (Top 12)")
        fi = load_feature_importance().head(12).iloc[::-1]
        fi.index = [friendly(i) for i in fi.index]
        fig = px.bar(
            fi, x="importance", y=fi.index, orientation="h",
            color="importance", color_continuous_scale=SEQ_SCALE,
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=420, xaxis_title="Importance", yaxis_title="",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Perbandingan Algoritma (5 Fold CV, dari notebook eksperimen)")
    # Baris XGBoost (Tuned) dibaca dari metadata agar selalu sinkron
    # dengan metric card R²/RMSE/MAE di atas — tidak hardcode lagi
    comparison = pd.DataFrame([
        {"Model": "XGBoost (Tuned) ✓", "Val R²": metrics["r2_val"], "Val RMSE ($)": metrics["rmse_val"]},
        {"Model": "LightGBM (Tuned)", "Val R²": 0.7210, "Val RMSE ($)": 24969},
        {"Model": "CatBoost", "Val R²": 0.7199, "Val RMSE ($)": 25037},
        {"Model": "CatBoost (Tuned)", "Val R²": 0.7198, "Val RMSE ($)": 24993},
        {"Model": "LightGBM", "Val R²": 0.7172, "Val RMSE ($)": 25079},
        {"Model": "XGBoost (Default)", "Val R²": 0.7167, "Val RMSE ($)": 25065},
        {"Model": "Ridge (Tuned)", "Val R²": 0.6778, "Val RMSE ($)": 26840},
        {"Model": "Linear Regression", "Val R²": 0.6778, "Val RMSE ($)": 26840},
    ])
    fig = px.bar(
        comparison.sort_values("Val R²"), x="Val R²", y="Model", orientation="h",
        color="Val R²", color_continuous_scale=SEQ_SCALE, template=PLOTLY_TEMPLATE,
        text="Val R²",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, xaxis_range=[0.6, 0.78],
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "XGBoost (tuned) terpilih sebagai model final karena Val R² tertinggi dan RMSE terendah "
        "di antara seluruh kandidat yang diuji (Linear/Ridge, XGBoost, LightGBM, CatBoost)"
    )

# PAGE: TENTANG PROYEK
else:
    st.markdown("## ℹ️ Tentang Proyek")

    st.markdown(
        """
        ### Latar Belakang
        Dataset ini berasal dari **Kaggle Playground Series Season 4, Episode 9
        (Used Car Price Prediction)**, dataset sintetis yang terkenal sangat berisik
        (*noisy*): banyak kombinasi brand model yang tidak masuk akal, serta sejumlah
        baris dengan harga yang jelas-jelas *placeholder* (misalnya brand non-mewah
        dihargai jutaan dolar). Tantangan utamanya bukan sekadar memilih algoritma,
        tapi membersihkan noise tanpa membuang sinyal yang berguna
        """
    )

    st.markdown("### 🧹 Strategi Penanganan Noise")
    n1, n2 = st.columns(2)
    with n1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color:{GOLD}; font-weight:700;">1. Brand Model Mislabel</div>
                <p style="color:{MUTED}; font-size:0.88rem;">
                Untuk setiap nama model, brand yang paling sering muncul dianggap sebagai
                brand "benar" (majority vote). Baris dengan kombinasi brand model yang
                menyimpang dari mayoritas (mis. "Ford GT-R", GT-R sebenarnya model Nissan)
                dianggap mislabel dan dibuang
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with n2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color:{GOLD}; font-weight:700;">2. Harga Placeholder</div>
                <p style="color:{MUTED}; font-size:0.88rem;">
                Untuk setiap kombinasi brand+model, dihitung rasio harga terhadap median
                harga grup tersebut. Baris dengan rasio &gt; 5x atau &lt; 0.2x median dianggap
                outlier harga (placeholder/typo) dan dibuang, mengikuti prinsip
                "garbage labels are worse than missing features"
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ⚙️ Rekayasa Fitur")
    st.markdown(
        """
        - **Mesin**: ekstraksi `horsepower`, `engine_size` (liter), `cylinder` dari teks
          mentah kolom *engine* via regex; ditambah indikator `turbo` dan `hybrid`,
          serta rasio `hp_per_liter` dan `hp_per_cylinder`
        - **Transmisi**: klasifikasi ke `Automatic / Manual / CVT / DCT / Unknown`, plus
          ekstraksi jumlah percepatan (`transmission_speed`)
        - **Usia & Pemakaian**: `milage_per_year` sebagai proxy intensitas pemakaian
          (lebih informatif daripada mileage mentah)
        - **Encoding**: *out-of-fold target encoding* (dengan smoothing) untuk `brand`
          dan `model` agar tidak bocor (*leakage*) ke data validasi; *frequency encoding*
          untuk warna eksterior/interior yang sangat bervariasi; one-hot encoding untuk
          `fuel_type` dan `transmission_type`
        """
    )

    st.markdown("### 🤖 Model & Evaluasi")
    st.markdown(
        f"""
        Empat kandidat algoritma dibandingkan via 5-fold cross-validation: Linear/Ridge
        Regression sebagai baseline, lalu tree-based ensemble (XGBoost, LightGBM,
        CatBoost). **XGBoost** dipilih sebagai model final setelah hyperparameter tuning
        (GridSearchCV) mencapai **Val R² = {metrics['r2_val']:.4f}** dan **MAE = ${metrics['mae_val']:,.0f}**
        pada data validasi (target di-log-transform dengan `log1p` untuk menstabilkan
        distribusi harga yang sangat skewed)
        """
    )

    st.markdown(
        f"""
        <div class="section-divider"></div>
        <p style="color:{MUTED}; font-size:0.85rem;">
        Dibangun dengan Streamlit · XGBoost · scikit-learn · Plotly &nbsp;|&nbsp;
        Dataset: Kaggle Playground Series S4E9
        </p>
        """,
        unsafe_allow_html=True,
    )