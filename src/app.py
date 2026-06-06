import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LR_MODEL_PATH = os.path.join(BASE_DIR, "models", "lr_model.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
POSITION_LR_MODELS_PATH = os.path.join(BASE_DIR, "models", "position_lr_models.pkl")
POSITION_RF_MODELS_PATH = os.path.join(BASE_DIR, "models", "position_rf_models.pkl")
POSITION_FEATURES_PATH = os.path.join(BASE_DIR, "models", "position_feature_map.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "merged.csv")

NUMERIC_COLS = [
    "age", "height_in_cm",
    "total_goals", "total_assists", "total_matches",
    "total_yellow_cards", "total_red_cards",
    "goals_per_match", "assists_per_match", "minutes_per_match"
]

CATEGORICAL_COLS = [
    "position",
    "sub_position",
    "foot",
    "country_of_citizenship",
    "player_club_domestic_competition_id"
]

st.set_page_config(page_title="Futbolcu Piyasa Değeri Tahmini", layout="centered")
st.title("Futbolcu Piyasa Değeri Tahmini")
st.write("Futbolcu bilgilerine göre tahmini piyasa değerini hesaplar.")


@st.cache_resource
def load_artifacts():
    lr_model = joblib.load(LR_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    position_lr_models = joblib.load(POSITION_LR_MODELS_PATH)
    position_rf_models = joblib.load(POSITION_RF_MODELS_PATH)
    position_features = joblib.load(POSITION_FEATURES_PATH)

    return lr_model, rf_model, feature_columns, encoders, scaler, position_lr_models, position_rf_models, position_features


@st.cache_data
def load_dropdown_options():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["total_goals", "total_assists", "total_minutes", "age"])
    df["foot"] = df["foot"].fillna("unknown")
    df["height_in_cm"] = df["height_in_cm"].fillna(df["height_in_cm"].mean())
    df["sub_position"] = df["sub_position"].fillna(df["position"])
    df["country_of_citizenship"] = df["country_of_citizenship"].fillna("unknown")
    df["player_club_domestic_competition_id"] = df["player_club_domestic_competition_id"].fillna("unknown")
    df = df[df["position"] != "Missing"]
    df = df[df["height_in_cm"] >= 150]
    df = df[(df["age"] >= 15) & (df["age"] < 38)]
    df = df[df["market_value_in_eur"] >= 100_000]
    return {col: sorted(df[col].astype(str).unique()) for col in CATEGORICAL_COLS}


lr_model, rf_model, feature_columns, encoders, scaler, position_lr_models, position_rf_models, position_features = load_artifacts()
options = load_dropdown_options()

st.subheader("Futbolcu Bilgileri")
st.caption("Lütfen futbolcunun kariyer istatistiklerini giriniz.")

position = st.selectbox("Mevki", options["position"])
sub_position = st.selectbox("Alt Mevki", options["sub_position"])
foot = st.selectbox("Ayak", options["foot"])
country = st.selectbox("Ülke", options["country_of_citizenship"])
competition = st.selectbox("Lig", options["player_club_domestic_competition_id"])

age = st.number_input("Yaş", min_value=15, max_value=37, value=24)
height = st.number_input("Boy (cm)", min_value=150, max_value=220, value=180)

total_matches = st.number_input("Toplam Maç", min_value=1, max_value=800, value=50)
total_goals = st.number_input("Toplam Gol", min_value=0, max_value=500, value=10)
total_assists = st.number_input("Toplam Asist", min_value=0, max_value=500, value=5)
minutes_per_match = st.number_input("Maç Başına Dakika", min_value=0.0, max_value=120.0, value=75.0)
total_yellow_cards = st.number_input("Sarı Kart", min_value=0, max_value=100, value=3)
total_red_cards = st.number_input("Kırmızı Kart", min_value=0, max_value=20, value=0)

if st.button("Tahmin Et"):
    goals_per_match = total_goals / total_matches
    assists_per_match = total_assists / total_matches

    raw_input = pd.DataFrame([{
        "age": age,
        "height_in_cm": height,
        "total_goals": total_goals,
        "total_assists": total_assists,
        "total_matches": total_matches,
        "total_yellow_cards": total_yellow_cards,
        "total_red_cards": total_red_cards,
        "goals_per_match": goals_per_match,
        "assists_per_match": assists_per_match,
        "minutes_per_match": minutes_per_match
    }])

    scaled = raw_input.copy()
    scaled[NUMERIC_COLS] = scaler.transform(raw_input[NUMERIC_COLS])

    scaled["position_encoded"] = encoders["position"].transform([position])[0]
    scaled["sub_position_encoded"] = encoders["sub_position"].transform([sub_position])[0]
    scaled["foot_encoded"] = encoders["foot"].transform([foot])[0]
    scaled["country_of_citizenship_encoded"] = encoders["country_of_citizenship"].transform([country])[0]
    scaled["player_club_domestic_competition_id_encoded"] = encoders["player_club_domestic_competition_id"].transform([competition])[0]

    for col in feature_columns:
        if col not in scaled.columns:
            scaled[col] = 0
    input_general = scaled[feature_columns]

    lr_log_pred = lr_model.predict(input_general)[0]
    rf_log_pred = rf_model.predict(input_general)[0]
    lr_prediction = np.expm1(lr_log_pred)
    rf_prediction = np.expm1(rf_log_pred)
    avg_prediction = (lr_prediction + rf_prediction) / 2

    pos_features = position_features.get(position, [])
    pos_lr_prediction = None
    pos_rf_prediction = None

    if pos_features and position in position_lr_models:
        input_pos = scaled[[f for f in pos_features if f in scaled.columns]]
        for col in pos_features:
            if col not in input_pos.columns:
                input_pos[col] = 0
        input_pos = input_pos[pos_features]

        pos_lr_prediction = np.expm1(position_lr_models[position].predict(input_pos)[0])
        pos_rf_prediction = np.expm1(position_rf_models[position].predict(input_pos)[0])

    st.subheader("Genel Model Sonuçları")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Linear Regression", f"€{lr_prediction:,.0f}")
    with col2:
        st.metric("Random Forest", f"€{rf_prediction:,.0f}")
    with col3:
        st.metric("Ortalama", f"€{avg_prediction:,.0f}")

    if pos_lr_prediction is not None:
        st.subheader(f"Mevki Bazlı Model Sonuçları ({position})")
        col4, col5 = st.columns(2)
        with col4:
            st.metric("LR (Mevki)", f"€{pos_lr_prediction:,.0f}")
        with col5:
            st.metric("RF (Mevki)", f"€{pos_rf_prediction:,.0f}")

    with st.expander("Modele gönderilen veri"):
        st.dataframe(input_general)

st.info("İstatistikler 22 Mart 2026 tarihine kadar olan maçları kapsamaktadır.")