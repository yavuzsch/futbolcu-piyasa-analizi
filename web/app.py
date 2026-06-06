import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "merged.csv")

st.set_page_config(
    page_title="Futbolcu Piyasa Değeri Tahmini",
    layout="centered"
)

st.title("Futbolcu Piyasa Değeri Tahmini")
st.write("Futbolcu bilgilerine göre tahmini piyasa değerini hesaplar.")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(COLUMNS_PATH)

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["total_goals", "total_assists", "total_minutes", "age"])

df["foot"] = df["foot"].fillna("unknown")
df["height_in_cm"] = df["height_in_cm"].fillna(df["height_in_cm"].mean())
df["sub_position"] = df["sub_position"].fillna(df["position"])
df["country_of_citizenship"] = df["country_of_citizenship"].fillna("unknown")
df["player_club_domestic_competition_id"] = df["player_club_domestic_competition_id"].fillna("unknown")

df = df[df["position"] != "Missing"]
df = df[df["height_in_cm"] >= 150]
df = df[(df["age"] >= 15) & (df["age"] <= 45)]
df = df[df["age"] < 38]
df = df[df["market_value_in_eur"] >= 100_000]

categorical_cols = [
    "position",
    "sub_position",
    "foot",
    "country_of_citizenship",
    "player_club_domestic_competition_id"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    encoders[col] = le

numeric_cols = [
    "age",
    "height_in_cm",
    "total_goals",
    "total_assists",
    "total_matches",
    "total_yellow_cards",
    "total_red_cards",
    "goals_per_match",
    "assists_per_match",
    "minutes_per_match"
]

scaler = StandardScaler()
scaler.fit(df[numeric_cols])

st.subheader("Futbolcu Bilgileri")

position = st.selectbox("Mevki", sorted(df["position"].astype(str).unique()))
sub_position = st.selectbox("Alt Mevki", sorted(df["sub_position"].astype(str).unique()))
foot = st.selectbox("Ayak", sorted(df["foot"].astype(str).unique()))
country = st.selectbox("Ülke", sorted(df["country_of_citizenship"].astype(str).unique()))
competition = st.selectbox("Lig", sorted(df["player_club_domestic_competition_id"].astype(str).unique()))

age = st.number_input("Yaş", min_value=15, max_value=37, value=24)
height = st.number_input("Boy (cm)", min_value=150, max_value=220, value=180)

total_matches = st.number_input("Toplam Maç", min_value=1, max_value=800, value=50)
total_goals = st.number_input("Toplam Gol", min_value=0, max_value=500, value=10)
total_assists = st.number_input("Toplam Asist", min_value=0, max_value=500, value=5)

minutes_per_match = st.number_input("Maç Başına Dakika", min_value=0.0, max_value=120.0, value=75.0)

total_yellow_cards = st.number_input("Sarı Kart", min_value=0, max_value=100, value=3)
total_red_cards = st.number_input("Kırmızı Kart", min_value=0, max_value=20, value=0)

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

scaled_input = raw_input.copy()
scaled_input[numeric_cols] = scaler.transform(raw_input[numeric_cols])

input_data = scaled_input.copy()

input_data["position"] = position
input_data["position_encoded"] = encoders["position"].transform([position])[0]
input_data["sub_position_encoded"] = encoders["sub_position"].transform([sub_position])[0]
input_data["foot_encoded"] = encoders["foot"].transform([foot])[0]
input_data["country_of_citizenship_encoded"] = encoders["country_of_citizenship"].transform([country])[0]
input_data["player_club_domestic_competition_id_encoded"] = encoders["player_club_domestic_competition_id"].transform([competition])[0]

for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[feature_columns]

if st.button("Tahmin Et"):
    log_prediction = model.predict(input_data)[0]
    prediction = np.expm1(log_prediction)

    st.success(f"Tahmini Piyasa Değeri: €{prediction:,.0f}")

    st.write("Modele gönderilen veri:")
    st.dataframe(input_data)