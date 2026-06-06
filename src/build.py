import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CATEGORICAL_COLS = [
    "position",
    "sub_position",
    "foot",
    "country_of_citizenship",
    "player_club_domestic_competition_id"
]

NUMERIC_COLS = [
    "age", "height_in_cm",
    "total_goals", "total_assists", "total_matches",
    "total_yellow_cards", "total_red_cards",
    "goals_per_match", "assists_per_match", "minutes_per_match"
]

position_features = {
    "Goalkeeper": [
        "age", "height_in_cm", "total_matches", "minutes_per_match",
        "total_yellow_cards", "total_red_cards",
        "position_encoded", "foot_encoded",
        "country_of_citizenship_encoded", "player_club_domestic_competition_id_encoded"
    ],
    "Defender": [
        "age", "height_in_cm", "total_matches", "minutes_per_match",
        "total_yellow_cards", "total_red_cards",
        "position_encoded", "sub_position_encoded", "foot_encoded",
        "country_of_citizenship_encoded", "player_club_domestic_competition_id_encoded"
    ],
    "Midfield": [
        "age", "total_goals", "total_assists", "total_matches",
        "goals_per_match", "assists_per_match", "minutes_per_match",
        "total_yellow_cards", "total_red_cards",
        "position_encoded", "sub_position_encoded", "foot_encoded",
        "country_of_citizenship_encoded", "player_club_domestic_competition_id_encoded"
    ],
    "Attack": [
        "age", "total_goals", "total_assists", "total_matches",
        "goals_per_match", "assists_per_match", "minutes_per_match",
        "total_yellow_cards", "total_red_cards",
        "sub_position_encoded", "foot_encoded",
        "country_of_citizenship_encoded", "player_club_domestic_competition_id_encoded"
    ]
}


def load_data(path):
    df = pd.read_csv(path)
    print(f"Veri yüklendi: {df.shape}")
    return df


def build_encoders_and_scaler(df):
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_COLS])

    return encoders, scaler


def split_features(df, target="log_market_value"):
    X = df.drop(columns=[target, "position"])
    y = df[target]
    print(f"Girdi boyutu: {X.shape}")
    print(f"Çıktı boyutu: {y.shape}")
    return X, y


def train_test(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return lr, rf


def evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} — MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return y_pred, {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


def cross_validate(lr, rf, X_train, y_train, cv=5):
    cv_rmse_lr = np.sqrt(-cross_val_score(
        lr, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
    ))
    cv_rmse_rf = np.sqrt(-cross_val_score(
        rf, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
    ))

    print(f"Linear Regression CV RMSE: {cv_rmse_lr.mean():.4f} ± {cv_rmse_lr.std():.4f}")
    print(f"Random Forest CV RMSE: {cv_rmse_rf.mean():.4f} ± {cv_rmse_rf.std():.4f}")

    return cv_rmse_lr, cv_rmse_rf


def train_position_models(df):
    position_results = []
    position_models = {}

    for position in sorted(df["position"].unique()):
        df_pos = df[df["position"] == position].copy()
        X_pos = df_pos[position_features[position]]
        y_pos = df_pos["log_market_value"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_pos, y_pos, test_size=0.2, random_state=42
        )

        lr_pos = LinearRegression()
        lr_pos.fit(X_tr, y_tr)
        y_pred_lr_pos = lr_pos.predict(X_te)
        cv_lr_pos = np.sqrt(-cross_val_score(
            lr_pos, X_tr, y_tr, cv=5, scoring="neg_mean_squared_error"
        ))

        rf_pos = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_pos.fit(X_tr, y_tr)
        y_pred_rf_pos = rf_pos.predict(X_te)
        cv_rf_pos = np.sqrt(-cross_val_score(
            rf_pos, X_tr, y_tr, cv=5, scoring="neg_mean_squared_error"
        ))

        position_results.append({
            "Mevki": position,
            "LR RMSE": np.sqrt(mean_squared_error(y_te, y_pred_lr_pos)),
            "RF RMSE": np.sqrt(mean_squared_error(y_te, y_pred_rf_pos)),
            "LR MAE": mean_absolute_error(y_te, y_pred_lr_pos),
            "RF MAE": mean_absolute_error(y_te, y_pred_rf_pos),
            "LR R²": r2_score(y_te, y_pred_lr_pos),
            "RF R²": r2_score(y_te, y_pred_rf_pos),
            "LR CV RMSE": cv_lr_pos.mean(),
            "RF CV RMSE": cv_rf_pos.mean()
        })
        position_models[position] = {"lr": lr_pos, "rf": rf_pos}
        print(f"{position:12} | LR R²: {position_results[-1]['LR R²']:.4f} | RF R²: {position_results[-1]['RF R²']:.4f}")

    position_df = pd.DataFrame(position_results)
    print()
    print(position_df.to_string(index=False))
    return position_models, position_df


def print_position_predictions(df, position_models):
    for position in sorted(df["position"].unique()):
        df_pos_all = df[df["position"] == position]
        X_pos_all = df_pos_all[position_features[position]]
        y_pos_all = df_pos_all["log_market_value"]

        _, X_te_pos, _, y_te_pos = train_test_split(
            X_pos_all, y_pos_all, test_size=0.2, random_state=42
        )

        lr_pos_sample = position_models[position]["lr"]
        rf_pos_sample = position_models[position]["rf"]

        pred_lr_eur = np.expm1(lr_pos_sample.predict(X_te_pos[:3]))
        pred_rf_eur = np.expm1(rf_pos_sample.predict(X_te_pos[:3]))
        actual_eur = np.expm1(y_te_pos.values[:3])

        sample_pos = pd.DataFrame({
            "Gerçek (€)": actual_eur,
            "LR Tahmini (€)": pred_lr_eur,
            "RF Tahmini (€)": pred_rf_eur
        }).map(lambda x: f"{x:,.0f}")
        print(f"\n{position}:")
        print(sample_pos.to_string(index=False))


def print_predictions(y_test, y_pred_lr, y_pred_rf, n=10):
    y_test_eur = np.expm1(y_test)
    y_pred_lr_eur = np.expm1(y_pred_lr)
    y_pred_rf_eur = np.expm1(y_pred_rf)

    sample = pd.DataFrame({
        "Gerçek (€)": y_test_eur.values[:n],
        "LR Tahmini (€)": y_pred_lr_eur[:n],
        "RF Tahmini (€)": y_pred_rf_eur[:n]
    })
    sample = sample.map(lambda x: f"{x:,.0f}")
    print("\nÖrnek Tahminler:")
    print(sample.to_string(index=False))


def main():
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    merged_path = os.path.join(BASE_DIR, "data", "processed", "merged.csv")
    df_raw = pd.read_csv(merged_path)

    df_raw = df_raw.dropna(subset=["total_goals", "total_assists", "total_minutes", "age"])
    df_raw["foot"] = df_raw["foot"].fillna("unknown")
    df_raw["height_in_cm"] = df_raw["height_in_cm"].fillna(df_raw["height_in_cm"].mean())
    df_raw["sub_position"] = df_raw["sub_position"].fillna(df_raw["position"])
    df_raw["country_of_citizenship"] = df_raw["country_of_citizenship"].fillna("unknown")
    df_raw["player_club_domestic_competition_id"] = df_raw["player_club_domestic_competition_id"].fillna("unknown")
    df_raw = df_raw[df_raw["position"] != "Missing"]
    df_raw = df_raw[df_raw["height_in_cm"] >= 150]
    df_raw = df_raw[df_raw["age"] < 38]
    df_raw = df_raw[df_raw["market_value_in_eur"] >= 100_000]

    encoders, scaler = build_encoders_and_scaler(df_raw)
    joblib.dump(encoders, os.path.join(models_dir, "encoders.pkl"))
    joblib.dump(scaler,   os.path.join(models_dir, "scaler.pkl"))
    print("Encoder ve scaler kaydedildi.")

    df = load_data(os.path.join(BASE_DIR, "data", "processed", "processed.csv"))
    X, y = split_features(df)
    X_train, X_test, y_train, y_test = train_test(X, y)

    lr, rf = train_models(X_train, y_train)

    y_pred_lr, metrics_lr = evaluate(lr, X_test, y_test, "Linear Regression")
    y_pred_rf, metrics_rf = evaluate(rf, X_test, y_test, "Random Forest")

    cv_rmse_lr, cv_rmse_rf = cross_validate(lr, rf, X_train, y_train)

    results = pd.DataFrame({
        "Model": [metrics_lr["model"], metrics_rf["model"]],
        "MAE": [metrics_lr["MAE"], metrics_rf["MAE"]],
        "RMSE": [metrics_lr["RMSE"], metrics_rf["RMSE"]],
        "R2": [metrics_lr["R2"], metrics_rf["R2"]],
        "CV RMSE": [cv_rmse_lr.mean(), cv_rmse_rf.mean()],
        "CV Std": [cv_rmse_lr.std(), cv_rmse_rf.std()]
    })
    print("\nModel Karşılaştırması:")
    print(results.to_string(index=False))

    print_predictions(y_test, y_pred_lr, y_pred_rf)

    position_models, _ = train_position_models(df)
    print_position_predictions(df, position_models)

    joblib.dump(lr, os.path.join(models_dir, "lr_model.pkl"))
    joblib.dump(rf, os.path.join(models_dir, "rf_model.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, "feature_columns.pkl"))

    position_lr_models = {pos: models["lr"] for pos, models in position_models.items()}
    position_rf_models = {pos: models["rf"] for pos, models in position_models.items()}

    joblib.dump(position_lr_models, os.path.join(models_dir, "position_lr_models.pkl"))
    joblib.dump(position_rf_models, os.path.join(models_dir, "position_rf_models.pkl"))
    joblib.dump(position_features, os.path.join(models_dir, "position_feature_map.pkl"))
    print("Tüm modeller kaydedildi.")


if __name__ == "__main__":
    main()