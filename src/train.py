import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def plot_results(y_test, y_pred_lr, y_pred_rf, output_path=None):
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "visuals", "actual_vs_predicted.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, y_pred, title in zip(
        axes,
        [y_pred_lr, y_pred_rf],
        ["Linear Regression", "Random Forest"]
    ):
        ax.scatter(y_test, y_pred, alpha=0.2, color="steelblue")
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], "r--", linewidth=1)
        ax.set_xlabel("Gerçek (log)")
        ax.set_ylabel("Tahmin (log)")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Görsel kaydedildi: {output_path}")
    plt.show()


def plot_feature_importance(rf, feature_names, output_path=None):
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "visuals", "feature_importance.png")

    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="steelblue")
    plt.xlabel("Önem Skoru")
    plt.title("Random Forest — Özellik Önemi")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Görsel kaydedildi: {output_path}")
    plt.show()


def plot_residuals(y_test, y_pred_lr, y_pred_rf, output_path=None):
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "visuals", "residuals.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, y_pred, title in zip(
        axes,
        [y_pred_lr, y_pred_rf],
        ["Linear Regression", "Random Forest"]
    ):
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.2, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Tahmin (log)")
        ax.set_ylabel("Artık (Gerçek - Tahmin)")
        ax.set_title(f"{title} — Artık Dağılımı")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Görsel kaydedildi: {output_path}")
    plt.show()


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

    plot_results(y_test, y_pred_lr, y_pred_rf)
    plot_feature_importance(rf, X.columns)
    plot_residuals(y_test, y_pred_lr, y_pred_rf)
    print_predictions(y_test, y_pred_lr, y_pred_rf)

    position_models, _ = train_position_models(df)
    print_position_predictions(df, position_models)

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(rf, os.path.join(models_dir, "random_forest.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, "feature_columns.pkl"))

    print("Model kaydedildi.")


if __name__ == "__main__":
    main()