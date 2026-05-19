import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(path):
    df = pd.read_csv(path)
    print(f"Veri yüklendi: {df.shape}")
    return df


def split_features(df, target="log_market_value"):
    X = df.drop(columns=[target])
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


def plot_results(y_test, y_pred_lr, y_pred_rf, output_path="../visuals/actual_vs_predicted.png"):
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


def plot_feature_importance(rf, feature_names, output_path="../visuals/feature_importance.png"):
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


def plot_residuals(y_test, y_pred_lr, y_pred_rf, output_path="../visuals/residuals.png"):
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
    df = load_data("../data/processed/processed.csv")
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


if __name__ == "__main__":
    main()