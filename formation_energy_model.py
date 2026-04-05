import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FormationEnergyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.training_time = 0

    def load_data(self, features_path, xrd_path):
        features_df = pd.read_csv(features_path)

        X = features_df.drop(
            ['material_id', 'band_gap', 'formation_energy_per_atom'],
            axis=1,
            errors='ignore'
        )

        y = features_df['formation_energy_per_atom'].values
        material_ids = features_df.get('material_id', np.arange(len(features_df)))

        xrd_data = np.load(xrd_path)

        min_len = min(len(X), len(xrd_data))
        X = X.iloc[:min_len]
        y = y[:min_len]
        xrd_data = xrd_data[:min_len]

        X_xrd = self.extract_xrd_features(xrd_data)

        X_final = pd.concat([X.reset_index(drop=True), X_xrd], axis=1)

        return X_final, y

    def extract_xrd_features(self, xrd):
        features = pd.DataFrame()
        features["xrd_mean"] = xrd.mean(axis=1)
        features["xrd_std"] = xrd.std(axis=1)
        features["xrd_max"] = xrd.max(axis=1)
        features["xrd_area"] = np.trapz(xrd, axis=1)

        return features

    def select_features(self, X, y, n=30):
        corr = []
        for col in X.columns:
            if X[col].std() > 0:
                corr.append(abs(np.corrcoef(X[col], y)[0, 1]))
            else:
                corr.append(0)

        score = pd.DataFrame({
            "feature": X.columns,
            "corr": corr,
            "var": X.var().values
        })

        score["final"] = score["corr"] * 0.7 + score["var"] * 0.3
        score = score.sort_values("final", ascending=False)

        self.selected_features = score.head(n)["feature"].tolist()
        return X[self.selected_features]

    def train(self, X_train, y_train, X_test, y_test):
        start = time.perf_counter()

        params = dict(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(X_train, y_train)

        pred = self.model.predict(X_test)

        metrics = {
            "r2": r2_score(y_test, pred),
            "mae": mean_absolute_error(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred))
        }

        self.training_time = time.perf_counter() - start

        return metrics, pred

    def save_model(self):
        joblib.dump(self.model, "formation_energy_model.pkl")
