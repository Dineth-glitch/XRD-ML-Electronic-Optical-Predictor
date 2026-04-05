import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib


class BandgapPredictor:
    def load_data(self, features_path, xrd_path):
        df = pd.read_csv(features_path)

        X = df.drop(['material_id','band_gap','formation_energy_per_atom'], axis=1, errors='ignore')
        y = df['band_gap'].values

        xrd = np.load(xrd_path)

        min_len = min(len(X), len(xrd))
        X = X.iloc[:min_len]
        y = y[:min_len]
        xrd = xrd[:min_len]

        X["xrd_mean"] = xrd.mean(axis=1)
        X["xrd_std"] = xrd.std(axis=1)
        X["xrd_max"] = xrd.max(axis=1)

        return X, y

    def train(self, X_train, y_train, X_test, y_test):
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = {
            "r2": r2_score(y_test, pred),
            "mae": mean_absolute_error(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred))
        }

        self.model = model
        return metrics, pred

    def save_model(self):
        joblib.dump(self.model, "bandgap_model.pkl")
