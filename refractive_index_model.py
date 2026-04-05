import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_pipeline():
    formation = pd.read_csv("formation_predictions.csv")
    bandgap = pd.read_csv("bandgap_predictions.csv")
    refr = pd.read_csv("refractive_index.csv")

    target = [c for c in refr.columns if c != "material_id"][0]

    data = formation.merge(bandgap, on="material_id").merge(refr, on="material_id")

    X = data.drop(columns=["material_id", target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("R2:", r2_score(y_test, pred))
    print("MAE:", mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

    pd.DataFrame({
        "actual": y_test,
        "predicted": pred
    }).to_csv("refractive_index_predictions.csv", index=False)


if __name__ == "__main__":
    run_pipeline()
