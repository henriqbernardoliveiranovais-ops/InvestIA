import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_ml_model(X, y, model_type="random_forest"):
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "lasso":
        model = Lasso(alpha=0.1)
    else:
        model = RandomForestRegressor(random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    scores, predictions = [], []

    for tr, te in tscv.split(X):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        scores.append({
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        })
        predictions.extend(y_pred)

    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    model.fit(X_scaled, y)

    return model, scaler_final, scores, np.asarray(predictions)

def make_predictions(model, scaler, last_data, features, days_ahead=5):
    last_features = last_data[features].values.reshape(1, -1)
    last_features_scaled = scaler.transform(last_features)
    return float(model.predict(last_features_scaled)[0])
