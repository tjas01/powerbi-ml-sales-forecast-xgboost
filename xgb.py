# === XGBoost monthly forecast with exogenous lags (80/20 split) ===
# Returns:
#   output : Month, Actual, Forecast, Fitted
#   metrics: Metric, Value, Split  (MAPE, R2, MAE, RMSE on 80/20 test)
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TARGET     = "Sales"      # predict Sales
H          = 6            # forecast horizon (months)
MAX_LAG    = 12           # lag depth for all series
EXOG_COLS  = ["Quantity", "Unit Price", "Cost", "Profit"]  # extra drivers

# 1) Parse Month and keep the needed columns
df = dataset.copy()
keep = ["Month", TARGET] + EXOG_COLS
df = df[keep].copy()

# Parse Month -> datetime and regularize to month-end frequency
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
df = df.dropna(subset=["Month"]).sort_values("Month")

# Aggregate by Month just in case (sum is standard for monthly totals)
df = df.groupby("Month", as_index=False).sum(numeric_only=True)

# Use month-end frequency; fill missing months with 0 (safe for portfolio demo)
g = df.set_index("Month").asfreq("ME").fillna(0.0)

# 2) Build features: lags for Sales + exogenous, and calendar month/year
feat = pd.DataFrame(index=g.index)
feat[TARGET] = g[TARGET]

# lags for Sales
for lag in range(1, MAX_LAG+1):
    feat[f"lag_Sales_{lag}"] = g[TARGET].shift(lag)

# lags for each exogenous column (so we don't need future values)
for col in EXOG_COLS:
    for lag in range(1, MAX_LAG+1):
        feat[f"lag_{col}_{lag}"] = g[col].shift(lag)

# calendar features
feat["month"] = feat.index.month
feat["year"]  = feat.index.year

# Drop warmup rows
data = feat.dropna()
X_all = data.drop(columns=[TARGET])
y_all = data[TARGET]

# 3) Chronological 80/20 split
n = len(data)
split = max(int(n * 0.8), 1)  # at least 1 sample
X_trn, y_trn = X_all.iloc[:split], y_all.iloc[:split]
X_tst, y_tst = X_all.iloc[split:], y_all.iloc[split:]

# 4) Train & evaluate
model = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42
)
model.fit(X_trn, y_trn)
y_hat_tst = model.predict(X_tst)

# Metrics on the 20% test window (compat-safe)
mask  = np.abs(y_tst.values) > 1e-12
mape  = float(np.mean(np.abs((y_tst.values[mask] - y_hat_tst[mask]) / np.abs(y_tst.values[mask])))) if mask.sum() > 0 else np.nan
r2    = float(r2_score(y_tst, y_hat_tst))
mae   = float(mean_absolute_error(y_tst, y_hat_tst))
rmse  = float(np.sqrt(mean_squared_error(y_tst, y_hat_tst)))

# 5) Refit on all history & get fitted values
model.fit(X_all, y_all)
fitted = pd.Series(model.predict(X_all), index=X_all.index, name="Fitted")

# 6) Recursive 6-month forecast
feature_cols = list(X_all.columns)  # exact training columns/order
cur_sales = g[TARGET].copy()

# keep static histories for exogenous series (we DON'T predict them)
cur_exog = {col: g[col].copy() for col in EXOG_COLS}

preds = []
last = g.index.max()
future_idx = pd.date_range(last + pd.offsets.MonthEnd(1), periods=H, freq="ME")

for d in future_idx:
    row = {}
    # lags of Sales (update with our own predictions)
    for lag in range(1, MAX_LAG+1):
        row[f"lag_Sales_{lag}"] = cur_sales.iloc[-lag]
    # lags of exogenous (historical only; we don't generate future exogenous)
    for col in EXOG_COLS:
        for lag in range(1, MAX_LAG+1):
            row[f"lag_{col}_{lag}"] = cur_exog[col].iloc[-lag]
    # calendar
    row["month"], row["year"] = d.month, d.year

    Xnew = pd.DataFrame([row], index=[d]).reindex(columns=feature_cols, fill_value=0)
    yhat = float(model.predict(Xnew)[0])
    preds.append((d, yhat))

    # append the prediction to extend Sales for next-step lags
    cur_sales.loc[d] = yhat
    # exogenous are left unchanged (no future values assumed)

# 7) Final tables
hist = g.reset_index().rename(columns={TARGET: "Actual"})
fc   = pd.DataFrame(preds, columns=["Month", "Forecast"])
fit  = fitted.reset_index().rename(columns={"index": "Month"})

output  = hist.merge(fc, on="Month", how="outer").merge(fit, on="Month", how="left")

metrics = pd.DataFrame(
    {"Metric": ["MAPE (80/20)", "R2 (80/20)", "MAE (80/20)", "RMSE (80/20)"],
     "Value":  [mape,           r2,          mae,           rmse],
     "Split":  ["80/20"]*4}
)
