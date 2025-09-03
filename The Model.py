# ==============================
# MatrixDNA: Models + Graphs + Tables
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------- Config -----------------

PATH = r"C:/Users/user/Desktop/MatrixDNA Technical Assignment\raw_data\cleaned_data/matrixdna_cleaned_data.xlsx"  # Update to your path if needed
TEST_H = 6                         # last 3 months as test
WHAT_IF_ORDERS_MULT = 1.10         # +20% orders
WHAT_IF_AOV_MULT = 1.1            # +30% AOV
OUTDIR      = Path("forecast_outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------- Load & monthly aggregate -------------
def load_orders(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if np.issubdtype(pd.Series(df["order_date"]).dtype, np.number):
        df["order_date"] = pd.to_datetime(df["order_date"], unit="ms", origin="unix")
    else:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df

def make_monthly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df["order_date"].dt.to_period("M")).agg(
        total_revenue=("line_amount", "sum"),
        num_orders=("order_id", "count"),
        avg_order_value=("line_amount", "mean"),
    )
    g.index = g.index.to_timestamp()
    g["month"] = g.index.month
    g["t"] = np.arange(len(g))
    g["month_sin"] = np.sin(2*np.pi*g["month"]/12.0)
    g["month_cos"] = np.cos(2*np.pi*g["month"]/12.0)
    return g

# ----------------- Helpers -----------------
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mape

def seasonal_naive(series: pd.Series, h: int) -> np.ndarray:
    """Return next h values using same-month-last-year; fallback to last value."""
    freq = 12
    vals = series.values
    out = []
    for i in range(1, h+1):
        out.append(vals[-freq + (i-1)] if len(vals) >= freq else vals[-1])
    return np.array(out, dtype=float)

# ----------------- Models -----------------
def sarimax_revenue(monthly: pd.DataFrame, test_h=3):
    y = monthly["total_revenue"].astype(float)
    train, test = y.iloc[:-test_h], y.iloc[-test_h:]
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.forecast(steps=test_h)
    mae, mape = metrics(test.values, fc.values)
    table = pd.DataFrame({
        "Month": test.index.strftime("%Y-%m"),
        "Actual Revenue": test.values,
        "Predicted Revenue": fc.values
    })
    return table, (mae, mape), res

REG_FEATURES = ["num_orders","avg_order_value","month_sin","month_cos","t"]

def regression_revenue(monthly: pd.DataFrame, test_h=3, use_future_features=False):
    df = monthly.copy()
    train, test = df.iloc[:-test_h], df.iloc[-test_h:]

    X_train = train[REG_FEATURES]
    y_train = train["total_revenue"].values
    model = LinearRegression().fit(X_train, y_train)

    if use_future_features:
        X_test = test[REG_FEATURES]
    else:
        # forecast Orders & AOV with seasonal naive
        X_test = test[REG_FEATURES].copy()
        X_test.loc[:, "num_orders"] = seasonal_naive(train["num_orders"], test_h)
        X_test.loc[:, "avg_order_value"] = seasonal_naive(train["avg_order_value"], test_h)

    pred = model.predict(X_test)
    mae, mape = metrics(test["total_revenue"].values, pred)

    table = pd.DataFrame({
        "Month": test.index.strftime("%Y-%m"),
        "Actual Revenue": test["total_revenue"].values,
        "Predicted Revenue": pred
    })
    return table, (mae, mape), model

def simulate_what_if(monthly: pd.DataFrame, model: LinearRegression, test_h=3,
                     orders_mult=1.20, aov_mult=1.30):
    test = monthly.iloc[-test_h:].copy()
    X_base = test[REG_FEATURES].copy()
    X_imp = X_base.copy()
    X_imp["num_orders"] = X_imp["num_orders"] * orders_mult
    X_imp["avg_order_value"] = X_imp["avg_order_value"] * aov_mult

    base_pred = model.predict(X_base)
    imp_pred  = model.predict(X_imp)
    out = pd.DataFrame({
        "Month": test.index.strftime("%Y-%m"),
        "Predicted Revenue (baseline)": base_pred,
        "Predicted Revenue (improved)": imp_pred,
        "Improvement": imp_pred - base_pred
    })
    return out

# ----------------- Plotting -----------------
def plot_actual_vs_pred(full_series: pd.Series,
                        test_h: int,
                        pred: pd.Series,
                        title: str,
                        fname: Path):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(full_series.index, full_series.values, label="Actual")
    test_idx = full_series.index[-test_h:]
    ax.plot(test_idx, pred, label="Forecast", linestyle="--", marker="o")
    # Shade test window
    ax.axvspan(test_idx[0], test_idx[-1], alpha=0.1, label="Test window")
    ax.set_title(title)
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue (₪)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.show()
    plt.close(fig)

def plot_what_if(what_if_df: pd.DataFrame, fname: Path):
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(len(what_if_df))
    w = 0.35
    ax.bar(x - w/2, what_if_df["Predicted Revenue (baseline)"], width=w, label="Baseline")
    ax.bar(x + w/2, what_if_df["Predicted Revenue (improved)"], width=w, label="Improved")
    ax.set_xticks(x); ax.set_xticklabels(what_if_df["Month"])
    ax.set_ylabel("Revenue (₪)"); ax.set_title("What-If: Baseline vs Improved")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)

# ----------------- Run -----------------
orders = load_orders(PATH)
monthly = make_monthly(orders)

# 1) SARIMAX backtest
sarimax_tbl, sarimax_metrics, sarimax_model = sarimax_revenue(monthly, TEST_H)
sarimax_tbl.to_csv(OUTDIR/"sarimax_backtest.csv", index=False)

# 2) Regression backtest (proper forecast: uses forecasted Orders&AOV)
reg_fore_tbl, reg_fore_metrics, reg_fore_model = regression_revenue(
    monthly, TEST_H, use_future_features=False
)
reg_fore_tbl.to_csv(OUTDIR/"reg_backtest_forecasted_features.csv", index=False)

# 3) Regression backtest (upper bound: uses true Orders&AOV of test)
reg_true_tbl, reg_true_metrics, reg_true_model = regression_revenue(
    monthly, TEST_H, use_future_features=True
)
reg_true_tbl.to_csv(OUTDIR/"reg_backtest_true_features.csv", index=False)

# 4) What-If simulation on last TEST_H months (use the upper-bound model for response curve)
what_if_tbl = simulate_what_if(
    monthly, reg_true_model, TEST_H,
    orders_mult=WHAT_IF_ORDERS_MULT,
    aov_mult=WHAT_IF_AOV_MULT
)
what_if_tbl.to_csv(OUTDIR/"what_if_simulation.csv", index=False)

# --------- Accuracy summary table ----------
summary = pd.DataFrame({
    "Model": [
        "SARIMAX (revenue)",
        "Regression (forecasted Orders & AOV)",
        "Regression (true Orders & AOV) — upper bound"
    ],
    "MAE": [sarimax_metrics[0], reg_fore_metrics[0], reg_true_metrics[0]],
    "MAPE": [sarimax_metrics[1], reg_fore_metrics[1], reg_true_metrics[1]],
})
summary.to_csv(OUTDIR/"accuracy_summary.csv", index=False)
print("\nAccuracy summary:\n", summary.round(4))

# --------- Plots ----------
# (A) Actual vs SARIMAX forecast (test window)
plot_actual_vs_pred(
    monthly["total_revenue"], TEST_H,
    sarimax_tbl["Predicted Revenue"].values,
    "Monthly Revenue — Actual vs SARIMAX (Test Window)",
    OUTDIR/"plot_actual_vs_sarimax.png"
)

# (B) Actual vs Regression (upper-bound) forecast (test window)
plot_actual_vs_pred(
    monthly["total_revenue"], TEST_H,
    reg_true_tbl["Predicted Revenue"].values,
    "Monthly Revenue — Actual vs Regression (Upper-Bound) (Test Window)",
    OUTDIR/"plot_actual_vs_reg_upper.png"
)

# (C) What-If bars
plot_what_if(what_if_tbl, OUTDIR/"plot_what_if_baseline_vs_improved.png")

print(f"\nSaved tables & figures to: {OUTDIR.resolve()}")
print("- sarimax_backtest.csv")
print("- reg_backtest_forecasted_features.csv")
print("- reg_backtest_true_features.csv")
print("- what_if_simulation.csv")
print("- accuracy_summary.csv")
print("- plot_actual_vs_sarimax.png")
print("- plot_actual_vs_reg_upper.png")
print("- plot_what_if_baseline_vs_improved.png")










# ----------------- Config -----------------
FUTURE_H = 6  # <--- NEW: how many months to forecast beyond the last actual

# ----------------- Helpers (NEW) -----------------
import matplotlib.dates as mdates

def make_future_exog(monthly: pd.DataFrame, h: int) -> pd.DataFrame:
    """Build exogenous features for the next h months for the regression model."""
    last_idx = monthly.index[-1]
    future_idx = pd.date_range(last_idx + pd.offsets.MonthBegin(1), periods=h, freq="MS")

    df = pd.DataFrame(index=future_idx)
    df["month"] = df.index.month
    # continue time index t
    df["t"] = np.arange(monthly["t"].iloc[-1] + 1, monthly["t"].iloc[-1] + 1 + h)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # naive seasonal forecasts for Orders & AOV
    df["num_orders"] = seasonal_naive(monthly["num_orders"], h)
    df["avg_order_value"] = seasonal_naive(monthly["avg_order_value"], h)
    return df




def plot_all_models_one_figure(
    monthly: pd.DataFrame,
    test_h: int,
    fut_h: int,
    sarimax_tbl: pd.DataFrame,
    reg_fore_tbl: pd.DataFrame,
    reg_true_tbl: pd.DataFrame,
    reg_fore_model: LinearRegression,
    outpath: Path,
    reg_true_model: LinearRegression | None = None,                  # NEW
    upper_future_features: pd.DataFrame | None = None                # NEW (index=future months; cols at least: num_orders, avg_order_value)
):
    import matplotlib.dates as mdates

    def _ensure_reg_features_from(df_future: pd.DataFrame) -> pd.DataFrame:
        out = df_future.copy()
        if "month" not in out: out["month"] = out.index.month
        start_t = int(monthly["t"].iloc[-1]) + 1
        out["t"] = np.arange(start_t, start_t + len(out))
        out["month_sin"] = np.sin(2*np.pi*out["month"]/12.0)
        out["month_cos"] = np.cos(2*np.pi*out["month"]/12.0)
        return out[REG_FEATURES]

    y = monthly["total_revenue"].astype(float)
    idx_all = y.index
    test_idx = idx_all[-test_h:]
    forecast_start = idx_all[-1] + pd.offsets.MonthBegin(1)

    # future SARIMAX
    sarimax_full = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarimax_future = sarimax_full.forecast(steps=fut_h)
    future_idx = sarimax_future.index

    # future Regression (forecasted features)
    X_fut = make_future_exog(monthly, fut_h)  # uses seasonal_naive
    reg_future = reg_fore_model.predict(X_fut[REG_FEATURES])

    fig, ax = plt.subplots(figsize=(14, 6))

    # Actual (black)
    l_actual, = ax.plot(idx_all, y.values, marker="o", linewidth=2, color="black", label="Actual")

    # Validation lines
    l_sarimax_val, = ax.plot(test_idx, sarimax_tbl["Predicted Revenue"].values, linestyle="--", marker="x", label="SARIMAX — validation")
    l_reg_fore_val, = ax.plot(test_idx, reg_fore_tbl["Predicted Revenue"].values, linestyle="--", marker="^", label="Regression (forecasted feats) — validation")
    l_reg_true_val, = ax.plot(test_idx, reg_true_tbl["Predicted Revenue"].values, linestyle="--", marker="s", label="Regression (upper bound) — validation")

    # Future lines
    l_sarimax_fut, = ax.plot(future_idx, sarimax_future.values, marker="d", linewidth=2, label=f"SARIMAX — forecast (+{fut_h}m)")
    l_reg_fut,    = ax.plot(future_idx, reg_future,            marker="P", linewidth=2, label=f"Regression (forecasted feats) — forecast (+{fut_h}m)")

    # Optional: Upper-bound future (needs true/planned future features)
    handles2 = [l_sarimax_val, l_reg_fore_val, l_reg_true_val, l_sarimax_fut, l_reg_fut]
    labels2  = ["Validation — SARIMAX", "Validation — Regression (forecasted feats)", "Validation — Regression (upper bound)",
                "Forecast — SARIMAX", "Forecast — Regression (forecasted feats)"]
    if (reg_true_model is not None) and (upper_future_features is not None):
        X_upper = _ensure_reg_features_from(upper_future_features)
        reg_upper_future = reg_true_model.predict(X_upper)
        l_reg_true_fut, = ax.plot(upper_future_features.index, reg_upper_future, marker="s", linewidth=2,
                                  label="Regression (upper bound) — forecast")
        handles2.append(l_reg_true_fut)
        labels2.append("Forecast — Regression (upper bound)")

    # Markers & axes
    ax.axvline(test_idx[0], color="grey", linestyle="--", alpha=0.7, label="Validation Start")
    ax.axvline(forecast_start, color="grey", linestyle=":",  alpha=0.9, label="Forecast Start")
    ax.set_title("Actual vs Models — Validation & Future Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Revenue (₪)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); ax.grid(alpha=0.2)

    # Split legends
    leg1 = ax.legend(handles=[l_actual], title="Actual", loc="upper left", frameon=True, fancybox=True, borderpad=0.4)
    leg2 = ax.legend(handles=handles2, labels=labels2, title="Validation vs Forecast",
                     loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=True, fancybox=True, borderpad=0.4)
    ax.add_artist(leg1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=160)
    plt.show()

    plt.close(fig)



# Example: use naive forecast then apply your what-if multipliers (acts as “planned” values)
FUTURE_H = 6
X_naive  = make_future_exog(monthly, FUTURE_H)
UPPER_FUT = pd.DataFrame(index=X_naive.index)
UPPER_FUT["num_orders"] = X_naive["num_orders"] * WHAT_IF_ORDERS_MULT   # e.g., +10%
UPPER_FUT["avg_order_value"] = X_naive["avg_order_value"] * WHAT_IF_AOV_MULT  # e.g., +10%

plot_all_models_one_figure(
    monthly, TEST_H, FUTURE_H,
    sarimax_tbl, reg_fore_tbl, reg_true_tbl,
    reg_fore_model,
    OUTDIR / "plot_all_models_backtest_and_forecast.png",
    reg_true_model=reg_true_model,
    upper_future_features=UPPER_FUT
)















############

from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates

def fs_backtest_and_forecast(monthly_top10: pd.DataFrame,
                             FS_FEATURES: list[str],
                             test_h: int,
                             fut_h: int,
                             upper_future_features: pd.DataFrame | None = None):
    """Train FS regression, produce validation & future forecasts."""
    y_all = monthly_top10["total_revenue"].astype(float)
    X_all = monthly_top10[FS_FEATURES].astype(float)

    # Train / validation split
    X_train = X_all.iloc[:-test_h]
    y_train = y_all.iloc[:-test_h]
    X_val_true = X_all.iloc[-test_h:]          # true features for validation (upper bound)
    y_val      = y_all.iloc[-test_h:]
    val_idx    = y_val.index

    # Train one linear model (used for both modes)
    lin_fs = LinearRegression().fit(X_train, y_train)

    # --- Validation predictions ---
    # Upper bound (true features)
    pred_val_upper = lin_fs.predict(X_val_true)

    # Proper forecast on the validation window: replace features with seasonal-naive
    X_val_fore = X_val_true.copy()
    for f in FS_FEATURES:
        X_val_fore[f] = seasonal_naive(X_train[f], test_h)
    pred_val_fore = lin_fs.predict(X_val_fore)

    # --- Future forecast (+fut_h months) ---
    future_idx = pd.date_range(y_all.index[-1] + pd.offsets.MonthBegin(1),
                               periods=fut_h, freq="MS")
    X_future_fore = pd.DataFrame(index=future_idx)
    for f in FS_FEATURES:
        X_future_fore[f] = seasonal_naive(X_all[f], fut_h)
    pred_future_fore = lin_fs.predict(X_future_fore)

    # Optional: future upper bound if user supplies planned/true future features
    pred_future_upper = None
    if upper_future_features is not None:
        X_future_upper = upper_future_features.reindex(future_idx)[FS_FEATURES].astype(float)
        pred_future_upper = lin_fs.predict(X_future_upper)

    return {
        "lin": lin_fs,
        "idx_all": y_all.index,
        "y_all": y_all.values,
        "val_idx": val_idx,
        "future_idx": future_idx,
        "pred_val_upper": pred_val_upper,
        "pred_val_fore": pred_val_fore,
        "pred_future_fore": pred_future_fore,
        "pred_future_upper": pred_future_upper,
        "y_val": y_val.values,
    }





def plot_fs_model_all(res: dict, test_h: int, fut_h: int, outpath: Path):
    fig, ax = plt.subplots(figsize=(14, 6))

    # Actual (black)
    l_actual, = ax.plot(res["idx_all"], res["y_all"], marker="o", linewidth=2,
                        color="black", label="Actual")

    # Validation lines
    l_fs_val_fore,  = ax.plot(res["val_idx"], res["pred_val_fore"],
                              linestyle="--", marker="^",
                              label="FS (forecasted feats) — validation")
    l_fs_val_upper, = ax.plot(res["val_idx"], res["pred_val_upper"],
                              linestyle="--", marker="s",
                              label="FS (upper bound) — validation")

    # Future lines
    l_fs_fut_fore, = ax.plot(res["future_idx"], res["pred_future_fore"],
                             marker="P", linewidth=2,
                             label=f"FS (forecasted feats) — forecast (+{fut_h}m)")

    handles2 = [l_fs_val_fore, l_fs_val_upper, l_fs_fut_fore]
    labels2  = ["Validation — FS (forecasted feats)",
                "Validation — FS (upper bound)",
                "Forecast — FS (forecasted feats)"]

    # Optional future upper bound (only if future features were provided)
    if res["pred_future_upper"] is not None:
        l_fs_fut_upper, = ax.plot(res["future_idx"], res["pred_future_upper"],
                                  marker="s", linewidth=2,
                                  label="FS (upper bound) — forecast")
        handles2.append(l_fs_fut_upper)
        labels2.append("Forecast — FS (upper bound)")

    # Vertical markers
    ax.axvline(res["val_idx"][0], color="grey", linestyle="--", alpha=0.7, label="Validation Start")
    ax.axvline(res["future_idx"][0],  color="grey", linestyle=":",  alpha=0.9, label="Forecast Start")

    # Axes/format
    ax.set_title("FS Model — Validation & Future Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Revenue (₪)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); ax.grid(alpha=0.2)

    # Split legends into two buckets
    leg1 = ax.legend(handles=[l_actual], title="Actual",
                     loc="upper left", frameon=True, fancybox=True, borderpad=0.4)
    leg2 = ax.legend(handles=handles2, labels=labels2, title="Validation vs Forecast",
                     loc="upper center", bbox_to_anchor=(0.5, 1.22),
                     ncol=2, frameon=True, fancybox=True, borderpad=0.4)
    ax.add_artist(leg1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=160)
    plt.show()
    plt.close(fig)



FUTURE_H = 6  # or any horizon

# Optional: planned/what-if future features to show an "upper-bound forecast" line
# (these must be monthly means for the same FS_FEATURES)
X_naive = pd.DataFrame(index=pd.date_range(monthly_top10.index[-1] + pd.offsets.MonthBegin(1),
                                           periods=FUTURE_H, freq="MS"))
for f in top10_features:
    X_naive[f] = seasonal_naive(monthly_top10[f], FUTURE_H)

UPPER_FUT = X_naive.copy()
UPPER_FUT[top10_features] = UPPER_FUT[top10_features] * 1.10  # example: +10% plan (adjust as needed)

# Run FS backtest + forecast (pass UPPER_FUT to draw the upper-bound forecast line)
fs_res = fs_backtest_and_forecast(
    monthly_top10=monthly_top10,
    FS_FEATURES=top10_features,
    test_h=TEST_H,
    fut_h=FUTURE_H,
    upper_future_features=UPPER_FUT  # or None to omit the future upper-bound line
)

# Plot
plot_fs_model_all(fs_res, test_h=TEST_H, fut_h=FUTURE_H,
                  outpath=OUTDIR / "plot_fs_validation_and_forecast.png")

print("- plot_fs_validation_and_forecast.png")





















































from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor


def build_fs_frame_from_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, list, pd.Series]:
    """Create a numeric, order-level feature frame for FS, starting from the raw orders table."""
    df = orders.copy()

    # Ensure datetime
    if np.issubdtype(pd.Series(df["order_date"]).dtype, np.number):
        df["order_date"] = pd.to_datetime(df["order_date"], unit="ms", origin="unix")
    else:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Basic temporal features (order level)
    df["dow"]    = df["order_date"].dt.weekday
    df["month"]  = df["order_date"].dt.month
    df["is_wend"] = df["dow"].isin([5,6]).astype(int)

    # Booleans → int
    for b in ["is_online","coupon_used","is_return_order"]:
        if b in df.columns:
            df[b] = df[b].astype(int)

    # One-hot for low-cardinality categoricals
    low_card_cols = [c for c in ["channel","region","top_department","top_size_group"] if c in df.columns]
    if low_card_cols:
        df = pd.get_dummies(df, columns=low_card_cols, prefix=low_card_cols, drop_first=False)

    # Frequency-encoding for high-cardinality categoricals (no column explosion)
    high_card_cols = [c for c in ["store_id","top_category","top_location","top_region"] if c in df.columns]
    for col in high_card_cols:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq).fillna(0)

    # Columns to drop (ids/text)
    drop_cols = [c for c in ["order_id","customer_phone","items_id_list","year_month",
                             "store_id","top_category","top_location","top_region"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Select numeric features and target
    feature_cols = [c for c in df.columns if c not in ["order_date","line_amount"]]
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df["line_amount"].astype(float)

    # Return a frame that still contains order_date + line_amount + numeric features
    df_fs = pd.concat([df[["order_date","line_amount"]], X], axis=1)
    return df_fs, list(X.columns), y



# === Build FS frame from original orders ===
df_fs, feature_cols_fs, y_fs = build_fs_frame_from_orders(orders)

# === Train FS models on order-level to get feature importances ===
ada  = AdaBoostRegressor(n_estimators=400, random_state=42)
et   = ExtraTreesRegressor(n_estimators=400, random_state=42)

ada.fit(df_fs[feature_cols_fs], y_fs)
et.fit (df_fs[feature_cols_fs], y_fs)

imp = pd.DataFrame({
    "feature": feature_cols_fs,
    "imp_ada": ada.feature_importances_,
    "imp_et":  et.feature_importances_
})

# --- Top-10 per model ---
top10_ada = imp.sort_values("imp_ada", ascending=False).head(10)["feature"].tolist()
top10_et  = imp.sort_values("imp_et",  ascending=False).head(10)["feature"].tolist()
top10_avg = imp.sort_values("imp_avg", ascending=False).head(10)["feature"].tolist()  # for optional upper bound line

print("Top-10 (Ada):", top10_ada)
print("Top-10 (ET): ", top10_et)

# --- Monthly frames for each feature set ---
monthly_ada = aggregate_monthly_selected(df_fs, top10_ada)
monthly_et  = aggregate_monthly_selected(df_fs, top10_et)
monthly_avg = aggregate_monthly_selected(df_fs, top10_avg)  # for upper bound


imp["imp_avg"] = (imp["imp_ada"] + imp["imp_et"]) / 2.0
top10_features = imp.sort_values("imp_avg", ascending=False).head(10)["feature"].tolist()
print("Top-10 features (AdaBoost + ExtraTrees):", top10_features)



from sklearn.linear_model import LinearRegression

def fs_make_preds(monthly_sel: pd.DataFrame,
                  features: list[str],
                  test_h: int,
                  fut_h: int,
                  future_true_features: pd.DataFrame | None = None):
    """
    Train a linear regression on monthly 'features' -> total_revenue.
    Returns validation predictions (proper-forecast & upper-bound) and future forecasts.
    future_true_features: optional DataFrame (index=future months) with the *true/planned* values for 'features'
                          to draw an upper-bound future line.
    """
    y = monthly_sel["total_revenue"].astype(float)
    X = monthly_sel[features].astype(float)

    # indices
    idx_all  = y.index
    val_idx  = idx_all[-test_h:]
    fut_idx  = pd.date_range(idx_all[-1] + pd.offsets.MonthBegin(1), periods=fut_h, freq="MS")

    # fit
    lin = LinearRegression().fit(X.iloc[:-test_h], y.iloc[:-test_h])

    # validation (upper bound uses true features)
    X_val_true = X.iloc[-test_h:]
    pred_val_upper = lin.predict(X_val_true)

    # validation proper-forecast: replace with seasonal-naive per feature
    X_val_fore = pd.DataFrame(index=val_idx)
    for f in features:
        X_val_fore[f] = seasonal_naive(X.iloc[:-test_h][f], test_h)
    pred_val_fore = lin.predict(X_val_fore)

    # future proper-forecast: seasonal-naive per feature from all history
    X_fut_fore = pd.DataFrame(index=fut_idx)
    for f in features:
        X_fut_fore[f] = seasonal_naive(X[f], fut_h)
    pred_fut_fore = lin.predict(X_fut_fore)

    # future upper-bound: only if provided
    pred_fut_upper = None
    if future_true_features is not None:
        X_fut_upper = future_true_features.reindex(fut_idx)[features].astype(float)
        pred_fut_upper = lin.predict(X_fut_upper)

    return {
        "lin": lin,
        "idx_all": idx_all,
        "y_all": y.values,
        "val_idx": val_idx,
        "fut_idx": fut_idx,
        "pred_val_fore": pred_val_fore,
        "pred_val_upper": pred_val_upper,
        "pred_fut_fore": pred_fut_fore,
        "pred_fut_upper": pred_fut_upper,
    }





import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def plot_fs_ada_vs_et(res_ada: dict,
                      res_et: dict,
                      res_upper: dict | None,
                      test_h: int,
                      fut_h: int,
                      outpath: Path):
    """
    One line per algorithm (Ada vs ET), each spanning validation + forecast (proper-forecast).
    Optionally add a third 'Upper Bound' line (validation + forecast if available).
    """
    # We'll take the "actual" from either result (same index)
    idx_all = res_ada["idx_all"]
    y_all   = res_ada["y_all"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Actual in black
    l_actual, = ax.plot(idx_all, y_all, marker="o", linewidth=2,
                        color="black", label="Actual")

    # --- ADA line (one continuous style: validation+forecast proper-forecast) ---
    x_ada = list(res_ada["val_idx"]) + list(res_ada["fut_idx"])
    y_ada = list(res_ada["pred_val_fore"]) + list(res_ada["pred_fut_fore"])
    l_ada, = ax.plot(x_ada, y_ada, marker="P", linewidth=2, label=f"AdaBoost — validation+forecast")

    # --- ET line (one continuous style: validation+forecast proper-forecast) ---
    x_et = list(res_et["val_idx"]) + list(res_et["fut_idx"])
    y_et = list(res_et["pred_val_fore"]) + list(res_et["pred_fut_fore"])
    l_et, = ax.plot(x_et, y_et, marker="D", linewidth=2, label=f"ExtraTrees — validation+forecast")

    # --- Optional: Upper-bound line (validation + future if available) ---
    handles_models = [l_ada, l_et]
    labels_models  = ["AdaBoost — validation+forecast", "ExtraTrees — validation+forecast"]

    if res_upper is not None:
        x_up = list(res_upper["val_idx"]) + list(res_upper["fut_idx"])
        y_up_val = list(res_upper["pred_val_upper"])  # validation upper bound
        # future upper bound might be None if you didn't pass future_true_features
        y_up_fut = list(res_upper["pred_fut_upper"]) if res_upper["pred_fut_upper"] is not None else []
        y_up = y_up_val + y_up_fut
        l_up, = ax.plot(x_up, y_up, linestyle="--", marker="s", linewidth=2,
                        label="Upper Bound — validation+forecast")
        handles_models.append(l_up)
        labels_models.append("Upper Bound — validation+forecast")

    # Vertical markers
    ax.axvline(res_ada["val_idx"][0], color="grey", linestyle="--", alpha=0.7, label="Validation Start")
    ax.axvline(res_ada["fut_idx"][0], color="grey", linestyle=":",  alpha=0.9, label="Forecast Start")

    # Axes
    ax.set_title("FS (Top-10) — AdaBoost vs ExtraTrees: Validation & Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Revenue (₪)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); ax.grid(alpha=0.2)

    # Split legends: Actual vs Models
    leg1 = ax.legend(handles=[l_actual], title="Actual",
                     loc="upper left", frameon=True, fancybox=True, borderpad=0.4)
    leg2 = ax.legend(handles=handles_models, labels=labels_models,
                     title="FS Models (validation+forecast)",
                     loc="upper center", bbox_to_anchor=(0.5, 1.22),
                     ncol=2, frameon=True, fancybox=True, borderpad=0.4)
    ax.add_artist(leg1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=160)
    plt.show()
    plt.close(fig)


FUTURE_H = 6  # or as you wish

# OPTIONAL: build future "upper-bound" features for the top10_avg set
# (If you want the dashed upper-bound line on future months)
future_idx = pd.date_range(monthly_avg.index[-1] + pd.offsets.MonthBegin(1), periods=FUTURE_H, freq="MS")
UPPER_FUT = pd.DataFrame(index=future_idx)
for f in top10_avg:
    # start from seasonal-naive guess and add a small uplift (adjust to your scenario)
    UPPER_FUT[f] = seasonal_naive(monthly_avg[f], FUTURE_H) * 1.10

# Compute predictions:
res_ada = fs_make_preds(monthly_ada, top10_ada, TEST_H, FUTURE_H)                 # proper-forecast only
res_et  = fs_make_preds(monthly_et,  top10_et,  TEST_H, FUTURE_H)                 # proper-forecast only
res_up  = fs_make_preds(monthly_avg, top10_avg, TEST_H, FUTURE_H, UPPER_FUT)      # upper-bound (val+future)

# Plot: 1 line per algorithm (+ optional upper bound dashed)
plot_fs_ada_vs_et(res_ada, res_et, res_up,
                  test_h=TEST_H, fut_h=FUTURE_H,
                  outpath=OUTDIR / "plot_fs_ada_vs_et_validation_and_forecast.png")

print("- plot_fs_ada_vs_et_validation_and_forecast.png")







def aggregate_monthly_selected(df_fs: pd.DataFrame, selected: list) -> pd.DataFrame:
    m = df_fs.copy()
    m["month_ts"] = m["order_date"].dt.to_period("M").dt.to_timestamp()

    # Target: monthly total revenue
    y_month = m.groupby("month_ts")["line_amount"].sum().rename("total_revenue")

    # Features: monthly MEANS of the Top-10 order-level features
    X_month = m.groupby("month_ts")[selected].mean()

    out = X_month.join(y_month)

    # Optional: add simple time features if תרצה (אפשר להשאיר בחוץ אם ביקשת רק Top-10)
    # out["month_num"] = out.index.month
    # out["t"] = np.arange(len(out))
    # out["month_sin"] = np.sin(2*np.pi*out["month_num"]/12.0)
    # out["month_cos"] = np.cos(2*np.pi*out["month_num"]/12.0)

    return out

monthly_top10 = aggregate_monthly_selected(df_fs, top10_features)






# === Upper-Bound Regression USING Top-10 features selected from orders-level ===
TEST_H = 6  # keep your current horizon

train_u = monthly_top10.iloc[:-TEST_H].copy()
test_u  = monthly_top10.iloc[-TEST_H:].copy()

FS_FEATURES = top10_features  # <- use ONLY the selected Top-10 (monthly means)
lin = LinearRegression().fit(train_u[FS_FEATURES], train_u["total_revenue"].values)
reg_upper_pred = lin.predict(test_u[FS_FEATURES])

mae_u = mean_absolute_error(test_u["total_revenue"].values, reg_upper_pred)
mape_u = mean_absolute_percentage_error(test_u["total_revenue"].values, reg_upper_pred)
print(f"Upper Bound (Top-10 from orders)  MAE={mae_u:,.0f}  MAPE={mape_u:.2%}")

reg_upper_tbl = pd.DataFrame({
    "Month": test_u.index.strftime("%Y-%m"),
    "Actual Revenue": test_u["total_revenue"].values,
    "Predicted Revenue": reg_upper_pred
})

