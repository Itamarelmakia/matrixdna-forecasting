# ==============================
# MatrixDNA — Ordered Pipeline: Baselines + FS (Ada/ET) + Plots/Tables
# ==============================
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------- Config -----------------
PATH = r"C:/Users/user/Desktop/MatrixDNA Technical Assignment/raw_data/cleaned_data/matrixdna_cleaned_data.xlsx"
OUTDIR = Path("forecast_outputs"); OUTDIR.mkdir(exist_ok=True)

TEST_H = 6            # Last N months for validation (backtest)
FUTURE_H = 6          # Months to forecast beyond the last actual

# What-if multipliers (used for scenario curves / optional upper-bound future)
WHAT_IF_ORDERS_MULT = 1.10   # +10% Orders
WHAT_IF_AOV_MULT    = 1.10   # +10% AOV

# Baseline regression feature set (non-FS models)
REG_FEATURES = ["num_orders", "avg_order_value", "month_sin", "month_cos", "t"]


# ----------------- IO & Basic Transforms -----------------
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
    """h-step seasonal naive (freq=12). Fallback: last value if <12 history."""
    freq = 12
    vals = series.values
    out = []
    for i in range(1, h+1):
        out.append(vals[-freq + (i-1)] if len(vals) >= freq else vals[-1])
    return np.array(out, dtype=float)


def make_future_exog(monthly: pd.DataFrame, h: int) -> pd.DataFrame:
    """Exog for next h months for baseline regression (seasonal-naive Orders/AOV)."""
    last_idx = monthly.index[-1]
    future_idx = pd.date_range(last_idx + pd.offsets.MonthBegin(1), periods=h, freq="MS")

    df = pd.DataFrame(index=future_idx)
    df["month"] = df.index.month
    df["t"] = np.arange(monthly["t"].iloc[-1] + 1, monthly["t"].iloc[-1] + 1 + h)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["num_orders"] = seasonal_naive(monthly["num_orders"], h)
    df["avg_order_value"] = seasonal_naive(monthly["avg_order_value"], h)
    return df


# ----------------- Baseline Models (No FS) -----------------
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


def regression_revenue(monthly: pd.DataFrame, test_h=3, use_future_features=False):
    df = monthly.copy()
    train, test = df.iloc[:-test_h], df.iloc[-test_h:]

    X_train = train[REG_FEATURES]
    y_train = train["total_revenue"].values
    model = LinearRegression().fit(X_train, y_train)

    if use_future_features:
        X_test = test[REG_FEATURES]
    else:
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
                     orders_mult=1.10, aov_mult=1.10):
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


# ----------------- Baseline Plots -----------------
def plot_actual_vs_pred(full_series: pd.Series, test_h: int, pred: np.ndarray,
                        title: str, fname: Path):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(full_series.index, full_series.values, label="Actual")
    test_idx = full_series.index[-test_h:]
    ax.plot(test_idx, pred, label="Forecast", linestyle="--", marker="o")
    ax.axvspan(test_idx[0], test_idx[-1], alpha=0.1, label="Validation Window")
    ax.set_title(title)
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue (₪)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.12), frameon=True, fancybox=True, borderpad=0.4)
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout(); fig.savefig(fname, dpi=160)
    plt.show(); plt.close(fig)


def plot_what_if(what_if_df: pd.DataFrame, fname: Path):
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(len(what_if_df)); w = 0.35
    ax.bar(x - w/2, what_if_df["Predicted Revenue (baseline)"], width=w, label="Baseline")
    ax.bar(x + w/2, what_if_df["Predicted Revenue (improved)"], width=w, label="Improved")
    ax.set_xticks(x); ax.set_xticklabels(what_if_df["Month"])
    ax.set_ylabel("Revenue (₪)"); ax.set_title("What-If: Baseline vs Improved")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.12), frameon=True, fancybox=True, borderpad=0.4)
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout(); fig.savefig(fname, dpi=160); plt.show(); plt.close(fig)


def plot_all_models_one_figure(
    monthly: pd.DataFrame,
    test_h: int,
    fut_h: int,
    sarimax_tbl: pd.DataFrame,
    reg_fore_tbl: pd.DataFrame,
    reg_true_tbl: pd.DataFrame,
    reg_fore_model: LinearRegression,
    outpath: Path,
    reg_true_model: LinearRegression | None = None,
    upper_future_features: pd.DataFrame | None = None,
):
    """One chart: Actual + validation lines (all models) + future forecasts.
       Optional: dashed 'upper-bound' future for regression with true/planned features.
    """
    y = monthly["total_revenue"].astype(float)
    idx_all = y.index
    test_idx = idx_all[-test_h:]
    forecast_start = idx_all[-1] + pd.offsets.MonthBegin(1)

    # Future SARIMAX (fit on full history)
    sarimax_full = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarimax_future = sarimax_full.forecast(steps=fut_h)
    future_idx = sarimax_future.index

    # Future Regression (forecasted features)
    X_fut = make_future_exog(monthly, fut_h)
    reg_future = reg_fore_model.predict(X_fut[REG_FEATURES])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Actual
    l_actual, = ax.plot(idx_all, y.values, marker="o", linewidth=2, color="black", label="Actual")

    # Validation lines
    l_sarimax_val, = ax.plot(test_idx, sarimax_tbl["Predicted Revenue"].values,
                             linestyle="--", marker="x", label="SARIMAX — validation")
    l_reg_fore_val, = ax.plot(test_idx, reg_fore_tbl["Predicted Revenue"].values,
                              linestyle="--", marker="^", label="Regression (forecasted feats) — validation")
    l_reg_true_val, = ax.plot(test_idx, reg_true_tbl["Predicted Revenue"].values,
                              linestyle="--", marker="s", label="Regression (upper bound) — validation")

    # Future lines
    l_sarimax_fut, = ax.plot(future_idx, sarimax_future.values, marker="d", linewidth=2,
                             label=f"SARIMAX — forecast (+{fut_h}m)")
    l_reg_fut, = ax.plot(future_idx, reg_future, marker="P", linewidth=2,
                         label=f"Regression (forecasted feats) — forecast (+{fut_h}m)")

    # Optional upper-bound future using true/planned exog
    handles2 = [l_sarimax_val, l_reg_fore_val, l_reg_true_val, l_sarimax_fut, l_reg_fut]
    labels2  = ["Validation — SARIMAX", "Validation — Regression (forecasted feats)",
                "Validation — Regression (upper bound)",
                "Forecast — SARIMAX", "Forecast — Regression (forecasted feats)"]
    if (reg_true_model is not None) and (upper_future_features is not None):
        # Ensure REG_FEATURES exist
        X_upper = upper_future_features.copy()
        if "month" not in X_upper:
            X_upper["month"] = X_upper.index.month
        start_t = int(monthly["t"].iloc[-1]) + 1
        X_upper["t"] = np.arange(start_t, start_t + len(X_upper))
        X_upper["month_sin"] = np.sin(2*np.pi*X_upper["month"]/12.0)
        X_upper["month_cos"] = np.cos(2*np.pi*X_upper["month"]/12.0)
        X_upper = X_upper[REG_FEATURES]

        reg_upper_future = reg_true_model.predict(X_upper)
        l_reg_true_fut, = ax.plot(upper_future_features.index, reg_upper_future,
                                  linestyle="--", marker="s", linewidth=2,
                                  label="Regression (upper bound) — forecast")
        handles2.append(l_reg_true_fut)
        labels2.append("Forecast — Regression (upper bound)")

    # Vertical markers & axes
    ax.axvline(test_idx[0], color="grey", linestyle="--", alpha=0.7, label="Validation Start")
    ax.axvline(forecast_start, color="grey", linestyle=":" , alpha=0.9, label="Forecast Start")
    ax.set_title("Actual vs Models — Validation & Future Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Revenue (₪)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); ax.grid(alpha=0.2)

    # Split legends
    leg1 = ax.legend(handles=[l_actual], title="Actual", loc="upper left",
                     frameon=True, fancybox=True, borderpad=0.4)
    leg2 = ax.legend(handles=handles2, labels=labels2, title="Validation vs Forecast",
                     loc="upper right", bbox_to_anchor=(1.0, -0.22), ncol=2,
                     frameon=True, fancybox=True, borderpad=0.4)
    ax.add_artist(leg1)

    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(outpath, dpi=160)
    plt.show(); plt.close(fig)


# ----------------- FS: Build frame & feature importance -----------------
def build_fs_frame_from_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Create a numeric, order-level FS frame from raw orders."""
    df = orders.copy()

    # Ensure datetime
    if np.issubdtype(pd.Series(df["order_date"]).dtype, np.number):
        df["order_date"] = pd.to_datetime(df["order_date"], unit="ms", origin="unix")
    else:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Basic temporal features
    df["dow"] = df["order_date"].dt.weekday
    df["month"] = df["order_date"].dt.month
    df["is_wend"] = df["dow"].isin([5,6]).astype(int)

    # Booleans → int
    for b in ["is_online","coupon_used","is_return_order"]:
        if b in df.columns:
            df[b] = df[b].astype(int)

    # One-hot low-cardinality categoricals
    low_card_cols = [c for c in ["channel","region","top_department","top_size_group"] if c in df.columns]
    if low_card_cols:
        df = pd.get_dummies(df, columns=low_card_cols, prefix=low_card_cols, drop_first=False)

    # Frequency-encoding for high-cardinality categoricals
    high_card_cols = [c for c in ["store_id","top_category","top_location","top_region"] if c in df.columns]
    for col in high_card_cols:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq).fillna(0)

    # Drop id/text columns (keep line_amount + order_date)
    drop_cols = [c for c in ["order_id","customer_phone","items_id_list","year_month",
                             "store_id","top_category","top_location","top_region"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Select numeric features and target
    feature_cols = [c for c in df.columns if c not in ["order_date","line_amount"]]
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df["line_amount"].astype(float)

    df_fs = pd.concat([df[["order_date","line_amount"]], X], axis=1)
    return df_fs, list(X.columns), y


def compute_feature_importances(df_fs: pd.DataFrame, feature_cols: list[str], y: pd.Series) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    ada = AdaBoostRegressor(n_estimators=400, random_state=42)
    et  = ExtraTreesRegressor(n_estimators=400, random_state=42)

    ada.fit(df_fs[feature_cols], y)
    et.fit (df_fs[feature_cols], y)

    imp = pd.DataFrame({
        "feature": feature_cols,
        "imp_ada": ada.feature_importances_,
        "imp_et":  et.feature_importances_,
    })
    imp["imp_avg"] = (imp["imp_ada"] + imp["imp_et"]) / 2.0

    top10_ada = imp.sort_values("imp_ada", ascending=False).head(10)["feature"].tolist()
    top10_et  = imp.sort_values("imp_et",  ascending=False).head(10)["feature"].tolist()
    top10_avg = imp.sort_values("imp_avg", ascending=False).head(10)["feature"].tolist()

    return imp, top10_ada, top10_et, top10_avg


# ----------------- FS: Monthly aggregation & modeling -----------------
def aggregate_monthly_selected(df_fs: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    m = df_fs.copy()
    m["month_ts"] = m["order_date"].dt.to_period("M").dt.to_timestamp()

    # Target: monthly total revenue
    y_month = m.groupby("month_ts")["line_amount"].sum().rename("total_revenue")

    # Features: monthly MEANS of the selected order-level features
    X_month = m.groupby("month_ts")[selected].mean()

    out = X_month.join(y_month)
    return out


def fs_make_preds(monthly_sel: pd.DataFrame, features: list[str], test_h: int, fut_h: int,
                  future_true_features: pd.DataFrame | None = None) -> dict:
    """Train LinearRegression on monthly features → total_revenue.
       Returns validation preds (proper-forecast & upper) and future forecasts.
    """
    y = monthly_sel["total_revenue"].astype(float)
    X = monthly_sel[features].astype(float)

    idx_all  = y.index
    val_idx  = idx_all[-test_h:]
    fut_idx  = pd.date_range(idx_all[-1] + pd.offsets.MonthBegin(1), periods=fut_h, freq="MS")

    lin = LinearRegression().fit(X.iloc[:-test_h], y.iloc[:-test_h])

    # Validation upper bound (true features)
    X_val_true = X.iloc[-test_h:]
    pred_val_upper = lin.predict(X_val_true)

    # Validation proper-forecast: seasonal-naive per feature
    X_val_fore = pd.DataFrame(index=val_idx)
    for f in features:
        X_val_fore[f] = seasonal_naive(X.iloc[:-test_h][f], test_h)
    pred_val_fore = lin.predict(X_val_fore)

    # Future proper-forecast
    X_fut_fore = pd.DataFrame(index=fut_idx)
    for f in features:
        X_fut_fore[f] = seasonal_naive(X[f], fut_h)
    pred_fut_fore = lin.predict(X_fut_fore)

    # Future upper-bound (optional)
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
        "y_val": monthly_sel.iloc[-test_h:]["total_revenue"].values,
    }


def plot_fs_ada_vs_et(res_ada: dict, res_et: dict, res_upper: dict | None,
                      test_h: int, fut_h: int, outpath: Path):
    """One line per algorithm (Ada vs ET) across validation+forecast.
       Optional: dashed 'Upper Bound' line (validation + future).
    """
    idx_all = res_ada["idx_all"]; y_all = res_ada["y_all"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Actual
    l_actual, = ax.plot(idx_all, y_all, marker="o", linewidth=2, color="black", label="Actual")

    # AdaBoost line
    x_ada = list(res_ada["val_idx"]) + list(res_ada["fut_idx"])
    y_ada = list(res_ada["pred_val_fore"]) + list(res_ada["pred_fut_fore"])
    l_ada, = ax.plot(x_ada, y_ada, marker="P", linewidth=2, label="AdaBoost — validation+forecast")

    # ExtraTrees line
    x_et = list(res_et["val_idx"]) + list(res_et["fut_idx"])
    y_et = list(res_et["pred_val_fore"]) + list(res_et["pred_fut_fore"])
    l_et, = ax.plot(x_et, y_et, marker="D", linewidth=2, label="ExtraTrees — validation+forecast")

    handles_models = [l_ada, l_et]
    labels_models  = ["AdaBoost — validation+forecast", "ExtraTrees — validation+forecast"]

    # Optional upper-bound (validation + future)
    if res_upper is not None:
        x_up = list(res_upper["val_idx"]) + list(res_upper["fut_idx"])
        y_up = list(res_upper["pred_val_upper"]) + (list(res_upper["pred_fut_upper"]) if res_upper["pred_fut_upper"] is not None else [])
        l_up, = ax.plot(x_up, y_up, linestyle="--", marker="s", linewidth=2, label="Upper Bound — validation+forecast")
        handles_models.append(l_up)
        labels_models.append("Upper Bound — validation+forecast")

    # Markers & axes
    ax.axvline(res_ada["val_idx"][0], color="grey", linestyle="--", alpha=0.7, label="Validation Start")
    ax.axvline(res_ada["fut_idx"][0], color="grey", linestyle=":",  alpha=0.9, label="Forecast Start")
    ax.set_title("FS (Top-10) — AdaBoost vs ExtraTrees: Validation & Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Revenue (₪)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); ax.grid(alpha=0.2)

    # Split legends
    leg1 = ax.legend(handles=[l_actual], title="Actual", loc="upper left", bbox_to_anchor=(0.0, -0.22), frameon=True, fancybox=True, borderpad=0.4)
    leg2 = ax.legend(handles=handles_models, labels=labels_models, title="FS Models (validation+forecast)",
                     loc="upper right", bbox_to_anchor=(1.0, -0.22), ncol=2,
                     frameon=True, fancybox=True, borderpad=0.4)
    ax.add_artist(leg1)

    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(outpath, dpi=160)
    plt.show(); plt.close(fig)


# ----------------- Orchestration -----------------
def run_pipeline():
    # Load & monthly aggregate
    orders = load_orders(PATH)
    monthly = make_monthly(orders)

    # ===== Baselines =====
    sarimax_tbl, sarimax_metrics, sarimax_model = sarimax_revenue(monthly, TEST_H)
    reg_fore_tbl, reg_fore_metrics, reg_fore_model = regression_revenue(monthly, TEST_H, use_future_features=False)
    reg_true_tbl, reg_true_metrics, reg_true_model = regression_revenue(monthly, TEST_H, use_future_features=True)

    # Save tables
    sarimax_tbl.to_csv(OUTDIR/"sarimax_backtest.csv", index=False)
    reg_fore_tbl.to_csv(OUTDIR/"reg_backtest_forecasted_features.csv", index=False)
    reg_true_tbl.to_csv(OUTDIR/"reg_backtest_true_features.csv", index=False)

    # Individual prints for baseline regression variants
    print("[Regression — forecasted features] Validation table:", reg_fore_tbl.round(2))
    print(f"MAE={reg_fore_metrics[0]:,.0f}  MAPE={reg_fore_metrics[1]:.2%}")

    print("[Regression — upper bound] Validation table:", reg_true_tbl.round(2))
    print(f"MAE={reg_true_metrics[0]:,.0f}  MAPE={reg_true_metrics[1]:.2%}")

    # What-if on last TEST_H months using the upper-bound regression
    what_if_tbl = simulate_what_if(monthly, reg_true_model, TEST_H,
                                   orders_mult=WHAT_IF_ORDERS_MULT, aov_mult=WHAT_IF_AOV_MULT)
    what_if_tbl.to_csv(OUTDIR/"what_if_simulation.csv", index=False)

    # Accuracy summary (baselines)
    summary = pd.DataFrame({
        "Model": [
            "SARIMAX (revenue)",
            "Regression (forecasted Orders & AOV)",
            "Regression (true Orders & AOV) — upper bound",
        ],
        "MAE":  [sarimax_metrics[0], reg_fore_metrics[0], reg_true_metrics[0]],
        "MAPE": [sarimax_metrics[1], reg_fore_metrics[1], reg_true_metrics[1]],
    })
    summary.to_csv(OUTDIR/"accuracy_summary_baselines.csv", index=False)

    # Plots (baselines)
    plot_actual_vs_pred(monthly["total_revenue"], TEST_H,
                        sarimax_tbl["Predicted Revenue"].values,
                        "Monthly Revenue — Actual vs SARIMAX (Validation)",
                        OUTDIR/"plot_actual_vs_sarimax.png")

    # New: Actual vs Regression (Forecasted Features) — validation
    plot_actual_vs_pred(
        monthly["total_revenue"], TEST_H,
        reg_fore_tbl["Predicted Revenue"].values,
        "Monthly Revenue — Actual vs Regression (Forecasted Features) (Validation)",
        OUTDIR/"plot_actual_vs_reg_forecasted.png"
    )

    plot_actual_vs_pred(monthly["total_revenue"], TEST_H,
                        reg_true_tbl["Predicted Revenue"].values,
                        "Monthly Revenue — Actual vs Regression (Upper-Bound) (Validation)",
                        OUTDIR/"plot_actual_vs_reg_upper.png")

    plot_what_if(what_if_tbl, OUTDIR/"plot_what_if_baseline_vs_improved.png")

    # Combined chart: validation + future
    # Build optional future upper-bound exog (planned Orders/AOV uplift)
    X_naive  = make_future_exog(monthly, FUTURE_H)
    UPPER_FUT = pd.DataFrame(index=X_naive.index)
    UPPER_FUT["num_orders"]     = X_naive["num_orders"] * WHAT_IF_ORDERS_MULT
    UPPER_FUT["avg_order_value"] = X_naive["avg_order_value"] * WHAT_IF_AOV_MULT

    plot_all_models_one_figure(
        monthly=monthly,
        test_h=TEST_H,
        fut_h=FUTURE_H,
        sarimax_tbl=sarimax_tbl,
        reg_fore_tbl=reg_fore_tbl,
        reg_true_tbl=reg_true_tbl,
        reg_fore_model=reg_fore_model,
        outpath=OUTDIR / "plot_all_models_backtest_and_forecast.png",
        reg_true_model=reg_true_model,
        upper_future_features=UPPER_FUT,
    )

    # ===== Feature Selection (Ada/ET) =====
    df_fs, feature_cols_fs, y_fs = build_fs_frame_from_orders(orders)
    imp, top10_ada, top10_et, top10_avg = compute_feature_importances(df_fs, feature_cols_fs, y_fs)
    imp.sort_values("imp_avg", ascending=False).to_csv(OUTDIR/"feature_importances_ada_et.csv", index=False)

    # Monthly frames for each feature set
    monthly_ada = aggregate_monthly_selected(df_fs, top10_ada)
    monthly_et  = aggregate_monthly_selected(df_fs, top10_et)
    monthly_avg = aggregate_monthly_selected(df_fs, top10_avg)  # used for optional upper bound line

    # Optional future upper-bound features for top10_avg
    future_idx = pd.date_range(monthly_avg.index[-1] + pd.offsets.MonthBegin(1), periods=FUTURE_H, freq="MS")
    UPPER_FUT_FS = pd.DataFrame(index=future_idx)
    for f in top10_avg:
        UPPER_FUT_FS[f] = seasonal_naive(monthly_avg[f], FUTURE_H) * 1.10  # example uplift

    # Predictions
    res_ada = fs_make_preds(monthly_ada, top10_ada, TEST_H, FUTURE_H)
    res_et  = fs_make_preds(monthly_et,  top10_et,  TEST_H, FUTURE_H)
    res_up  = fs_make_preds(monthly_avg, top10_avg, TEST_H, FUTURE_H, UPPER_FUT_FS)

    # FS validation metrics table
    mae_ada, mape_ada = metrics(res_ada["y_val"], res_ada["pred_val_fore"])
    mae_et,  mape_et  = metrics(res_et["y_val"],  res_et["pred_val_fore"])
    mae_up,  mape_up  = metrics(res_up["y_val"],  res_up["pred_val_upper"])

    fs_summary = pd.DataFrame([
        {"Model": "FS AdaBoost (Top-10) — forecasted feats", "MAE": mae_ada, "MAPE": mape_ada},
        {"Model": "FS ExtraTrees (Top-10) — forecasted feats", "MAE": mae_et,  "MAPE": mape_et},
        {"Model": "FS Upper-Bound (Top-10 avg) — true feats", "MAE": mae_up,  "MAPE": mape_up},
    ])
    fs_summary.to_csv(OUTDIR/"accuracy_summary_fs.csv", index=False)

    # Plot: single line per algorithm (+ optional upper bound dashed)
    plot_fs_ada_vs_et(res_ada, res_et, res_up, TEST_H, FUTURE_H,
                      OUTDIR / "plot_fs_ada_vs_et_validation_and_forecast.png")

    # Console summary
    print("[Baselines] Accuracy summary:", summary.round(4))
    print("[FS] Top-10 features:")
    print("AdaBoost:", top10_ada)
    print("ExtraTrees:", top10_et)
    print("Average:", top10_avg)
    print("Saved outputs →", OUTDIR.resolve())

    # ================= Summary table (Validation + Forecast + What-If) =================
    rows = []

    def add_row(model_name: str, window: str, idx: pd.DatetimeIndex,
                actual: pd.Series | None, pred: pd.Series | np.ndarray,
                mae_val: float | None = None, mape_val: float | None = None,
                wif_orders: float | None = None, wif_aov: float | None = None,
                wif_base_mean: float | None = None, wif_impr_mean: float | None = None,
                wif_impr_total: float | None = None):
        pred_s = pd.Series(pred, index=idx)
        if actual is not None:
            actual_s = pd.Series(actual.values if hasattr(actual, 'values') else actual, index=idx)
            mae_ = mae_val if mae_val is not None else mean_absolute_error(actual_s, pred_s)
            mape_ = mape_val if mape_val is not None else mean_absolute_percentage_error(actual_s, pred_s)
            a_mean = float(actual_s.mean()); a_med = float(actual_s.median())
        else:
            mae_, mape_ = np.nan, np.nan
            a_mean, a_med = np.nan, np.nan
        rows.append({
            "Model": model_name,
            "Window": window,
            "Start": idx.min().strftime('%Y-%m'),
            "End": idx.max().strftime('%Y-%m'),
            "N_months": int(len(idx)),
            "Actual_Mean": a_mean,
            "Actual_Median": a_med,
            "Pred_Mean": float(pd.Series(pred, index=idx).mean()),
            "Pred_Median": float(pd.Series(pred, index=idx).median()),
            "MAE": mae_,
            "MAPE": mape_,
            "WIF_Orders_Mult": wif_orders,
            "WIF_AOV_Mult": wif_aov,
            "WIF_Baseline_Mean": wif_base_mean,
            "WIF_Improved_Mean": wif_impr_mean,
            "WIF_Improvement_Total": wif_impr_total,
        })

    # --- Validation windows (have actuals) ---
    test_idx = monthly.index[-TEST_H:]
    add_row("SARIMAX (revenue)", "Validation", test_idx,
            actual=monthly["total_revenue"].iloc[-TEST_H:],
            pred=sarimax_tbl["Predicted Revenue"].values,
            mae_val=sarimax_metrics[0], mape_val=sarimax_metrics[1])

    add_row("Regression (forecasted Orders & AOV)", "Validation", test_idx,
            actual=monthly["total_revenue"].iloc[-TEST_H:],
            pred=reg_fore_tbl["Predicted Revenue"].values,
            mae_val=reg_fore_metrics[0], mape_val=reg_fore_metrics[1])

    add_row("Regression (true Orders & AOV) — upper bound", "Validation", test_idx,
            actual=monthly["total_revenue"].iloc[-TEST_H:],
            pred=reg_true_tbl["Predicted Revenue"].values,
            mae_val=reg_true_metrics[0], mape_val=reg_true_metrics[1])

    # FS validation
    add_row("FS AdaBoost (Top-10)", "Validation", res_ada["val_idx"],
            actual=pd.Series(res_ada["y_val"], index=res_ada["val_idx"]),
            pred=res_ada["pred_val_fore"],
            mae_val=mean_absolute_error(res_ada["y_val"], res_ada["pred_val_fore"]),
            mape_val=mean_absolute_percentage_error(res_ada["y_val"], res_ada["pred_val_fore"]))

    add_row("FS ExtraTrees (Top-10)", "Validation", res_et["val_idx"],
            actual=pd.Series(res_et["y_val"], index=res_et["val_idx"]),
            pred=res_et["pred_val_fore"],
            mae_val=mean_absolute_error(res_et["y_val"], res_et["pred_val_fore"]),
            mape_val=mean_absolute_percentage_error(res_et["y_val"], res_et["pred_val_fore"]))

    add_row("FS Upper-Bound (Top-10 avg)", "Validation", res_up["val_idx"],
            actual=pd.Series(res_up["y_val"], index=res_up["val_idx"]),
            pred=res_up["pred_val_upper"],
            mae_val=mean_absolute_error(res_up["y_val"], res_up["pred_val_upper"]),
            mape_val=mean_absolute_percentage_error(res_up["y_val"], res_up["pred_val_upper"]))

    # --- Forecast windows (no actuals) ---
    # Recompute future baselines here to capture values for the table
    y_all = monthly["total_revenue"].astype(float)
    sarimax_full = SARIMAX(y_all, order=(1,1,1), seasonal_order=(1,1,1,12),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarimax_future = sarimax_full.forecast(steps=FUTURE_H)

    X_fut = make_future_exog(monthly, FUTURE_H)
    reg_future = reg_fore_model.predict(X_fut[REG_FEATURES])

    add_row("SARIMAX (revenue)", "Forecast", sarimax_future.index,
            actual=None, pred=sarimax_future.values)
    add_row("Regression (forecasted Orders & AOV)", "Forecast", X_fut.index,
            actual=None, pred=reg_future)

    add_row("FS AdaBoost (Top-10)", "Forecast", res_ada["fut_idx"], actual=None,
            pred=res_ada["pred_fut_fore"])
    add_row("FS ExtraTrees (Top-10)", "Forecast", res_et["fut_idx"], actual=None,
            pred=res_et["pred_fut_fore"])
    if res_up["pred_fut_upper"] is not None:
        add_row("FS Upper-Bound (Top-10 avg)", "Forecast", res_up["fut_idx"], actual=None,
                pred=res_up["pred_fut_upper"])

    # --- What-If summary ---
    # On the last TEST_H months (validation window) for the upper-bound regression
    wif_idx = what_if_tbl.index if isinstance(what_if_tbl.index, pd.DatetimeIndex) else test_idx
    wif_base_mean = float(what_if_tbl["Predicted Revenue (baseline)"].mean())
    wif_impr_mean = float(what_if_tbl["Predicted Revenue (improved)"].mean())
    wif_impr_total = float((what_if_tbl["Predicted Revenue (improved)"] - what_if_tbl["Predicted Revenue (baseline)"]).sum())

    add_row("What-If — Regression Upper-Bound", "Validation", test_idx,
            actual=monthly["total_revenue"].iloc[-TEST_H:],
            pred=what_if_tbl["Predicted Revenue (baseline)"].values,
            wif_orders=WHAT_IF_ORDERS_MULT, wif_aov=WHAT_IF_AOV_MULT,
            wif_base_mean=wif_base_mean, wif_impr_mean=wif_impr_mean,
            wif_impr_total=wif_impr_total)

    summary_all = pd.DataFrame(rows)
    # Save & print
    summary_path_xlsx = OUTDIR/"model_summary.xlsx"
    summary_path_csv  = OUTDIR/"model_summary.csv"
    try:
        summary_all.to_excel(summary_path_xlsx, index=False)
    except Exception as e:
        print("[WARN] Could not save XLSX (install openpyxl). Saving CSV instead.", e)
    summary_all.to_csv(summary_path_csv, index=False)
    print("Model summary (head):", summary_all.round(2).head(20))



    # ---- Extended summary: UB forecast row + WIF for every model ----
    try:
        rows2 = []
        uplift_k = WHAT_IF_ORDERS_MULT * WHAT_IF_AOV_MULT
        test_idx = monthly.index[-TEST_H:]
        train = monthly.iloc[:-TEST_H]
        test  = monthly.iloc[-TEST_H:]

        def _wif_stats(base, imp):
            base = np.asarray(base); imp = np.asarray(imp)
            return float(base.mean()), float(imp.mean()), float((imp - base).sum())

        def _add(model, window, idx, actual, pred, mae=None, mape=None, wif_base=None, wif_imp=None):
            if actual is not None:
                a_mean = float(pd.Series(actual, index=idx).mean())
                a_med  = float(pd.Series(actual, index=idx).median())
            else:
                a_mean = np.nan; a_med = np.nan
            wb, wi, wt = (np.nan, np.nan, np.nan)
            if (wif_base is not None) and (wif_imp is not None):
                wb, wi, wt = _wif_stats(wif_base, wif_imp)
            rows2.append({
                "Model": model, "Window": window,
                "Start": idx.min().strftime('%Y-%m'), "End": idx.max().strftime('%Y-%m'),
                "N_months": int(len(idx)),
                "Actual_Mean": a_mean, "Actual_Median": a_med,
                "Pred_Mean": float(np.mean(pred)), "Pred_Median": float(np.median(pred)),
                "MAE": mae if mae is not None else np.nan,
                "MAPE": mape if mape is not None else np.nan,
                "WIF_Orders_Mult": WHAT_IF_ORDERS_MULT, "WIF_AOV_Mult": WHAT_IF_AOV_MULT,
                "WIF_Baseline_Mean": wb, "WIF_Improved_Mean": wi, "WIF_Improvement_Total": wt,
            })

        # === Validation ===
        # SARIMAX (proxy WIF = proportional uplift)
        sarimax_val = sarimax_tbl["Predicted Revenue"].values
        _add("SARIMAX (revenue)", "Validation", test_idx,
             actual=monthly["total_revenue"].iloc[-TEST_H:], pred=sarimax_val,
             mae=sarimax_metrics[0], mape=sarimax_metrics[1],
             wif_base=sarimax_val, wif_imp=sarimax_val * uplift_k)

        # Regression (forecasted Orders & AOV) — proper forecast + WIF via exog uplift
        X_val_fore = test[REG_FEATURES].copy()
        X_val_fore.loc[:, "num_orders"]      = seasonal_naive(train["num_orders"], TEST_H)
        X_val_fore.loc[:, "avg_order_value"] = seasonal_naive(train["avg_order_value"], TEST_H)
        pred_rf_val = reg_fore_model.predict(X_val_fore)

        X_val_fore_imp = X_val_fore.copy()
        X_val_fore_imp["num_orders"]      *= WHAT_IF_ORDERS_MULT
        X_val_fore_imp["avg_order_value"] *= WHAT_IF_AOV_MULT
        pred_rf_val_imp = reg_fore_model.predict(X_val_fore_imp)

        _add("Regression (forecasted Orders & AOV)", "Validation", test_idx,
             actual=monthly["total_revenue"].iloc[-TEST_H:], pred=pred_rf_val,
             mae=reg_fore_metrics[0], mape=reg_fore_metrics[1],
             wif_base=pred_rf_val, wif_imp=pred_rf_val_imp)

        # Regression (true Orders & AOV) — upper bound + WIF via exog uplift
        X_val_true = test[REG_FEATURES].copy()
        pred_ru_val = reg_true_model.predict(X_val_true)
        X_val_true_imp = X_val_true.copy()
        X_val_true_imp["num_orders"]      *= WHAT_IF_ORDERS_MULT
        X_val_true_imp["avg_order_value"] *= WHAT_IF_AOV_MULT
        pred_ru_val_imp = reg_true_model.predict(X_val_true_imp)

        _add("Regression (true Orders & AOV) — upper bound", "Validation", test_idx,
             actual=monthly["total_revenue"].iloc[-TEST_H:], pred=pred_ru_val,
             mae=reg_true_metrics[0], mape=reg_true_metrics[1],
             wif_base=pred_ru_val, wif_imp=pred_ru_val_imp)

        # FS models (proxy WIF = proportional uplift)
        _add("FS AdaBoost (Top-10)", "Validation", res_ada["val_idx"],
             actual=res_ada["y_val"], pred=res_ada["pred_val_fore"],
             mae=mean_absolute_error(res_ada["y_val"], res_ada["pred_val_fore"]),
             mape=mean_absolute_percentage_error(res_ada["y_val"], res_ada["pred_val_fore"]),
             wif_base=res_ada["pred_val_fore"], wif_imp=np.array(res_ada["pred_val_fore"]) * uplift_k)

        _add("FS ExtraTrees (Top-10)", "Validation", res_et["val_idx"],
             actual=res_et["y_val"], pred=res_et["pred_val_fore"],
             mae=mean_absolute_error(res_et["y_val"], res_et["pred_val_fore"]),
             mape=mean_absolute_percentage_error(res_et["y_val"], res_et["pred_val_fore"]),
             wif_base=res_et["pred_val_fore"], wif_imp=np.array(res_et["pred_val_fore"]) * uplift_k)

        _add("FS Upper-Bound (Top-10 avg)", "Validation", res_up["val_idx"],
             actual=res_up["y_val"], pred=res_up["pred_val_upper"],
             mae=mean_absolute_error(res_up["y_val"], res_up["pred_val_upper"]),
             mape=mean_absolute_percentage_error(res_up["y_val"], res_up["pred_val_upper"]),
             wif_base=res_up["pred_val_upper"], wif_imp=np.array(res_up["pred_val_upper"]) * uplift_k)

        # === Forecast ===
        y_all = monthly["total_revenue"].astype(float)
        sarimax_full2 = SARIMAX(y_all, order=(1,1,1), seasonal_order=(1,1,1,12),
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        sarimax_future = sarimax_full2.forecast(steps=FUTURE_H)
        _add("SARIMAX (revenue)", "Forecast", sarimax_future.index,
             actual=None, pred=sarimax_future.values,
             wif_base=sarimax_future.values, wif_imp=sarimax_future.values * uplift_k)
        X_fut = make_future_exog(monthly, FUTURE_H)
        pred_rf_fut = reg_fore_model.predict(X_fut[REG_FEATURES])
        X_fut_imp = X_fut.copy()
        X_fut_imp["num_orders"]      *= WHAT_IF_ORDERS_MULT
        X_fut_imp["avg_order_value"] *= WHAT_IF_AOV_MULT
        pred_rf_fut_imp = reg_fore_model.predict(X_fut_imp[REG_FEATURES])
        _add("Regression (forecasted Orders & AOV)", "Forecast", X_fut.index,
             actual=None, pred=pred_rf_fut, wif_base=pred_rf_fut, wif_imp=pred_rf_fut_imp)

        # ADDED: UB Regression forecast (base = no exog uplift, improved = uplifted exog)
        pred_ru_fut_base = reg_true_model.predict(X_fut[REG_FEATURES])
        pred_ru_fut_imp  = reg_true_model.predict(X_fut_imp[REG_FEATURES])
        _add("Regression (true Orders & AOV) — upper bound", "Forecast", X_fut.index,
             actual=None, pred=pred_ru_fut_base, wif_base=pred_ru_fut_base, wif_imp=pred_ru_fut_imp)

        # FS forecasts (proxy WIF)
        _add("FS AdaBoost (Top-10)", "Forecast", res_ada["fut_idx"], actual=None,
             pred=res_ada["pred_fut_fore"], wif_base=res_ada["pred_fut_fore"],
             wif_imp=np.array(res_ada["pred_fut_fore"]) * uplift_k)

        _add("FS ExtraTrees (Top-10)", "Forecast", res_et["fut_idx"], actual=None,
             pred=res_et["pred_fut_fore"], wif_base=res_et["pred_fut_fore"],
             wif_imp=np.array(res_et["pred_fut_fore"]) * uplift_k)

        if res_up["pred_fut_upper"] is not None:
            _add("FS Upper-Bound (Top-10 avg)", "Forecast", res_up["fut_idx"], actual=None,
                 pred=res_up["pred_fut_upper"], wif_base=res_up["pred_fut_upper"],
                 wif_imp=np.array(res_up["pred_fut_upper"]) * uplift_k)

        summary_all2 = pd.DataFrame(rows2)
        summary_path_xlsx = OUTDIR/"model_summary.xlsx"
        summary_path_csv  = OUTDIR/"model_summary.csv"
        
        print(monthly.loc["2023-08":"2024-01", "total_revenue"].describe())

        
        try:
            summary_all2.to_excel(summary_path_xlsx, index=False)
        except Exception as e:
            print("[WARN] Could not save XLSX (install openpyxl). Saving CSV instead.", e)
        summary_all2.to_csv(summary_path_csv, index=False)
        print("\n[Extended] Model summary (head):\n", summary_all2.round(2).head(30))
    except Exception as e:
        print("[WARN] Extended summary failed:", e)


if __name__ == "__main__":
    run_pipeline()
