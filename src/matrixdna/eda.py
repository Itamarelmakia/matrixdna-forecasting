# ============================================
# MatrixDNA — Exploratory Data Analysis (EDA)
# Focused on forecasting drivers
# ============================================
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
CLEANED_XLSX = r"C:/Users/user/Desktop/MatrixDNA Technical Assignment\raw_data\cleaned_data/matrixdna_cleaned_data.xlsx"  # Update to your path if needed
OUT_DIR      = Path("outputs/eda"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# >>> NEW: plot toggles <<<
SHOW_PLOTS = True      # show every figure (plt.show)
SAVE_PLOTS = True      # save every figure to disk

# ---------- Helpers ----------
def safe_read_cleaned_excel(path):
    xl = pd.ExcelFile(path)
    tabs = {name.lower(): name for name in xl.sheet_names}
    orders = xl.parse(tabs.get("orders", "Orders"))
    items  = xl.parse(tabs.get("items", "Items"))
    cust   = xl.parse(tabs.get("customers", "Customers"))
    return orders, items, cust

def ensure_datetime(df, col="order_date"):
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def add_time_parts(df, date_col="order_date"):
    if date_col not in df.columns:
        return df
    df["year"]      = df[date_col].dt.year
    df["month_num"] = df[date_col].dt.month
    df["year_month"] = df[date_col].dt.to_period("M").astype(str)
    df["weekday"]   = df[date_col].dt.weekday
    df["month_name"] = df[date_col].dt.month_name()
    return df

def monthly_agg_orders(orders):
    if "order_date" not in orders.columns:
        return None
    rev_col = "line_amount" if "line_amount" in orders.columns else None
    gp = orders.dropna(subset=["order_date"]).copy()
    gp["year_month"] = gp["order_date"].dt.to_period("M").astype(str)
    out = gp.groupby("year_month").agg(
        monthly_orders=("order_id", "nunique") if "order_id" in orders.columns else ("year_month","size"),
        monthly_revenue=(rev_col, "sum") if rev_col else ("year_month","size")
    ).reset_index()
    return out

# >>> REPLACED: safe_plot now shows & saves every figure <<<
def safe_plot(title, fname, plot_func, *, show=None, save=None, block=False):
    _show = SHOW_PLOTS if show is None else show
    _save = SAVE_PLOTS if save is None else save
    try:
        fig = plt.figure()
        plot_func()
        plt.title(title)
        plt.tight_layout()

        # <<< NEW: rotate x-axis ticks if they look like dates >>>
        for label in plt.gca().get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

        if _save:
            (OUT_DIR / fname).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(OUT_DIR / fname, dpi=150)
        if _show:
            plt.show(block=block)
        if not block:
            plt.close(fig)
    except Exception as e:
        print(f"[SKIP] {title} → {e}")


def write_stats(d, fname="eda_stats.json"):
    with open(OUT_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

# ---------- Load ----------
orders, items, customers = safe_read_cleaned_excel(CLEANED_XLSX)
orders = ensure_datetime(orders, "order_date")
orders = add_time_parts(orders, "order_date")

# ---------- 1) Monthly trend & seasonality ----------
mon = monthly_agg_orders(orders)
stats_out = {}

if mon is not None and not mon.empty:
    # Monthly revenue line
    def _plot_rev():
        x = pd.to_datetime(mon["year_month"] + "-01")
        y = mon["monthly_revenue"].values
        plt.plot(x, y, marker="o")
        plt.xlabel("Month")
        plt.ylabel("Revenue (₪)")
    safe_plot("Monthly Revenue", "monthly_revenue.png", _plot_rev)

    # Monthly orders count
    def _plot_cnt():
        x = pd.to_datetime(mon["year_month"] + "-01")
        y = mon["monthly_orders"].values
        plt.plot(x, y, marker="o")
        plt.xlabel("Month")
        plt.ylabel("# Orders")
    safe_plot("Monthly Orders", "monthly_orders.png", _plot_cnt)
    
        
        
    def _plot_rev_and_cnt():
        x = pd.to_datetime(mon["year_month"] + "-01")
        y_rev = mon["monthly_revenue"].values
        y_cnt = mon["monthly_orders"].values
    
        fig, ax1 = plt.subplots()
    
        # Left Y axis (Revenue)
        ax1.plot(x, y_rev, color="tab:blue", marker="o", label="Revenue (₪)")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Revenue (₪)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
    
        # Rotate x-axis labels
        ax1.tick_params(axis="x", rotation=45)
    
        # Right Y axis (Orders)
        ax2 = ax1.twinx()
        ax2.plot(x, y_cnt, color="tab:orange", marker="s", label="# Orders")
        ax2.set_ylabel("# Orders", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
    
        # Title & legend
        fig.suptitle("Monthly Revenue & Orders")
        fig.tight_layout()
    
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    
        return fig
    
    safe_plot("Monthly Revenue & Orders", "monthly_rev_orders.png", _plot_rev_and_cnt)
    


    # ACF & seasonal decomposition (optional)
    try:
        from statsmodels.tsa.stattools import acf
        series = mon.set_index(pd.to_datetime(mon["year_month"] + "-01"))["monthly_revenue"].astype(float)
        series = series.asfreq("MS")
        acfs = acf(series.fillna(method="ffill").values, nlags=min(24, len(series)-1), fft=True)
        def _plot_acf():
            plt.stem(range(len(acfs)), acfs, use_line_collection=True)
            plt.xlabel("Lag (months)")
            plt.ylabel("ACF")
        safe_plot("ACF — Monthly Revenue", "acf_monthly_revenue.png", _plot_acf)
        stats_out["acf_head"] = list(np.round(acfs[:13], 3))
    except Exception as e:
        print("[SKIP] ACF:", e)

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        series = mon.set_index(pd.to_datetime(mon["year_month"] + "-01"))["monthly_revenue"].astype(float).asfreq("MS")
        result = seasonal_decompose(series.fillna(method="ffill"), model="additive", period=12)
        def _plot_trend():
            plt.plot(result.trend.index, result.trend.values)
            plt.xlabel("Month"); plt.ylabel("Trend")
        safe_plot("Seasonal Decomposition — Trend", "decomp_trend.png", _plot_trend)
        def _plot_seasonal():
            plt.plot(result.seasonal.index, result.seasonal.values)
            plt.xlabel("Month"); plt.ylabel("Seasonality")
        safe_plot("Seasonal Decomposition — Seasonal", "decomp_seasonal.png", _plot_seasonal)
        def _plot_resid():
            plt.plot(result.resid.index, result.resid.values)
            plt.xlabel("Month"); plt.ylabel("Residual")
        safe_plot("Seasonal Decomposition — Residual", "decomp_resid.png", _plot_resid)
    except Exception as e:
        print("[SKIP] Seasonal decomposition:", e)



def _plot_channel_revenue():
    if "channel" in orders.columns and "line_amount" in orders.columns:
        # Aggregate revenue per channel
        rev_per_channel = orders.groupby("channel")["line_amount"].sum().sort_values(ascending=False)

        # Plot
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            rev_per_channel,
            labels=rev_per_channel.index,
            autopct="%.1f%%",
            startangle=90,
            counterclock=False
        )
        ax.set_title("Revenue Share by Channel")
        plt.tight_layout()
        return fig

safe_plot("Revenue Share by Channel", "pie_channel_revenue.png", _plot_channel_revenue)



# ---------- 2) AOV by channel over time ----------
if {"order_date","channel"}.issubset(orders.columns):
    rev_col = "line_amount" if "line_amount" in orders.columns else None
    aov_df = orders.dropna(subset=["order_date","channel"]).copy()
    aov_df["year_month"] = aov_df["order_date"].dt.to_period("M").astype(str)
    agg = aov_df.groupby(["year_month","channel"]).agg(
        revenue=(rev_col, "sum") if rev_col else ("channel","size"),
        orders=("order_id","nunique") if "order_id" in orders.columns else ("channel","size")
    ).reset_index()
    agg["AOV"] = agg["revenue"] / np.where(agg["orders"]==0, 1, agg["orders"])

    def _plot_aov():
        for ch, sub in agg.groupby("channel"):
            x = pd.to_datetime(sub["year_month"] + "-01")
            plt.plot(x, sub["AOV"].values, marker="o", label=str(ch))
        plt.xlabel("Month"); plt.ylabel("Average Order Value (₪)")
        plt.legend()
    safe_plot("AOV by Channel over Time", "aov_by_channel.png", _plot_aov)

# ---------- 3) Region mix over time (stacked area) ----------
if {"order_date","region"}.issubset(orders.columns):
    reg = orders.dropna(subset=["order_date","region"]).copy()
    reg["year_month"] = reg["order_date"].dt.to_period("M").astype(str)
    pivot = reg.pivot_table(index="year_month", columns="region", values="order_id",
                            aggfunc="nunique" if "order_id" in reg.columns else "size", fill_value=0)
    pivot = pivot.sort_index()
    def _plot_region_area():
        x = pd.to_datetime(pivot.index + "-01")
        y = pivot.values
        plt.stackplot(x, y.T, labels=pivot.columns)
        plt.xlabel("Month"); plt.ylabel("# Orders")
        plt.legend(loc="upper left")
    safe_plot("Region Mix Over Time (Orders)", "region_mix_area.png", _plot_region_area)

# ---------- 4) Categories: top contributors ----------
def top_categories(items_df, orders_df, top_n=10):
    if "category" not in items_df.columns:
        return None, None
    if "line_amount" in items_df.columns:
        cat_rev = items_df.groupby("category")["line_amount"].sum().sort_values(ascending=False)
        mode = "revenue_items"
    elif {"order_id","line_amount"}.issubset(orders_df.columns):
        joined = items_df[["order_id","category"]].merge(
            orders_df[["order_id","line_amount"]], on="order_id", how="left")
        counts = joined.groupby("order_id")["category"].transform("count").replace(0, np.nan)
        joined["alloc"] = joined["line_amount"] / counts
        cat_rev = joined.groupby("category")["alloc"].sum().sort_values(ascending=False)
        mode = "revenue_allocated"
    else:
        cat_rev = items_df.groupby("category").size().sort_values(ascending=False)
        mode = "units"
    return cat_rev.head(top_n), mode

top10, top_mode = top_categories(items, orders)
if top10 is not None:
    def _plot_top10():
        plt.bar(top10.index.astype(str), top10.values)
        plt.xticks(rotation=45, ha="right")
        ylabel = "Revenue (₪)" if "rev" in top_mode else ("Units" if top_mode=="units" else "Revenue (₪)")
        plt.ylabel(ylabel)
        plt.xlabel("Category")
    safe_plot("Top Categories", "top_categories.png", _plot_top10)

# ---------- Category time series for top 5 (no merge needed) ----------
top5 = list(top10.index[:5])

if {"order_date", "category", "line_amount"}.issubset(items.columns):
    # Work directly on items dataframe
    cat_ts = items[items["category"].isin(top5)].copy()

    # Ensure order_date is datetime
    cat_ts["order_date"] = pd.to_datetime(cat_ts["order_date"], errors="coerce")
    cat_ts = cat_ts.dropna(subset=["order_date"])

    # Create year-month column
    cat_ts["year_month"] = cat_ts["order_date"].dt.to_period("M").astype(str)

    # Aggregate revenue per category per month
    ts = (cat_ts.groupby(["year_month","category"])["line_amount"]
                 .sum().reset_index())

    # Plot
    def _plot_cat_ts():
        for c in top5:
            sub = ts[ts["category"] == c]
            if sub.empty:
                continue
            x = pd.to_datetime(sub["year_month"] + "-01")
            plt.plot(x, sub["line_amount"].values, marker="o", label=str(c))
        plt.xlabel("Month")
        plt.ylabel("Revenue (₪)")
        plt.legend()
    safe_plot("Top Categories over Time", "top_categories_over_time.png", _plot_cat_ts)

elif {"order_date","category"}.issubset(items.columns):
    # Fallback: if no revenue column, count units
    cat_ts = items[items["category"].isin(top5)].copy()
    cat_ts["order_date"] = pd.to_datetime(cat_ts["order_date"], errors="coerce")
    cat_ts = cat_ts.dropna(subset=["order_date"])
    cat_ts["year_month"] = cat_ts["order_date"].dt.to_period("M").astype(str)

    ts = (cat_ts.groupby(["year_month","category"])
                 .size().reset_index(name="units"))

    def _plot_cat_ts_units():
        for c in top5:
            sub = ts[ts["category"] == c]
            if sub.empty:
                continue
            x = pd.to_datetime(sub["year_month"] + "-01")
            plt.plot(x, sub["units"].values, marker="o", label=str(c))
        plt.xlabel("Month")
        plt.ylabel("Units")
        plt.legend()
    safe_plot("Top Categories over Time", "top_categories_over_time.png", _plot_cat_ts_units)








# ---------- 5) Basket size distribution (p90) & correlation ----------
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr
import numpy as np

qty_col_candidates = [c for c in ["total_items_net","total_items","items_qty","quantity"] if c in orders.columns]
val_col = "line_amount" if "line_amount" in orders.columns else None

if qty_col_candidates and val_col:
    qcol = qty_col_candidates[0]

    # 1) Prepare data & compute 90th percentile to cut outliers
    qvals_all = pd.to_numeric(orders[qcol], errors="coerce").dropna()
    if len(qvals_all) > 0:
        p90 = float(np.percentile(qvals_all, 99))
        # keep only observations <= p90
        filt = orders[pd.to_numeric(orders[qcol], errors="coerce").le(p90)].copy()

        # integer bins from 0..p90 (inclusive)
        upper_int = max(1, int(np.floor(p90)))
        bins = np.arange(0, upper_int + 2)  # +1 for last edge, +1 more so the last integer has a full bin

        def _plot_hist_p90_by_channel():
            ax = plt.gca()
            if "channel" in filt.columns:
                channels = [str(c) for c in filt["channel"].dropna().unique()]
                for ch in channels:
                    sub = pd.to_numeric(filt.loc[filt["channel"] == ch, qcol], errors="coerce").dropna()
                    if len(sub) == 0:
                        continue
                    plt.hist(sub, bins=bins, alpha=0.6, label=ch)  # default MPL colors; legend per channel
                plt.legend(title="Channel")
            else:
                plt.hist(pd.to_numeric(filt[qcol], errors="coerce").dropna(), bins=bins, alpha=0.8)

            plt.xlabel("Items per Order")
            plt.ylabel("Frequency")
            plt.xlim(0, upper_int + 1)
            # nice integer ticks on x-axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.title(f"Basket Size Distribution (≤ p90 = {upper_int})")

        safe_plot("Basket Size Distribution by Channel",
                  "basket_size_hist_by_channel.png", _plot_hist_p90_by_channel)

        # 2) Correlation (computed on the same filtered data)
        tmp = filt[[qcol, val_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(tmp) > 5:
            rho, pval = spearmanr(tmp[qcol], tmp[val_col])
            stats_out["basket_value_spearman"] = {
                "rho": round(float(rho), 3),
                "pvalue": float(pval),
                "p90_cutoff": upper_int
            }
    else:
        print("[INFO] No numeric values found for basket size column:", qcol)


def _plot_rev_and_cnt():
    x = pd.to_datetime(mon["year_month"] + "-01")
    y_rev = mon["monthly_revenue"].values
    y_cnt = mon["monthly_orders"].values

    fig, ax1 = plt.subplots()

    # Left Y axis (Revenue)
    ax1.plot(x, y_rev, color="tab:blue", marker="o", label="Revenue (₪)")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Revenue (₪)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Rotate x-axis labels
    ax1.tick_params(axis="x", rotation=45)

    # Right Y axis (Orders)
    ax2 = ax1.twinx()
    ax2.plot(x, y_cnt, color="tab:orange", marker="s", label="# Orders")
    ax2.set_ylabel("# Orders", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Combine legends from both axes and place above title
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines + lines2,
        labels + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),  # push above plot & title
        ncol=2,
        frameon=True  # show border
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    return fig

safe_plot("Monthly Revenue & Orders", "monthly_rev_orders.png", _plot_rev_and_cnt)





import numpy as np
from scipy.stats import pearsonr

def _plot_orders_vs_revenue():
    x = mon["monthly_orders"].values
    y = mon["monthly_revenue"].values

    fig, ax = plt.subplots()

    # Scatter plot
    ax.scatter(x, y, color="tab:blue", alpha=0.7, label="Monthly data")
    ax.set_xlabel("# Orders (Monthly)")
    ax.set_ylabel("Revenue (₪ Monthly)")
    ax.set_title("Correlation: Monthly Orders vs Revenue")

    # Fit linear regression line
    coeffs = np.polyfit(x, y, deg=1)  # degree=1 = linear
    poly_eq = np.poly1d(coeffs)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = poly_eq(x_line)
    ax.plot(x_line, y_line, color="red", linewidth=2, label="Fitted Line")

    # Correlation coefficient
    r, pval = pearsonr(x, y)
    ax.text(0.05, 0.95,
            f"Pearson r = {r:.3f}\n(p={pval:.3g})",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.legend()
    fig.tight_layout()
    return fig

safe_plot("Monthly Orders vs Revenue", "orders_vs_revenue.png", _plot_orders_vs_revenue)


import numpy as np
from scipy.stats import pearsonr

def _plot_orders_vs_revenue_by_channel():
    # --- Build monthly aggregates by channel ---
    if "order_date" not in orders.columns or "line_amount" not in orders.columns or "channel" not in orders.columns:
        raise ValueError("Required columns: 'order_date', 'line_amount', 'channel' (and ideally 'order_id').")

    df = orders.dropna(subset=["order_date", "channel"]).copy()
    df["year_month"] = df["order_date"].dt.strftime("%Y-%m")
    grp = df.groupby(["channel", "year_month"])

    if "order_id" in df.columns:
        mon_ch = grp.agg(
            monthly_orders=("order_id", "nunique"),
            monthly_revenue=("line_amount", "sum")
        ).reset_index()
    else:
        # Fallback if order_id is missing
        mon_ch = grp.size().rename("monthly_orders").to_frame().join(
            grp["line_amount"].sum().rename("monthly_revenue")
        ).reset_index()

    # --- Plot per channel on same axes ---
    fig, ax = plt.subplots()
    legend_entries = []

    for ch, sub in mon_ch.groupby("channel"):
        x = sub["monthly_orders"].values.astype(float)
        y = sub["monthly_revenue"].values.astype(float)

        # Skip channels with <2 points
        if len(x) < 2 or np.all(x == x[0]):
            ax.scatter(x, y, alpha=0.7, label=f"{ch} (n={len(x)})")
            continue

        # Scatter
        sc = ax.scatter(x, y, alpha=0.7)

        # Fit line (linear)
        coeffs = np.polyfit(x, y, deg=1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, poly(x_line), linewidth=2)

        # Correlation
        r, p = pearsonr(x, y)
        legend_entries.append((sc, f"{ch} (r={r:.2f}, n={len(x)})"))

    ax.set_xlabel("# Orders (Monthly)")
    ax.set_ylabel("Revenue (₪ Monthly)")
    ax.set_title("Monthly Orders vs Revenue by Channel (with fit)")

    # Build legend (above title) with border
    handles, labels = zip(*legend_entries) if legend_entries else ([], [])
    leg = ax.legend(handles, labels, loc="lower center",
                    bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True)
    if leg is not None:
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    return fig

safe_plot("Orders vs Revenue by Channel", "orders_vs_revenue_by_channel.png", _plot_orders_vs_revenue_by_channel)





# ---------- 6) Weekday x Month heatmaps (calendar effects) ----------
if "order_date" in orders.columns:
    cal = orders.dropna(subset=["order_date"]).copy()
    cal["month_num"]   = cal["order_date"].dt.month

    # Map weekday to Sun=1, Mon=2, ..., Sat=7
    # pandas: Monday=0 ... Sunday=6  -> ((d+1)%7)+1 gives Sun=1, Mon=2, ..., Sat=7
    cal["weekday_num"] = ((cal["order_date"].dt.dayofweek + 1) % 7) + 1

    # Helper to build a (7 x up-to-12) matrix sorted nicely
    def _make_mat(df, value_col=None, agg="nunique"):
        # value_col: "order_id" for counts (nunique), "line_amount" for sum, or None with agg="size"
        if value_col is not None:
            mat = df.pivot_table(index="weekday_num", columns="month_num",
                                 values=value_col, aggfunc=agg, fill_value=0)
        else:
            mat = df.pivot_table(index="weekday_num", columns="month_num",
                                 aggfunc="size", fill_value=0)
        # sort rows 1..7 and columns 1..12
        mat = mat.reindex(index=[1,2,3,4,5,6,7])
        mat = mat.reindex(columns=sorted(mat.columns))
        return mat

    # ---- Overall: # Orders heatmap
    if "order_id" in cal.columns:
        mat_orders = _make_mat(cal, value_col="order_id", agg="nunique")
    else:
        mat_orders = _make_mat(cal, value_col=None, agg="size")

    def _plot_heat_orders():
        plt.imshow(mat_orders.values, aspect="auto")
        plt.xticks(range(len(mat_orders.columns)), mat_orders.columns)
        plt.yticks(range(len(mat_orders.index)), mat_orders.index)
        plt.xlabel("Month")
        plt.ylabel("Weekday (Sun=1 … Sat=7)")
        plt.title("Calendar Heatmap — # Orders")
        plt.colorbar(label="# Orders")
    safe_plot("Calendar Heatmap — Orders", "calendar_heatmap_orders.png", _plot_heat_orders)

    # ---- Per-channel heatmaps: (a) # Orders, (b) Revenue (₪) if available
    if "channel" in cal.columns:
        for ch in sorted(cal["channel"].dropna().unique()):
            cal_ch = cal[cal["channel"] == ch]

            # (a) # Orders per channel
            if "order_id" in cal_ch.columns:
                mat_ch_orders = _make_mat(cal_ch, value_col="order_id", agg="nunique")
            else:
                mat_ch_orders = _make_mat(cal_ch, value_col=None, agg="size")

            def _plot_heat_ch_orders(ch_name=ch, M=mat_ch_orders):
                plt.imshow(M.values, aspect="auto")
                plt.xticks(range(len(M.columns)), M.columns)
                plt.yticks(range(len(M.index)), M.index)
                plt.xlabel("Month")
                plt.ylabel("Weekday (Sun=1 … Sat=7)")
                plt.title(f"Calendar Heatmap — # Orders (Channel: {ch_name})")
                plt.colorbar(label="# Orders")
            safe_plot(f"Calendar Heatmap — Orders — {ch}", f"calendar_heatmap_orders_{str(ch)}.png", _plot_heat_ch_orders)

            # (b) Revenue per channel (optional if line_amount exists)
            if "line_amount" in cal_ch.columns:
                mat_ch_rev = cal_ch.pivot_table(index="weekday_num", columns="month_num",
                                                values="line_amount", aggfunc="sum", fill_value=0)
                mat_ch_rev = mat_ch_rev.reindex(index=[1,2,3,4,5,6,7])
                mat_ch_rev = mat_ch_rev.reindex(columns=sorted(mat_ch_rev.columns))

                def _plot_heat_ch_rev(ch_name=ch, M=mat_ch_rev):
                    plt.imshow(M.values, aspect="auto")
                    plt.xticks(range(len(M.columns)), M.columns)
                    plt.yticks(range(len(M.index)), M.index)
                    plt.xlabel("Month")
                    plt.ylabel("Weekday (Sun=1 … Sat=7)")
                    plt.title(f"Calendar Heatmap — Revenue (₪) (Channel: {ch_name})")
                    plt.colorbar(label="Revenue (₪)")
                safe_plot(f"Calendar Heatmap — Revenue — {ch}", f"calendar_heatmap_revenue_{str(ch)}.png", _plot_heat_ch_rev)




# ---------- 7) Correlation between items/order and revenue ----------
import seaborn as sns
from scipy.stats import spearmanr

qty_col_candidates = [c for c in ["total_items_net","total_items","items_qty","quantity"] if c in orders.columns]
val_col = "line_amount" if "line_amount" in orders.columns else None

if qty_col_candidates and val_col:
    qcol = qty_col_candidates[0]

    # Drop missing values and filter items/order <= 20
    corr_df = (
        orders.dropna(subset=[qcol, val_col, "channel"])
              .copy()
    )
    corr_df[qcol] = corr_df[qcol].astype(int)
    corr_df[val_col] = corr_df[val_col].astype(float)
    corr_df = corr_df[corr_df[qcol] <= 20]

    # ---- Scatter plot with regression line, color legend by channel
    def _plot_corr():
        sns.lmplot(
            data=corr_df,
            x=qcol, y=val_col,
            hue="channel",
            height=6, aspect=1.2,
            scatter_kws={"alpha":0.5, "s":30},
            line_kws={"lw":2}
        )
        plt.xlabel("Items per Order (≤ 20)")
        plt.ylabel("Revenue (₪)")
        plt.title("Correlation between Items per Order and Revenue\n(by Channel, ≤ 20 items)")
        # Force x-axis ticks 1..20
        plt.xticks(range(1, 21, 1))
    safe_plot("Items vs Revenue by Channel", "items_vs_revenue_channel.png", _plot_corr)

    # ---- Compute correlation per channel
    channel_corrs = {}
    for ch, sub in corr_df.groupby("channel"):
        if len(sub) > 5:
            rho, pval = spearmanr(sub[qcol], sub[val_col])
            channel_corrs[ch] = {"spearman_rho": round(float(rho),3),
                                 "pvalue": round(float(pval),4),
                                 "n": len(sub)}
    # Overall correlation
    if len(corr_df) > 5:
        rho, pval = spearmanr(corr_df[qcol], corr_df[val_col])
        channel_corrs["ALL"] = {"spearman_rho": round(float(rho),3),
                                "pvalue": round(float(pval),4),
                                "n": len(corr_df)}

    stats_out["items_revenue_corr"] = channel_corrs




# ---------- 7) Correlation between items/order and revenue (Polynomial Fit) ----------
import seaborn as sns
from scipy.stats import spearmanr

qty_col_candidates = [c for c in ["total_items_net","total_items","items_qty","quantity"] if c in orders.columns]
val_col = "line_amount" if "line_amount" in orders.columns else None

if qty_col_candidates and val_col:
    qcol = qty_col_candidates[0]

    # Drop missing values and filter items/order <= 20
    corr_df = (
        orders.dropna(subset=[qcol, val_col, "channel"])
              .copy()
    )
    corr_df[qcol] = corr_df[qcol].astype(int)
    corr_df[val_col] = corr_df[val_col].astype(float)
    corr_df = corr_df[corr_df[qcol] <= 20]

    # ---- Polynomial regression (order=2) per channel
    def _plot_poly_corr():
        sns.lmplot(
            data=corr_df,
            x=qcol, y=val_col,
            hue="channel",
            order=2,                 # quadratic fit
            height=6, aspect=1.2,
            scatter_kws={"alpha":0.5, "s":30},
            line_kws={"lw":2}
        )
        plt.xlabel("Items per Order (≤ 20)")
        plt.ylabel("Revenue (₪)")
        plt.title("Polynomial (Quadratic) Fit: Items vs Revenue by Channel")
        plt.xticks(range(1, 21, 1))
    safe_plot("Items vs Revenue by Channel (Quadratic)", 
              "items_vs_revenue_channel_poly2.png", 
              _plot_poly_corr)

    # ---- Compute correlation per channel (still Spearman, monotonic)
    channel_corrs = {}
    for ch, sub in corr_df.groupby("channel"):
        if len(sub) > 5:
            rho, pval = spearmanr(sub[qcol], sub[val_col])
            channel_corrs[ch] = {"spearman_rho": round(float(rho),3),
                                 "pvalue": round(float(pval),4),
                                 "n": len(sub)}
    if len(corr_df) > 5:
        rho, pval = spearmanr(corr_df[qcol], corr_df[val_col])
        channel_corrs["ALL"] = {"spearman_rho": round(float(rho),3),
                                "pvalue": round(float(pval),4),
                                "n": len(corr_df)}

    stats_out["items_revenue_corr_poly2"] = channel_corrs
    



# ---------- 7) Statistical tests tied to forecasting ----------
from collections import defaultdict
tests = defaultdict(dict)

# Online vs Store AOV
if {"channel","order_id"}.issubset(orders.columns) and val_col:
    aov = orders.groupby(["order_id","channel"])[val_col].sum().reset_index()
    on  = aov[aov["channel"].astype(str).str.lower().eq("online")][val_col].dropna().values
    st  = aov[~aov["channel"].astype(str).str.lower().eq("online")][val_col].dropna().values
    if len(on) > 5 and len(st) > 5:
        from scipy.stats import ttest_ind, mannwhitneyu
        tests["aov_online_vs_store"]["t_test"] = {
            "stat": float(ttest_ind(on, st, equal_var=False, nan_policy="omit").statistic),
            "pvalue": float(ttest_ind(on, st, equal_var=False, nan_policy="omit").pvalue)
        }
        mw = mannwhitneyu(on, st, alternative="two-sided")
        tests["aov_online_vs_store"]["mann_whitney"] = {"stat": float(mw.statistic), "pvalue": float(mw.pvalue)}

# Month effect on revenue (seasonality evidence)
if {"order_date", val_col}.issubset(orders.columns):
    rev = orders.dropna(subset=["order_date", val_col]).copy()
    rev["month_num"] = rev["order_date"].dt.month
    groups = [g[val_col].values for _, g in rev.groupby("month_num")]
    try:
        from scipy.stats import kruskal
        kw = kruskal(*[g for g in groups if len(g)>0])
        tests["month_effect_revenue_kruskal"] = {"stat": float(kw.statistic), "pvalue": float(kw.pvalue)}
    except Exception as e:
        print("[SKIP] Kruskal:", e)

# Stationarity (ADF) on monthly revenue
if mon is not None and "monthly_revenue" in mon.columns and len(mon) >= 8:
    try:
        from statsmodels.tsa.stattools import adfuller
        series = mon["monthly_revenue"].astype(float).values
        adf = adfuller(series, autolag="AIC")
        tests["adf_monthly_revenue"] = {
            "adf_stat": float(adf[0]), "pvalue": float(adf[1]),
            "nobs": int(adf[3])
        }
    except Exception as e:
        print("[SKIP] ADF:", e)




import pandas as pd
import matplotlib.pyplot as plt

orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
orders["month"] = orders["order_date"].dt.to_period("M").dt.to_timestamp()

def _boxplot_monthly_by(col, title, filename):
    grouped = (
        orders.groupby([col, "month"])["line_amount"]
        .sum()
        .reset_index()
        .dropna(subset=[col, "line_amount"])
    )

    # Compute medians by group
    medians = (
        grouped.groupby(col)["line_amount"]
        .median()
        .sort_values()
    )

    # Order labels by median ascending
    labels = medians.index.tolist()
    data = [grouped.loc[grouped[col] == v, "line_amount"] for v in labels]

    def _draw():
        plt.boxplot(data, labels=labels, patch_artist=True)
        plt.ylabel("Monthly Revenue (₪)")
        plt.xlabel(col)
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        hue="region", 
        plt.tight_layout()

    safe_plot(title, filename, _draw)

# ---------- Run the three boxplots ----------
if "channel" in orders.columns:
    _boxplot_monthly_by("channel",
                        "Monthly Revenue Variability by Channel",
                        "boxplot_channel_monthly.png")

if "region" in orders.columns:
    _boxplot_monthly_by("region",
                        "Monthly Revenue Variability by Region",
                        "boxplot_region_monthly.png")

if "store_id" in orders.columns:
    _boxplot_monthly_by("store_id",
                        "Monthly Revenue Variability by Store ID",
                        "boxplot_store_monthly.png")





# === Horizontal boxplot of monthly revenue per store_id, colored by region ===
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure dates exist
orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
orders["month"] = orders["order_date"].dt.to_period("M").dt.to_timestamp()

def _boxplot_store_by_region_h():
    # 1) Aggregate to monthly revenue per store & region
    df = (
        orders
        .dropna(subset=["store_id", "region", "line_amount"])
        .groupby(["store_id", "region", "month"], as_index=False)["line_amount"]
        .sum()
    )

    # 2) Order stores by (region, median monthly revenue)
    med = (
        df.groupby(["store_id", "region"])["line_amount"]
          .median()
          .reset_index()
          .sort_values(["region", "line_amount"], ascending=[True, True])
    )
    store_order = med["store_id"].tolist()

    # 3) Plot: horizontal, one box per store_id, colored by its region
    plt.figure(figsize=(12, max(6, 0.45 * len(store_order))), dpi=180)
    sns.boxplot(
        data=df,
        y="store_id", x="line_amount",
        order=store_order,
        hue="region",
        dodge=False,          # one box per store, just colored by region
        orient="h",
        linewidth=0.8,
        fliersize=2
    )

    plt.xlabel("Monthly Revenue (₪)")
    plt.ylabel("store_id")
    plt.title("Monthly Revenue Variability by Store ID — colored by Region")
    plt.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

safe_plot("Monthly Revenue Variability by Store ID — colored by Region",
          "boxplot_store_monthly_region_h.png",
          _boxplot_store_by_region_h)


stats_out["tests"] = tests
write_stats(stats_out)



print(f"EDA complete. Figures + stats written to: {OUT_DIR.resolve()}")
