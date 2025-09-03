"""
MatrixDNA Technical Assignment – Data Cleaning & Pre‑processing
------------------------------------------------------------

This script performs an initial loading, inspection and basic cleaning of the raw
data provided for the MatrixDNA technical assignment.  It is designed to run
inside Spyder or any other Python IDE.  Adjust the ``DATA_DIR`` variable to
point at the folder where your CSV files live.  The script loads three
tables (``item_spec.csv``, ``order_spec.csv`` and ``customer_spec.csv``),
examines their structure, reports on missing values and performs a few
reasonable cleaning steps:

* Parsing dates into ``datetime`` objects.
* Converting numeric columns to appropriate dtypes (e.g., floats to ints where
  counts are expected).
* Filling missing categorical values with the label ``"Unknown"`` to avoid
  dropping rows or introducing invalid categories.
* Filling missing numeric customer features that represent counts or amounts
  with zeros (e.g., number of orders in the last 3 months); for other
  numeric features we leave NaNs untouched for now.
* Dropping duplicate rows based on key columns.

After cleaning, the script outputs a few basic summaries (shape, head,
missing‑value counts) and writes the cleaned data back to disk in a
``cleaned_data`` subdirectory for later analysis.

To run this script, simply execute it in Spyder or from the command line:

>>> python data_cleaning.py

The code uses only pandas and numpy.  No external dependencies are required.
"""

import os
import pandas as pd
import numpy as np
from ast import literal_eval
import re



def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files from the specified directory.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing ``item_spec.csv``, ``order_spec.csv`` and
        ``customer_spec.csv``.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing (item_spec, order_spec, customer_spec).
    """
    print(f"Loading data from {data_dir}\n")
    item_path = os.path.join(data_dir, "item_spec.csv")
    order_path = os.path.join(data_dir, "order_spec.csv")
    cust_path = os.path.join(data_dir, "customer_spec.csv")

    item_spec = pd.read_csv(item_path)
    order_spec = pd.read_csv(order_path)
    customer_spec = pd.read_csv(cust_path)

    return item_spec, order_spec, customer_spec


def report_missing(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Generate a missing value summary for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyse.
    name : str
        Label for the DataFrame (used in prints).

    Returns
    -------
    pd.DataFrame
        A summary table with columns 'column', 'missing_count' and 'missing_%'.
    """
    total = len(df)
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / total * 100).round(1)
    summary = pd.DataFrame({
        'column': missing_counts.index,
        'missing_count': missing_counts.values,
        'missing_%': missing_pct.values,
    })
    print(f"Missing value summary for {name}:")
    print(summary.loc[summary['missing_count'] > 0].sort_values('missing_%', ascending=False))
    print("\n")
    return summary


def convert_types(order_df: pd.DataFrame) -> pd.DataFrame:
    """Convert data types for the order table.

    - Parse ``order_date`` to datetime.
    - Convert numeric columns representing counts to integers.

    Parameters
    ----------
    order_df : pd.DataFrame
        The raw order dataframe.

    Returns
    -------
    pd.DataFrame
        A copy of ``order_df`` with updated dtypes.
    """
    df = order_df.copy()
    # Parse date column
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Identify numeric columns storing counts (they should be floats but without decimals)
    count_cols = [
        'total_items_net', 'num_items', 'num_products', 'quantity'
    ]
    for col in count_cols:
        if col in df.columns:
            # Convert to numeric first to avoid dtype errors, then drop non‑finite values
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Coerce missing numeric values to NaN; we'll handle them later if necessary
            df[col] = df[col].astype('Int64')
    return df


def clean_categorical(order_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values for categorical columns with 'Unknown'.

    We identify a set of likely categorical columns in the order dataframe.  If
    any of these contain missing values, we replace them with the literal
    string 'Unknown'.  You can extend or modify this list based on the actual
    schema of your data.

    Parameters
    ----------
    order_df : pd.DataFrame
        The orders dataframe to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned dataframe with missing categorical values filled.
    """
    df = order_df.copy()
    cat_cols = ['top_location', 'channel', 'gender']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    return df


def fill_online_store_region(df: pd.DataFrame) -> pd.DataFrame:
    """For item_spec rows flagged as online, fill ``store_id`` and ``region``.

    Some items are sold via an online channel rather than through a physical
    store.  In such cases, the ``store_id`` should be set to the string
    ``"Online"`` and ``region`` should be filled with ``"אונליין"`` (Hebrew for
    ``online``).  This function operates in place on the provided DataFrame and
    returns it for convenience.

    Parameters
    ----------
    df : pd.DataFrame
        The item specification DataFrame containing ``channel``, ``store_id`` and
        ``region`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated ``store_id`` and ``region`` for online items.
    """
    # Ensure the comparison is case‑insensitive and skip missing values
    mask = df['channel'].str.lower().eq('online')
    df.loc[mask, 'store_id'] = 'Online'
    df.loc[mask, 'region'] = 'Online'



    # ---- Convert Hebrew cities in region to English ----
    hebrew_to_english = {
        'מרכז': 'Central',
        'ירושלים': 'Jerusalem',
        'דרום': 'South',
        'חיפה': 'Haifa',
    }
    df['region'] = df['region'].replace(hebrew_to_english)

    return df


def fill_missing_category_sku_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing 'category' in item_spec with a 3-step strategy:
      (1) Use SKU→category mapping (from rows where category is known)
      (2) Use subcategory→category mode mapping
      (3) Heuristic from subcategory text (strip colors/sizes)
    Prints NA counts after each step.
    """
    if 'category' not in df.columns:
        df['category'] = pd.NA
    if 'subcategory' not in df.columns:
        # nothing useful we can do without subcategory; still try SKU mapping if present
        df['subcategory'] = pd.NA

    missing_before = int(df['category'].isna().sum())
    print(f"[category] missing before: {missing_before}")

    # ---------- (1) Fill by SKU mapping first ----------
    if 'sku' in df.columns:
        known_sku = df.loc[df['category'].notna() & df['sku'].notna(), ['sku', 'category']]
        if not known_sku.empty:
            # mode per SKU (robust if there are rare conflicts)
            sku_map = (
                known_sku.groupby('sku')['category']
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
                .dropna()
                .to_dict()
            )
            mask_na = df['category'].isna() & df['sku'].notna()
            filled_from_sku = df.loc[mask_na, 'sku'].map(sku_map)
            df.loc[mask_na, 'category'] = filled_from_sku
    missing_after_sku = int(df['category'].isna().sum())
    print(f"[category] missing after SKU map: {missing_after_sku}")

    # ---------- (2) Fill by subcategory→category mode ----------
    known = df[df['category'].notna() & df['subcategory'].notna()]
    if not known.empty:
        subcat_map = (
            known.groupby('subcategory')['category']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            .dropna()
            .to_dict()
        )
        mask_na = df['category'].isna() & df['subcategory'].notna()
        df.loc[mask_na, 'category'] = df.loc[mask_na, 'subcategory'].map(subcat_map)
    missing_after_subcat = int(df['category'].isna().sum())
    print(f"[category] missing after subcategory map: {missing_after_subcat}")

    # ---------- (3) Heuristic from subcategory text ----------
    colours = {
        'white','black','beige','turquoise','blue','red','green','pink',
        'yellow','orange','purple','grey','gray','brown','navy','gold',
        'silver','cream','multi','khaki','burgundy','turqoise'
    }
    size_tokens = {'xs','s','m','l','xl','xxl','xxxl','one-size','onesize'}
    size_regex = re.compile(r'^\d+([x×]\d+)?$|^\d+(\.\d+)?(ml|l|cm|mm|kg|g)$', re.IGNORECASE)

    def _heuristic_from_subcat(subcat: str):
        if pd.isna(subcat):
            return np.nan
        # take part before first comma
        first_part = str(subcat).split(',')[0].strip()
        words = first_part.split()
        # remove trailing color/size tokens
        while words:
            w = words[-1].lower()
            if (w in colours) or (w in size_tokens) or size_regex.match(w):
                words = words[:-1]
            else:
                break
        if words:
            return ' '.join(w.capitalize() for w in words)
        return np.nan

    mask_na = df['category'].isna() & df['subcategory'].notna()
    if mask_na.any():
        df.loc[mask_na, 'category'] = df.loc[mask_na, 'subcategory'].apply(_heuristic_from_subcat)

    missing_after_heur = int(df['category'].isna().sum())
    print(f"[category] missing after heuristic: {missing_after_heur}")

    return df

def enrich_orders_with_online_and_topcat(orders_df: pd.DataFrame, items_clean: pd.DataFrame) -> pd.DataFrame:
    """
    - store_id = "Online" when:
        * is_online is True (accepts 'true'/'1'), OR
        * channel contains "Online" (list, stringified list, or plain string)
    - Fill missing region with "אונליין" (STRING).
    - For missing top_category: concatenate categories from items_clean via items_id_list
      (prefers items_clean.top_category, else items_clean.category).
    - Accept either 'item_id' or 'sku' as the item key in items_clean.
    - Normalize: convert list-ish values in store_id/region to STRINGS (["מרכז"] -> "מרכז").
    - De-dup comma-separated strings in top_region, top_location, and top_category.
    """
    df = orders_df.copy()

    # ---- helpers ----
    def _truthy(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and not pd.isna(v):
            return int(v) == 1
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "y", "yes", "t"}
        return False

    def _to_list(val):
        if isinstance(val, list):
            return val
        if pd.isna(val):
            return []
        s = str(val).strip()
        try:
            parsed = literal_eval(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [t.strip() for t in s.split(",") if t.strip()]

    def _listish_to_string(val):
        """
        Convert list or stringified-list like ["מרכז"] -> "מרכז".
        If list has multiple tokens -> join with ", ".
        If already a plain string, return as-is.
        """
        if isinstance(val, list):
            toks = [str(t).strip() for t in val if str(t).strip()]
            if not toks:
                return pd.NA
            return toks[0] if len(toks) == 1 else ", ".join(toks)
        if pd.isna(val):
            return pd.NA
        s = str(val).strip()
        # try parsing stringified list
        try:
            parsed = literal_eval(s)
            if isinstance(parsed, list):
                toks = [str(t).strip() for t in parsed if str(t).strip()]
                if not toks:
                    return pd.NA
                return toks[0] if len(toks) == 1 else ", ".join(toks)
        except Exception:
            pass
        return s

    def _dedup_comma_string(x):
        """Split on ',', strip, drop dups (order-preserving). Join ', ' if >1; else keep single."""
        if pd.isna(x):
            return x
        parts = [p.strip() for p in str(x).split(",") if p.strip()]
        seen, uniq = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        if len(uniq) == 0:
            return pd.NA
        if len(uniq) == 1:
            return uniq[0]
        return ", ".join(uniq)

    # Ensure columns exist
    if 'store_id' not in df.columns:
        df['store_id'] = pd.NA
    if 'region' not in df.columns:
        df['region'] = pd.NA

    # ---- ONLINE detection (is_online True OR channel contains "Online") ----
    mask_flag = df['is_online'].apply(_truthy) if 'is_online' in df.columns else False
    if 'channel' in df.columns:
        chan_list = df['channel'].apply(_to_list)
        mask_chan = chan_list.apply(lambda lst: any(str(x).strip().lower() == "online" for x in lst))
    else:
        mask_chan = False
    mask_online_any = mask_flag | mask_chan

    # Set store_id / region as STRINGS (not lists)
    if getattr(mask_online_any, "any", lambda: False)():
        df.loc[mask_online_any, 'store_id'] = "Online"
        df.loc[mask_online_any, 'region']   = "Online"

    # Fill remaining missing region with the string "Online"
    df['region'] = df['region'].fillna("Online")

    # ---- Convert Hebrew cities in region to English ----
    hebrew_to_english = {
            'מרכז': 'Central',
            'ירושלים': 'Jerusalem',
            'דרום': 'South',
            'חיפה': 'Haifa'
            # Add more as needed
        }

    # Normalize store_id / region cells to STRINGS (["מרכז"] -> "מרכז")
    df['store_id'] = df['store_id'].apply(_listish_to_string)
    df['region']   = df['region'].apply(_listish_to_string)
    df['store_id']   = df['store_id'].apply(_listish_to_string)
    df['channel']   = df['channel'].apply(_listish_to_string)



    # ---- Build top_category for missing rows, using items_clean ----
    cat_col = 'top_category' if 'top_category' in items_clean.columns else (
        'category' if 'category' in items_clean.columns else None
    )
    item_key_col = 'item_id' if 'item_id' in items_clean.columns else (
        'sku' if 'sku' in items_clean.columns else None
    )

    if cat_col is not None and item_key_col is not None and 'items_id_list' in df.columns:
        map_df = items_clean[[item_key_col, cat_col]].dropna(subset=[item_key_col, cat_col]).copy()
        map_df[item_key_col] = map_df[item_key_col].astype(str)
        item_to_cat = dict(zip(map_df[item_key_col], map_df[cat_col].astype(str)))

        df['items_id_list'] = df['items_id_list'].apply(_to_list)

        if 'top_category' not in df.columns:
            df['top_category'] = pd.NA

        mask_topcat_missing = df['top_category'].isna()

        def _join_cats(id_list):
            cats = [item_to_cat.get(str(i)) for i in id_list]
            cats = [c for c in cats if pd.notna(c)]
            return ", ".join(cats)

        if mask_topcat_missing.any():
            df.loc[mask_topcat_missing, 'top_category'] = (
                df.loc[mask_topcat_missing, 'items_id_list'].apply(_join_cats)
            )

    # ---- De-dup comma-separated strings in top_region, top_location, top_category ----
    if 'top_region' in df.columns:
        df['top_region'] = df['top_region'].apply(_dedup_comma_string)
    if 'top_location' in df.columns:
        df['top_location'] = df['top_location'].apply(_dedup_comma_string)
    if 'top_category' in df.columns:
        df['top_category'] = df['top_category'].apply(_dedup_comma_string)

    # Round column avg_amount_per_item to 2 decimals if it exists
    if 'avg_amount_per_item' in df.columns: 
        df['avg_amount_per_item'] = df['avg_amount_per_item'].round(2)
    
    df['region'] = df['region'].replace(hebrew_to_english)

    return df


def engineer_features(order_df: pd.DataFrame) -> pd.DataFrame:
    """Add additional useful features to the orders dataframe.

    The following features are added:

    * ``year_month``: YYYY-MM string extracted from ``order_date``.
    * ``month``: numeric month (1–12).
    * ``quarter``: quarter of the year (1–4).
    * ``is_online``: boolean flag indicating whether the channel is 'Online'.
    * ``avg_amount_per_item``: ratio of ``line_amount`` to ``total_items_net``.

    Parameters
    ----------
    order_df : pd.DataFrame
        The cleaned orders dataframe.

    Returns
    -------
    pd.DataFrame
        Orders dataframe with new features.
    """
    df = order_df.copy()
    if 'order_date' in df.columns:
        df['year_month'] = df['order_date'].dt.to_period('M').astype(str)
        df['month'] = df['order_date'].dt.month
        df['quarter'] = df['order_date'].dt.quarter
    # Binary flag for online channel
    if 'channel' in df.columns:
        df['is_online'] = df['channel'].str.lower().eq('online')
    # Average amount per item
    if set(['line_amount', 'total_items_net']).issubset(df.columns):
        # Avoid division by zero; where total_items_net is zero or NA, result will be NaN
        df['avg_amount_per_item'] = df['line_amount'] / df['total_items_net']
    return df


def clean_customer(customer_df: pd.DataFrame) -> pd.DataFrame:
    """Clean the customer_spec table.

    For customer features representing counts or monetary amounts over different
    time windows (e.g., ``orders_in_3m``, ``spend_in_6m``), missing values
    likely mean the customer did not place an order in that window.  We replace
    NaNs with zeros for such columns.  You may adjust this behaviour if you
    believe the missingness has another cause.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Raw customer dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned customer dataframe.
    """
    df = customer_df.copy()
    # Identify numeric columns that reflect counts/amounts and likely missing means zero
    count_like = [
        col for col in df.columns
        if any(substr in col.lower() for substr in ['count', 'num', 'orders', 'spend', 'purchase'])
    ]
    for col in count_like:
        if df[col].dtype.kind in 'ifb':  # numeric types
            df[col] = df[col].fillna(0)
    return df


def drop_duplicates(df: pd.DataFrame, key: str | list[str], name: str) -> pd.DataFrame:
    """Remove duplicate rows based on a key (or keys) and report how many were dropped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to deduplicate.
    key : str or list of str
        Column name(s) to use for identifying duplicates.
    name : str
        Name of the DataFrame (for logging).

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    before = len(df)
    deduped = df.drop_duplicates(subset=key)
    after = len(deduped)
    dropped = before - after
    if dropped > 0:
        print(f"Dropped {dropped} duplicate rows from {name}\n")
    return deduped


def _factorize_key(df: pd.DataFrame, cols: list[str], name: str) -> pd.Series:
    """
    Create a stable Surrogate Key (integer) from a combination of columns.
    """
    key = df[cols].astype(str).agg('|'.join, axis=1)
    sk = pd.factorize(key)[0] + 1  # 1..K
    return pd.Series(sk, index=df.index, name=f"{name}_sk")



def main():
    # Define where the raw CSV files are located.  Modify this path to match your
    # environment.  In a Windows machine, use a raw string (r"...") or double
    # backslashes.
    # set path to your file
    raw_dir = cfg["paths"]["raw_dir"]  # e.g., "data/raw"

    item_spec, order_spec, customer_spec = load_data(raw_dir)


    # Report basic info
    print("Data shapes:")
    print(f"item_spec: {item_spec.shape}")
    print(f"order_spec: {order_spec.shape}")
    print(f"customer_spec: {customer_spec.shape}\n")

    # Display first few rows
    print("Sample rows from order_spec:")
    print(order_spec.head())
    print("\n")

    # Report missing values
    report_missing(item_spec, 'item_spec')
    report_missing(order_spec, 'order_spec')
    report_missing(customer_spec, 'customer_spec')

    # Clean order_spec: convert types, fill categorical, engineer features
    orders_clean = convert_types(order_spec)
    orders_clean = clean_categorical(orders_clean)
    orders_clean = engineer_features(orders_clean)
    orders_clean = drop_duplicates(orders_clean, key='order_id', name='order_spec')


    report_missing(orders_clean, 'order_spec')


    # Clean item_spec:
    # 1. Fill unknown store_id with 'Unknown' for non‑online rows
    # 2. Set store_id='Online' and region='אונליין' for online channel rows
    # 3. Fill missing categories using subcategory information
    items_clean = item_spec.copy()
    # Fill generic missing store_id with 'Unknown'
    if 'store_id' in items_clean.columns:
        items_clean['store_id'] = items_clean['store_id'].fillna('Unknown')
    # Apply online store/region rule
    if set(['channel', 'store_id', 'region']).issubset(items_clean.columns):
        items_clean = fill_online_store_region(items_clean)
    # Fill missing categories based on subcategory
    if set(['category', 'subcategory']).issubset(items_clean.columns):
        items_clean = fill_missing_category_sku_heuristic(items_clean)
    # Remove duplicates if item_id exists
    items_clean = drop_duplicates(
        items_clean,
        key=['item_id','order_id','order_date','customer_phone','sku'] if 'item_id' in items_clean.columns else None,
        name='item_spec',
    )

    # Apply Online/store/region/top_category enrichment to orders
    orders_clean = enrich_orders_with_online_and_topcat(orders_clean, items_clean)

    # Clean customer_spec
    customers_clean = clean_customer(customer_spec)
    customers_clean = drop_duplicates(customers_clean, key='customer_id' if 'customer_id' in customers_clean.columns else None, name='customer_spec')

    # Create output directory
    processed_xlsx = cfg["paths"]["processed_xlsx"]  # e.g., "data/processed/matrixdna_cleaned_data.xlsx"
    os.makedirs(os.path.dirname(processed_xlsx), exist_ok=True)
    print(f"Saving cleaned data to {processed_xlsx}\n")

    with pd.ExcelWriter(processed_xlsx, engine="openpyxl") as writer:
        orders_clean.to_excel(writer, sheet_name="Orders", index=False)
        items_clean.to_excel(writer, sheet_name="Items", index=False)
        customers_clean.to_excel(writer, sheet_name="Customers", index=False)


    print("Cleaning complete.  Cleaned datasets have been written to disk.\n")


if __name__ == '__main__':
    main()