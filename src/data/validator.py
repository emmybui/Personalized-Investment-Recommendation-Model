"""Dataset validation and split leakage checks.

Data quality policy (unified with snapshot.py):
- Exact duplicate:            remove (not an error).
- Conflicting temporal key:   ERROR (pipeline must stop).
- Invalid transactionType:    ERROR.
- Invalid units / totalValue: ERROR.
- Unknown marketID in txn:    ERROR (needed for graph).
- Sort order:                 checked on the stable event order.

validate_dataset()  -- structural / quality checks on the six cleaned tables.
validate_primary_split()  -- temporal leakage checks on primary split.
validate_rolling_split()  -- temporal leakage checks on a rolling split.
"""

from pathlib import Path
import json
import pandas as pd

from .config import EVENT_ORDER, PRIMARY_TEST_END, TRANSACTION_KEY


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _temporal_key_conflicts(df, keys):
    """Return rows that share the same temporal key but differ in content."""
    deduped = df.drop_duplicates()
    duplicated = deduped.duplicated(subset=keys, keep=False)
    return deduped.loc[duplicated].sort_values(keys)


# Import from loader inline to avoid circular imports at module level
def _get_required_columns():
    from .loader import REQUIRED_COLUMNS
    return REQUIRED_COLUMNS


# -----------------------------------------------------------------------
# Dataset-level validation (six cleaned tables)
# -----------------------------------------------------------------------

def validate_dataset(data):
    """Return a machine-readable quality report.

    Policy: all structural issues go into errors[]; informational items
    go into warnings[].  The pipeline MUST stop if errors is non-empty.
    """
    errors = []
    warnings = []
    stats = {}

    REQUIRED_COLUMNS = _get_required_columns()

    customers = data["customers"]
    assets = data["assets"]
    markets = data["markets"]
    prices = data["close_prices"]
    limits = data["limit_prices"]
    tx = data["transactions"]

    # --- Basic stats per table ---
    for name, df in data.items():
        stats[name] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

        missing = REQUIRED_COLUMNS[name] - set(df.columns)
        if missing:
            errors.append(f"{name}: missing columns {sorted(missing)}")

    # --- Null primary keys ---
    if customers["customerID"].isna().any():
        errors.append("customers: null customerID exists")
    if assets["ISIN"].isna().any():
        errors.append("assets: null ISIN exists")
    if markets["marketID"].isna().any():
        errors.append("markets: null marketID exists")
    if tx["transactionID"].isna().any():
        errors.append("transactions: null transactionID exists")

    if tx.duplicated(TRANSACTION_KEY).any():
        errors.append(
            "transactions: composite (customerID, transactionID) key is not unique"
        )
    reused_ids = tx.loc[
        tx["transactionID"].duplicated(keep=False), "transactionID"
    ].nunique()
    stats["transactions"]["globally_reused_transaction_ids"] = int(reused_ids)

    # --- Temporal key conflicts (unified ERROR policy) ---
    c_conflict = _temporal_key_conflicts(
        customers, ["customerID", "timestamp"])
    if not c_conflict.empty:
        errors.append(
            f"customers: {len(c_conflict)} rows have duplicated "
            "(customerID, timestamp) keys with different content"
        )

    a_conflict = _temporal_key_conflicts(assets, ["ISIN", "timestamp"])
    if not a_conflict.empty:
        # Report but allow known FAR-Trans conflicts (3 dual-classified ETFs)
        from .snapshot import _KNOWN_ASSET_CONFLICT_ISINS
        unknown = a_conflict.loc[
            ~a_conflict["ISIN"].isin(_KNOWN_ASSET_CONFLICT_ISINS)]
        if not unknown.empty:
            errors.append(
                f"assets: {len(unknown)} rows have UNKNOWN duplicated "
                "(ISIN, timestamp) keys with different content"
            )
        else:
            warnings.append(
                f"assets: {len(a_conflict)} rows belong to {len(a_conflict[['ISIN','timestamp']].drop_duplicates())} "
                "known whitelisted (ISIN, timestamp) conflicts; resolved by keeping last"
            )

    p_conflict = _temporal_key_conflicts(prices, ["ISIN", "timestamp"])
    if not p_conflict.empty:
        errors.append(
            f"close_prices: {len(p_conflict)} rows have duplicated "
            "(ISIN, timestamp) keys with different content"
        )

    # --- Market conflicts ---
    m_conflict = _temporal_key_conflicts(markets, ["marketID"])
    if not m_conflict.empty:
        errors.append(
            f"markets: {len(m_conflict)} rows share marketID but differ "
            "in other columns"
        )

    # --- Referential integrity (all ERROR for graph) ---
    customer_ids = set(customers["customerID"].dropna())
    asset_ids = set(assets["ISIN"].dropna())
    market_ids = set(markets["marketID"].dropna())

    bad_customer = tx.loc[~tx["customerID"].isin(customer_ids)]
    bad_asset = tx.loc[~tx["ISIN"].isin(asset_ids)]
    bad_market_tx = tx.loc[~tx["marketID"].isin(market_ids)]
    bad_asset_price = prices.loc[~prices["ISIN"].isin(asset_ids)]
    bad_asset_limit = limits.loc[~limits["ISIN"].isin(asset_ids)]

    if not bad_customer.empty:
        errors.append(
            f"transactions: {len(bad_customer)} rows with unknown customerID")
    if not bad_asset.empty:
        errors.append(
            f"transactions: {len(bad_asset)} rows with unknown ISIN")
    if not bad_market_tx.empty:
        # Upgraded to ERROR -- marketID is needed for graph construction
        errors.append(
            f"transactions: {len(bad_market_tx)} rows with unknown marketID")
    if not bad_asset_price.empty:
        warnings.append(
            f"close_prices: {len(bad_asset_price)} rows with unknown ISIN")
    if not bad_asset_limit.empty:
        warnings.append(
            f"limit_prices: {len(bad_asset_limit)} rows with unknown ISIN")

    # --- transactionType validation (new) ---
    valid_types = {"Buy", "Sell"}
    actual_types = set(tx["transactionType"].dropna().unique())
    invalid_types = actual_types - valid_types
    if invalid_types:
        errors.append(
            f"transactions: invalid transactionType values: {sorted(invalid_types)}")

    # --- units > 0 validation (new) ---
    if tx["units"].isna().any():
        errors.append(
            f"transactions: {tx['units'].isna().sum()} null units values")
    elif (tx["units"].astype(float) <= 0).any():
        n_bad = (tx["units"].astype(float) <= 0).sum()
        errors.append(
            f"transactions: {n_bad} rows with units <= 0")

    # --- totalValue >= 0 validation (new) ---
    if tx["totalValue"].isna().any():
        errors.append(
            f"transactions: {tx['totalValue'].isna().sum()} null totalValue values")
    elif (tx["totalValue"].astype(float) < 0).any():
        n_bad = (tx["totalValue"].astype(float) < 0).sum()
        errors.append(
            f"transactions: {n_bad} rows with totalValue < 0")

    # --- Value / date sanity ---
    if (prices["closePrice"] <= 0).any():
        errors.append("close_prices: non-positive closePrice exists")

    if (limits["minDate"] > limits["maxDate"]).any():
        errors.append("limit_prices: minDate > maxDate exists")

    for name, df, col in [
        ("transactions", tx, "timestamp"),
        ("customers", customers, "timestamp"),
        ("assets", assets, "timestamp"),
        ("close_prices", prices, "timestamp"),
    ]:
        if df[col].isna().any():
            errors.append(f"{name}: null {col} exists")

    # --- Transaction sort order: stable and globally deterministic ---
    if not tx.empty and len(tx) > 1:
        expected = tx.sort_values(EVENT_ORDER, kind="stable").reset_index(drop=True)
        if not tx[EVENT_ORDER].reset_index(drop=True).equals(expected[EVENT_ORDER]):
            errors.append(f"transactions: not sorted by {EVENT_ORDER}")

    # --- Date range coverage ---
    if not tx.empty:
        stats["transactions"]["min_timestamp"] = str(tx["timestamp"].min())
        stats["transactions"]["max_timestamp"] = str(tx["timestamp"].max())

    expected_end = pd.Timestamp(PRIMARY_TEST_END)
    if (
        not tx.empty
        and tx["timestamp"].max().normalize() < expected_end.normalize()
    ):
        warnings.append(
            f"transactions: max timestamp {tx['timestamp'].max()} is earlier "
            f"than proposal dataset end {expected_end}"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


# -----------------------------------------------------------------------
# Split-level leakage validation
# -----------------------------------------------------------------------

def validate_primary_split(split):
    """Check the primary split for temporal leakage."""
    issues = []

    train_end = split.train_end
    val_end = split.validation_end
    test_end = split.test_end

    # Transaction temporal boundaries
    if not split.train.empty:
        max_train_ts = split.train["timestamp"].max()
        if max_train_ts > train_end:
            issues.append(
                f"LEAKAGE: train contains timestamps after train_end "
                f"({max_train_ts} > {train_end})")

    if not split.validation.empty:
        min_val_ts = split.validation["timestamp"].min()
        max_val_ts = split.validation["timestamp"].max()
        if min_val_ts <= train_end:
            issues.append(
                f"LEAKAGE: validation contains timestamps in train range "
                f"({min_val_ts} <= {train_end})")
        if max_val_ts > val_end:
            issues.append(
                f"LEAKAGE: validation contains timestamps after val_end "
                f"({max_val_ts} > {val_end})")

    if not split.test.empty:
        min_test_ts = split.test["timestamp"].min()
        max_test_ts = split.test["timestamp"].max()
        if min_test_ts <= val_end:
            issues.append(
                f"LEAKAGE: test contains timestamps in val range "
                f"({min_test_ts} <= {val_end})")
        if max_test_ts > test_end:
            issues.append(
                f"LEAKAGE: test contains timestamps after test_end "
                f"({max_test_ts} > {test_end})")

    # Snapshot leakage
    for attr, cutoff, label in [
        ("customer_snapshot_train", train_end, "customer_snapshot@train_end"),
        ("asset_snapshot_train", train_end, "asset_snapshot@train_end"),
        ("prices_upto_train_end", train_end, "prices@train_end"),
        ("customer_snapshot_validation", val_end, "customer_snapshot@val_end"),
        ("asset_snapshot_validation", val_end, "asset_snapshot@val_end"),
        ("prices_upto_validation_end", val_end, "prices@val_end"),
    ]:
        snap = getattr(split, attr, None)
        if snap is not None and not snap.empty and "timestamp" in snap.columns:
            max_snap = snap["timestamp"].max()
            if max_snap > cutoff:
                issues.append(
                    f"LEAKAGE: {label} has timestamp {max_snap} > {cutoff}")
        if snap is not None and not snap.empty and "snapshot_asof_date" in snap.columns:
            provenance = pd.to_datetime(snap["snapshot_asof_date"])
            if not (provenance == cutoff).all():
                issues.append(
                    f"LEAKAGE/PROVENANCE: {label} snapshot_asof_date "
                    f"does not equal {cutoff}"
                )
        if snap is not None and "customer_snapshot" in label:
            if snap["customerID"].duplicated().any():
                issues.append(f"LEAKAGE/SCHEMA: {label} has duplicate customerID")
        if snap is not None and "asset_snapshot" in label:
            if snap["ISIN"].duplicated().any():
                issues.append(f"LEAKAGE/SCHEMA: {label} has duplicate ISIN")

    return {"ok": not issues, "issues": issues}


def validate_rolling_split(item):
    """Check a single rolling split for temporal leakage."""
    issues = []
    cutoff = item["cutoff"]
    test_end = item["test_end"]

    if not item["train"].empty:
        max_t = item["train"]["timestamp"].max()
        if max_t > cutoff:
            issues.append(
                f"LEAKAGE: rolling train timestamps exceed cutoff "
                f"({max_t} > {cutoff})")

    if not item["test"].empty:
        min_t = item["test"]["timestamp"].min()
        max_t = item["test"]["timestamp"].max()
        if min_t <= cutoff:
            issues.append(
                f"LEAKAGE: rolling test overlaps with train range "
                f"({min_t} <= {cutoff})")
        if max_t > test_end:
            issues.append(
                f"LEAKAGE: rolling test exceeds test_end "
                f"({max_t} > {test_end})")

    for key, label in [
        ("customer_snapshot", "customer_snapshot"),
        ("asset_snapshot", "asset_snapshot"),
        ("prices_upto_t", "prices"),
    ]:
        if key in item and not item[key].empty and "timestamp" in item[key].columns:
            max_snap = item[key]["timestamp"].max()
            if max_snap > cutoff:
                issues.append(
                    f"LEAKAGE: rolling {label} has timestamp {max_snap} > cutoff {cutoff}")

    return {"ok": not issues, "issues": issues}


# -----------------------------------------------------------------------
# Report I/O
# -----------------------------------------------------------------------

def save_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
