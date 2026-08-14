"""Point-in-time snapshot generation.

Each snapshot captures entity attributes as known at a given cutoff date.
This prevents data leakage from future attribute updates into model features.

Data quality policy (unified with validator.py):
- Exact duplicates (all columns identical):  removed automatically.
- Conflicting duplicates (same temporal key, different content):  ERROR.
  The pipeline must stop; the conflict must be resolved in clean.py first.
  Exception: asset_information has 3 known dual-classified ETFs in FAR-Trans.
  These are whitelisted via allow_known_asset_conflicts=True.

Fixes applied vs. the original clean.py snapshot logic:
- RED 02: Asset eligibility uses only information <= as_of_date (no test_end).
- RED 05: Conflicting same-timestamp records are detected and reported.
"""

import pandas as pd

from .clean import KNOWN_ASSET_CONFLICT_ISINS

# 3 known FAR-Trans ISINs that are dual-classified (Stock / MTF) at the
# same timestamp.  These are NOT real conflicts; they are classification
# artefacts in the source dataset.
_KNOWN_ASSET_CONFLICT_ISINS = KNOWN_ASSET_CONFLICT_ISINS


def _check_conflicts(df, key_cols, entity_name, known_whitelist=None):
    """Remove exact duplicates; raise ValueError on true conflicts.

    If known_whitelist is provided, conflicting keys whose ID value (first
    element of key_cols) is in the whitelist are resolved by keeping the
    last row instead of raising.
    """
    df = df.drop_duplicates()

    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if not dup_mask.any():
        return df

    conflicts = df.loc[dup_mask].copy()
    id_col = key_cols[0]

    if known_whitelist:
        is_known = conflicts[id_col].isin(known_whitelist)
        unknown_conflicts = conflicts.loc[~is_known]
        known_conflicts = conflicts.loc[is_known]

        # Resolve known conflicts deterministically
        if not known_conflicts.empty:
            import warnings
            n = known_conflicts[key_cols].drop_duplicates().shape[0]
            warnings.warn(
                f"[{entity_name}] {n} known whitelisted conflict(s) "
                f"resolved by keeping last row."
            )

        if unknown_conflicts.empty:
            df = df.drop_duplicates(subset=key_cols, keep="last")
            return df

        # Unknown conflicts exist -> raise
        conflicts = unknown_conflicts

    sample = (
        conflicts[key_cols]
        .drop_duplicates()
        .head(5)
        .to_dict("records")
    )
    raise ValueError(
        f"{entity_name}: {len(conflicts)} rows share identical "
        f"{key_cols} but differ in other columns. "
        f"Fix these in clean.py before running the pipeline. "
        f"Conflicting keys: {sample}"
    )


def get_customer_snapshot_asof(customer_df, as_of_date):
    """One row per customerID using the most recent update <= as_of_date.

    Raises ValueError if any (customerID, timestamp) pair has conflicting
    rows (exact duplicates are removed automatically).
    """
    as_of_date = pd.Timestamp(as_of_date)
    valid = customer_df.loc[customer_df["timestamp"] <= as_of_date].copy()
    if valid.empty:
        return valid

    valid = _check_conflicts(
        valid, ["customerID", "timestamp"], "customer_snapshot"
    )
    snapshot = (
        valid.sort_values(["customerID", "timestamp"])
        .drop_duplicates(subset=["customerID"], keep="last")
        .reset_index(drop=True)
    )
    snapshot["snapshot_asof_date"] = as_of_date
    return snapshot


def get_asset_snapshot_asof(asset_df, as_of_date):
    """One row per ISIN using the most recent update <= as_of_date.

    RED 02 fix: asset eligibility is determined solely from asset_information
    records available at as_of_date.  No future price data is consulted.

    3 known FAR-Trans ISINs with dual classification are whitelisted and
    resolved by keeping the last row.  All other conflicts raise ValueError.
    """
    as_of_date = pd.Timestamp(as_of_date)
    valid = asset_df.loc[asset_df["timestamp"] <= as_of_date].copy()
    if valid.empty:
        return valid

    valid = _check_conflicts(
        valid, ["ISIN", "timestamp"], "asset_snapshot",
        known_whitelist=_KNOWN_ASSET_CONFLICT_ISINS,
    )
    snapshot = (
        valid.sort_values(["ISIN", "timestamp"])
        .drop_duplicates(subset=["ISIN"], keep="last")
        .reset_index(drop=True)
    )
    snapshot["snapshot_asof_date"] = as_of_date
    return snapshot


def get_prices_asof(close_prices_df, as_of_date):
    """Return close prices up to and including *as_of_date* only."""
    as_of_date = pd.Timestamp(as_of_date)
    return (
        close_prices_df.loc[close_prices_df["timestamp"] <= as_of_date]
        .copy()
        .sort_values(["ISIN", "timestamp"])
        .reset_index(drop=True)
    )
