"""Temporal splitting: primary protocol + auxiliary rolling windows.

The primary split follows the thesis Proposal:
    Train:      up to 2021-12-31
    Validation: 2022-01-01 .. 2022-06-30
    Test:       2022-07-01 .. 2022-11-29

The five rolling windows provide auxiliary robustness evaluations and
are NOT replacements for the primary Train/Validation/Test.

Candidate policy (thesis mode):
    NO pair-overlap removal between train/test.
    Instead, candidate exclusion is done via holdings state:
      candidate_assets = all_assets - currently_held_assets(customer, cutoff)
    This is implemented in state.py and applied during evaluation, not here.
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .config import (
    PRIMARY_TRAIN_END, PRIMARY_VAL_START, PRIMARY_VAL_END,
    PRIMARY_TEST_START, PRIMARY_TEST_END,
    ROLLING_CUTOFFS, ROLLING_HORIZON_MONTHS,
)
from .snapshot import (
    get_customer_snapshot_asof,
    get_asset_snapshot_asof,
    get_prices_asof,
)


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class PrimarySplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp
    # Snapshots at train_end  (for model training + validation prediction)
    customer_snapshot_train: pd.DataFrame
    asset_snapshot_train: pd.DataFrame
    prices_upto_train_end: pd.DataFrame
    # Snapshots at validation_end  (for final test prediction)
    customer_snapshot_validation: pd.DataFrame
    asset_snapshot_validation: pd.DataFrame
    prices_upto_validation_end: pd.DataFrame


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _prepare_tx(tx):
    tx = tx.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    return tx.sort_values(
        ["timestamp", "transactionID"], kind="stable"
    ).reset_index(drop=True)


# -----------------------------------------------------------------------
# Primary split
# -----------------------------------------------------------------------

def build_primary_split(transactions, customers, assets, close_prices):
    """Proposal-aligned temporal protocol with point-in-time snapshots.

    Snapshots are generated at two cutoffs:
      1. train_end  — features available when training the model and when
         generating validation predictions.
      2. val_end    — features available when generating test predictions.
    """
    tx = _prepare_tx(transactions)
    train_end = pd.Timestamp(PRIMARY_TRAIN_END)
    val_start = pd.Timestamp(PRIMARY_VAL_START)
    val_end = pd.Timestamp(PRIMARY_VAL_END)
    test_start = pd.Timestamp(PRIMARY_TEST_START)
    test_end = pd.Timestamp(PRIMARY_TEST_END)

    train = tx.loc[tx["timestamp"] <= train_end].copy()
    val = tx.loc[
        (tx["timestamp"] >= val_start) & (tx["timestamp"] <= val_end)
    ].copy()
    test = tx.loc[
        (tx["timestamp"] >= test_start) & (tx["timestamp"] <= test_end)
    ].copy()

    # Hard temporal invariants
    assert train.empty or train["timestamp"].max() <= train_end
    assert val.empty or (
        val["timestamp"].min() >= val_start
        and val["timestamp"].max() <= val_end
    )
    assert test.empty or (
        test["timestamp"].min() >= test_start
        and test["timestamp"].max() <= test_end
    )

    # Point-in-time snapshots — no future information
    cust_snap_train = get_customer_snapshot_asof(customers, train_end)
    asset_snap_train = get_asset_snapshot_asof(assets, train_end)
    prices_train = get_prices_asof(close_prices, train_end)

    cust_snap_val = get_customer_snapshot_asof(customers, val_end)
    asset_snap_val = get_asset_snapshot_asof(assets, val_end)
    prices_val = get_prices_asof(close_prices, val_end)

    return PrimarySplit(
        train=train,
        validation=val,
        test=test,
        train_end=train_end,
        validation_end=val_end,
        test_end=test_end,
        customer_snapshot_train=cust_snap_train,
        asset_snapshot_train=asset_snap_train,
        prices_upto_train_end=prices_train,
        customer_snapshot_validation=cust_snap_val,
        asset_snapshot_validation=asset_snap_val,
        prices_upto_validation_end=prices_val,
    )


# -----------------------------------------------------------------------
# Rolling splits
# -----------------------------------------------------------------------

def build_rolling_splits(transactions, customers, assets, close_prices):
    """Five auxiliary 6-month windows for robustness evaluation.

    Each window generates:
      - train/test transactions (split at cutoff)
      - customer/asset snapshots as-of cutoff
      - price history up to cutoff
    """
    tx = _prepare_tx(transactions)
    result = []

    for cutoff_text in ROLLING_CUTOFFS:
        cutoff = pd.Timestamp(cutoff_text)
        end = cutoff + pd.DateOffset(months=ROLLING_HORIZON_MONTHS)

        train = tx.loc[tx["timestamp"] <= cutoff].copy()
        test = tx.loc[
            (tx["timestamp"] > cutoff) & (tx["timestamp"] <= end)
        ].copy()

        cust_snap = get_customer_snapshot_asof(customers, cutoff)
        asset_snap = get_asset_snapshot_asof(assets, cutoff)
        prices = get_prices_asof(close_prices, cutoff)

        result.append({
            "cutoff": cutoff,
            "test_end": end,
            "train": train,
            "test": test,
            "customer_snapshot": cust_snap,
            "asset_snapshot": asset_snap,
            "prices_upto_t": prices,
        })

    return result


# -----------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------

def _split_stats(df, label):
    if df.empty:
        return {"split": label, "start": None, "end": None,
                "rows": 0, "customers": 0, "assets": 0}
    return {
        "split": label,
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "rows": len(df),
        "customers": int(df["customerID"].nunique()),
        "assets": int(df["ISIN"].nunique()),
    }


def save_primary_split(split: PrimarySplit, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transactions
    split.train.to_csv(output_dir / "train_transactions.csv", index=False)
    split.validation.to_csv(output_dir / "validation_transactions.csv", index=False)
    split.test.to_csv(output_dir / "test_transactions.csv", index=False)

    # Snapshots at train_end
    split.customer_snapshot_train.to_csv(
        output_dir / "train_customer_snapshot.csv", index=False)
    split.asset_snapshot_train.to_csv(
        output_dir / "train_asset_snapshot.csv", index=False)
    split.prices_upto_train_end.to_csv(
        output_dir / "train_prices.csv", index=False)

    # Snapshots at validation_end
    split.customer_snapshot_validation.to_csv(
        output_dir / "validation_customer_snapshot.csv", index=False)
    split.asset_snapshot_validation.to_csv(
        output_dir / "validation_asset_snapshot.csv", index=False)
    split.prices_upto_validation_end.to_csv(
        output_dir / "validation_prices.csv", index=False)

    # Summary
    summary = pd.DataFrame([
        _split_stats(split.train, "train"),
        _split_stats(split.validation, "validation"),
        _split_stats(split.test, "test"),
    ])
    summary.to_csv(output_dir / "primary_split_summary.csv", index=False)


def save_rolling_splits(rolling_splits, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, item in enumerate(rolling_splits, start=1):
        folder = output_dir / f"rolling_{i:02d}_{item['cutoff'].date()}"
        folder.mkdir(parents=True, exist_ok=True)

        item["train"].to_csv(folder / "train_transactions.csv", index=False)
        item["test"].to_csv(folder / "test_transactions.csv", index=False)
        item["customer_snapshot"].to_csv(
            folder / "customer_snapshot.csv", index=False)
        item["asset_snapshot"].to_csv(
            folder / "asset_snapshot.csv", index=False)
        item["prices_upto_t"].to_csv(
            folder / "prices_upto_t.csv", index=False)

        rows.append({
            "window": i,
            "cutoff": str(item["cutoff"]),
            "test_end": str(item["test_end"]),
            "train_rows": len(item["train"]),
            "test_rows": len(item["test"]),
            "train_customers": int(item["train"]["customerID"].nunique()),
            "test_customers": int(item["test"]["customerID"].nunique()) if not item["test"].empty else 0,
            "train_assets": int(item["train"]["ISIN"].nunique()),
            "test_assets": int(item["test"]["ISIN"].nunique()) if not item["test"].empty else 0,
        })

    pd.DataFrame(rows).to_csv(
        output_dir / "rolling_split_summary.csv", index=False
    )
