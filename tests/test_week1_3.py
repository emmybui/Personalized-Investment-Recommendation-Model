"""
Week 1-3 Verification Tests
============================
Run from project root:
    python tests/test_week1_3.py

Each test prints [PASS] or [FAIL] with details.
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def record(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {name}" + (f" -- {detail}" if detail else "")
    print(msg)
    results.append((name, ok, detail))


# =====================================================================
print("=" * 70)
print("TEST GROUP 1: Loader -- 6 cleaned files readable from one loader")
print("=" * 70)

try:
    from src.data.loader import FARTransLoader, REQUIRED_COLUMNS
    loader = FARTransLoader()
    data = loader.load_all()
    record("Loader imports", True)
except Exception as e:
    record("Loader imports", False, str(e))
    print("\n*** Cannot continue without loader. Fix the error above. ***")
    sys.exit(1)

EXPECTED_TABLES = ["customers", "assets", "markets", "close_prices",
                   "limit_prices", "transactions"]
for name in EXPECTED_TABLES:
    ok = name in data and len(data[name]) > 0
    record(f"Table '{name}' loaded",
           ok, f"{len(data[name]):,} rows" if ok else "MISSING or empty")

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 2: Schema validation")
print("=" * 70)

for name in EXPECTED_TABLES:
    df = data[name]
    missing = REQUIRED_COLUMNS[name] - set(df.columns)
    record(f"Schema '{name}'", not missing,
           f"missing: {sorted(missing)}" if missing else "all columns present")

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 3: Null primary keys & invalid values")
print("=" * 70)

checks = [
    ("customerID nulls", data["customers"]["customerID"].isna().sum()),
    ("ISIN nulls (assets)", data["assets"]["ISIN"].isna().sum()),
    ("marketID nulls", data["markets"]["marketID"].isna().sum()),
    ("transactionID nulls", data["transactions"]["transactionID"].isna().sum()),
    ("transactionID duplicates", data["transactions"]["transactionID"].duplicated().sum()),
]
for label, count in checks:
    record(label, count == 0, f"found {count}" if count > 0 else "none")

invalid_types = set(data["transactions"]["transactionType"].dropna().unique()) - {"Buy", "Sell"}
record("transactionType in {'Buy', 'Sell'}", len(invalid_types) == 0, f"invalid: {invalid_types}" if invalid_types else "OK")

units_bad = (data["transactions"]["units"].astype(float) <= 0).sum()
record("units > 0", units_bad == 0, f"{units_bad} invalid rows" if units_bad else "all positive")

val_bad = (data["transactions"]["totalValue"].astype(float) < 0).sum()
record("totalValue >= 0", val_bad == 0, f"{val_bad} invalid rows" if val_bad else "all non-negative")

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 4: Temporal key conflicts & Strict Whitelist Verification")
print("=" * 70)

def count_conflicts(df, keys):
    deduped = df.drop_duplicates()
    return deduped.duplicated(subset=keys, keep=False).sum()

c_conflicts = count_conflicts(data["customers"], ["customerID", "timestamp"])
record("Customer (customerID, timestamp) conflicts",
       c_conflicts == 0, f"{c_conflicts} conflicting rows" if c_conflicts else "none")

# Strict check against whitelisted ISINs
from src.data.snapshot import _KNOWN_ASSET_CONFLICT_ISINS
deduped_assets = data["assets"].drop_duplicates()
dup_mask = deduped_assets.duplicated(subset=["ISIN", "timestamp"], keep=False)
conflicting_isin_set = set(deduped_assets.loc[dup_mask, "ISIN"])
unknown_conflicts = conflicting_isin_set - set(_KNOWN_ASSET_CONFLICT_ISINS)

record("Asset conflict ISINs match whitelist exactly",
       len(unknown_conflicts) == 0,
       f"unknown conflicts: {unknown_conflicts}" if unknown_conflicts else f"matched {len(conflicting_isin_set)} whitelisted ISINs")

p_conflicts = count_conflicts(data["close_prices"], ["ISIN", "timestamp"])
record("Price (ISIN, timestamp) conflicts",
       p_conflicts == 0, f"{p_conflicts} conflicting rows" if p_conflicts else "none")

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 5: Referential integrity")
print("=" * 70)

cust_ids = set(data["customers"]["customerID"].dropna())
asset_ids = set(data["assets"]["ISIN"].dropna())
market_ids = set(data["markets"]["marketID"].dropna())
tx = data["transactions"]

bad_cust = (~tx["customerID"].isin(cust_ids)).sum()
bad_asset = (~tx["ISIN"].isin(asset_ids)).sum()
bad_mkt = (~tx["marketID"].isin(market_ids)).sum()

record("Txn -> customers ref", bad_cust == 0, f"{bad_cust} orphan txns" if bad_cust else "OK")
record("Txn -> assets ref", bad_asset == 0, f"{bad_asset} orphan txns" if bad_asset else "OK")
record("Txn -> markets ref", bad_mkt == 0, f"{bad_mkt} orphan txns" if bad_mkt else "OK")

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 6: Transaction deterministic sort")
print("=" * 70)

tx_sorted = tx.sort_values(["timestamp", "transactionID"], kind="stable")
order_ok = (tx.index == tx_sorted.index).all()
record("Transactions sorted by [timestamp, transactionID]", order_ok,
       "order matches" if order_ok else "ORDER MISMATCH")

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 7: Value sanity")
print("=" * 70)

neg_price = (data["close_prices"]["closePrice"] <= 0).sum()
record("closePrice > 0", neg_price == 0, f"{neg_price} invalid" if neg_price else "all positive")

bad_dates = (data["limit_prices"]["minDate"] > data["limit_prices"]["maxDate"]).sum()
record("limit_prices minDate <= maxDate", bad_dates == 0,
       f"{bad_dates} invalid" if bad_dates else "all valid")

ts_range = f'{tx["timestamp"].min().date()} -> {tx["timestamp"].max().date()}'
record("Transaction date range", True, ts_range)

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 8: Primary split exists and is correct")
print("=" * 70)

primary_dir = Path("data/splits/primary")
primary_files = [
    "train_transactions.csv", "validation_transactions.csv",
    "test_transactions.csv", "primary_split_summary.csv",
    "train_customer_snapshot.csv", "train_asset_snapshot.csv", "train_prices.csv",
    "validation_customer_snapshot.csv", "validation_asset_snapshot.csv", "validation_prices.csv",
    "holdings_asof_train_end.csv", "holdings_asof_validation_end.csv",
]
for f in primary_files:
    exists = (primary_dir / f).exists()
    size = (primary_dir / f).stat().st_size if exists else 0
    record(f"primary/{f}", exists and size > 0,
           f"{size:,} bytes" if exists else "FILE MISSING")

if (primary_dir / "train_transactions.csv").exists():
    train = pd.read_csv(primary_dir / "train_transactions.csv", parse_dates=["timestamp"])
    val = pd.read_csv(primary_dir / "validation_transactions.csv", parse_dates=["timestamp"])
    test = pd.read_csv(primary_dir / "test_transactions.csv", parse_dates=["timestamp"])

    train_end = pd.Timestamp("2021-12-31 23:59:59")
    val_start = pd.Timestamp("2022-01-01")
    val_end = pd.Timestamp("2022-06-30 23:59:59")
    test_start = pd.Timestamp("2022-07-01")
    test_end = pd.Timestamp("2022-11-29 23:59:59")

    record("Train max <= 2021-12-31",
           train["timestamp"].max() <= train_end,
           f"max={train['timestamp'].max()}")
    record("Validation range 2022-01-01..2022-06-30",
           val["timestamp"].min() >= val_start and val["timestamp"].max() <= val_end,
           f"{val['timestamp'].min().date()}..{val['timestamp'].max().date()}")
    record("Test range 2022-07-01..2022-11-29",
           test["timestamp"].min() >= test_start and test["timestamp"].max() <= test_end,
           f"{test['timestamp'].min().date()}..{test['timestamp'].max().date()}")

    record("No train/val overlap", train["timestamp"].max() < val["timestamp"].min())
    record("No val/test overlap", val["timestamp"].max() < test["timestamp"].min())

    print(f"\n  Primary sizes: train={len(train):,}  val={len(val):,}  test={len(test):,}")

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 9: Snapshot leakage & Provenance (snapshot_asof_date)")
print("=" * 70)

if (primary_dir / "train_customer_snapshot.csv").exists():
    snap_c = pd.read_csv(primary_dir / "train_customer_snapshot.csv",
                         parse_dates=["timestamp", "snapshot_asof_date"])
    snap_a = pd.read_csv(primary_dir / "train_asset_snapshot.csv",
                         parse_dates=["timestamp", "snapshot_asof_date"])
    prices_t = pd.read_csv(primary_dir / "train_prices.csv", parse_dates=["timestamp"])

    record("Customer snapshot <= train_end",
           snap_c["timestamp"].max() <= train_end,
           f"max={snap_c['timestamp'].max()}")
    record("Asset snapshot <= train_end",
           snap_a["timestamp"].max() <= train_end,
           f"max={snap_a['timestamp'].max()}")
    record("Prices <= train_end",
           prices_t["timestamp"].max() <= train_end,
           f"max={prices_t['timestamp'].max()}")
    record("One row per customerID",
           not snap_c["customerID"].duplicated().any(),
           f"{snap_c['customerID'].nunique()} unique customers")
    record("One row per ISIN",
           not snap_a["ISIN"].duplicated().any(),
           f"{snap_a['ISIN'].nunique()} unique assets")

    # snapshot_asof_date provenance verification
    c_asof_ok = (snap_c["snapshot_asof_date"].dt.date == pd.Timestamp("2021-12-31").date()).all()
    a_asof_ok = (snap_a["snapshot_asof_date"].dt.date == pd.Timestamp("2021-12-31").date()).all()
    record("Customer snapshot_asof_date equals cutoff", c_asof_ok)
    record("Asset snapshot_asof_date equals cutoff", a_asof_ok)

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 10: Rolling splits (5 windows with 6-month horizon limit)")
print("=" * 70)

rolling_dir = Path("data/splits/rolling")
CUTOFFS = ["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01", "2022-01-01"]

for i, cutoff in enumerate(CUTOFFS, 1):
    folder = rolling_dir / f"rolling_{i:02d}_{cutoff}"
    expected = ["train_transactions.csv", "test_transactions.csv",
                "customer_snapshot.csv", "asset_snapshot.csv", "prices_upto_t.csv",
                "holdings_asof_cutoff.csv"]
    all_exist = all((folder / f).exists() for f in expected)
    record(f"Rolling {i} ({cutoff}) files", all_exist,
           "all 6 files present" if all_exist else "MISSING files")

    if all_exist:
        r_train = pd.read_csv(folder / "train_transactions.csv", parse_dates=["timestamp"])
        r_test = pd.read_csv(folder / "test_transactions.csv", parse_dates=["timestamp"])
        c = pd.Timestamp(cutoff)
        horizon_end = c + pd.DateOffset(months=6)

        ok_train = r_train.empty or r_train["timestamp"].max() <= c
        ok_test_start = r_test.empty or r_test["timestamp"].min() > c
        ok_test_end = r_test.empty or r_test["timestamp"].max() <= horizon_end

        record(f"  Rolling {i} no leakage & <= 6M horizon",
               ok_train and ok_test_start and ok_test_end,
               f"train_max={r_train['timestamp'].max().date()}, test_min={r_test['timestamp'].min().date()}, test_max={r_test['timestamp'].max().date()}")

summary_ok = (rolling_dir / "rolling_split_summary.csv").exists()
record("rolling_split_summary.csv", summary_ok)

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 11: Holdings state & Point-in-Time Candidate Generator")
print("=" * 70)

try:
    from src.data.state import build_holdings_asof, currently_held_assets, get_candidate_assets

    holdings = build_holdings_asof(data["transactions"], "2021-12-31")
    n_held = holdings["currently_held"].sum()
    n_not = (~holdings["currently_held"]).sum()
    record("Holdings reconstruction works", len(holdings) > 0,
           f"{n_held:,} held pairs, {n_not:,} not-held pairs")

    nan_units = holdings["net_units"].isna().sum()
    record("No NaN in net_units", nan_units == 0,
           f"{nan_units} NaN values" if nan_units else "all valid")

    sample_cust = data["transactions"]["customerID"].value_counts().index[0]
    held_set = currently_held_assets(data["transactions"], sample_cust, "2021-12-31")
    record(f"currently_held_assets('{sample_cust}')", True,
           f"{len(held_set)} assets currently held")

    # Test point-in-time candidate selection
    pit_candidates = get_candidate_assets(data["transactions"], sample_cust, eligible_assets=None, as_of="2021-12-31")
    hist_assets_at_t = set(data["transactions"].loc[data["transactions"]["timestamp"] <= "2021-12-31", "ISIN"])
    record("get_candidate_assets() point-in-time candidate pool",
           len(pit_candidates) == len(hist_assets_at_t) - len(held_set),
           f"{len(pit_candidates)} PIT candidates from {len(hist_assets_at_t)} assets at cutoff")

except Exception as e:
    record("Holdings reconstruction", False, str(e))

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 12: Split Loader API & Shared ID Mapping")
print("=" * 70)

try:
    from src.data.loader import FARTransLoader, build_id_mapping

    p_split = loader.load_primary_split()
    p_ok = "train" in p_split and "validation" in p_split and "test" in p_split
    record("load_primary_split() API", p_ok, f"train={len(p_split['train']):,} rows")

    r_split = loader.load_rolling_split(1)
    r_ok = "train" in r_split and "test" in r_split and r_split["cutoff"] == "2020-01-01"
    record("load_rolling_split(1) API", r_ok, f"train={len(r_split['train']):,} rows")

    cust_map, asset_map = build_id_mapping(data["transactions"])
    id_map_ok = len(cust_map) > 0 and len(asset_map) > 0
    record("build_id_mapping()", id_map_ok, f"{len(cust_map):,} customers, {len(asset_map):,} assets mapped")

    id_map_file = Path("data/splits/id_mapping.json")
    record("id_mapping.json file exists", id_map_file.exists() and id_map_file.stat().st_size > 0)

except Exception as e:
    record("Split loader & ID mapping", False, str(e))

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 13: Data quality report")
print("=" * 70)

import json

report_path = Path("data/reports/data_quality_report.json")
if report_path.exists() and report_path.stat().st_size > 0:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    record("Report exists and readable", True, f"{report_path.stat().st_size} bytes")
    record("Report ok=true (0 errors)", report.get("ok", False),
           f"errors: {report.get('errors', [])}" if not report.get("ok") else "no errors")
    record("Primary leakage check in report",
           "primary_leakage" in report and report["primary_leakage"]["ok"])
    record("Rolling leakage checks in report",
           "rolling_leakage" in report
           and all(r["ok"] for r in report["rolling_leakage"]))
    n_warnings = len(report.get("warnings", []))
    record("Warnings documented", True, f"{n_warnings} warnings")
else:
    record("Report exists", False, "FILE MISSING or empty")

# =====================================================================
print("\n" + "=" * 70)
print("TEST GROUP 14: Config & Documentation matches Proposal")
print("=" * 70)

from src.data.config import (
    PRIMARY_TRAIN_END, PRIMARY_VAL_START, PRIMARY_VAL_END,
    PRIMARY_TEST_START, PRIMARY_TEST_END, ROLLING_CUTOFFS,
)

record("PRIMARY_TRAIN_END = 2021-12-31", "2021-12-31" in PRIMARY_TRAIN_END)
record("PRIMARY_VAL = 2022-01-01..2022-06-30",
       "2022-01-01" in PRIMARY_VAL_START and "2022-06-30" in PRIMARY_VAL_END)
record("PRIMARY_TEST = 2022-07-01..2022-11-29",
       "2022-07-01" in PRIMARY_TEST_START and "2022-11-29" in PRIMARY_TEST_END)
record("5 rolling cutoffs unchanged",
       ROLLING_CUTOFFS == ["2020-01-01", "2020-07-01", "2021-01-01",
                           "2021-07-01", "2022-01-01"])

doc_protocol = Path("docs/data_protocol.md")
record("docs/data_protocol.md exists", doc_protocol.exists() and doc_protocol.stat().st_size > 0)

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)

print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")

if failed > 0:
    print(f"\n  FAILED TESTS:")
    for name, ok, detail in results:
        if not ok:
            print(f"    [FAIL] {name} -- {detail}")

print()
if failed == 0:
    print("  >>> ALL TESTS PASSED -- Week 1-3 deliverables are 100% complete. <<<")
else:
    print(f"  >>> {failed} TEST(S) FAILED -- review the issues above. <<<")

sys.exit(0 if failed == 0 else 1)
