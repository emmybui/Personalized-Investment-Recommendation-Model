"""Build the complete FAR-Trans dataset for the thesis.

Usage (from project root):
    python -m src.data.build_dataset

Pipeline flow (single authoritative pipeline):
    raw/  -->  clean.py  -->  processed/  -->  build_dataset.py  -->  splits/

Steps executed:
    1. Load all six cleaned tables via the central loader.
    2. Validate schema, keys, temporal keys, referential integrity,
       transactionType, units, totalValue, sort order.
    3. Build the primary Train/Validation/Test split with snapshots.
    4. Build five auxiliary rolling 6-month windows with snapshots.
    5. Compute holdings state at each cutoff.
    6. Run leakage checks on every split.
    7. Save a machine-readable data quality report.
"""

import argparse
import sys
from pathlib import Path

from .config import PROCESSED_DIR, SPLITS_DIR, PRIMARY_TRAIN_END, PRIMARY_VAL_END
from .loader import FARTransLoader, build_id_mapping
from .validator import (
    validate_dataset, validate_primary_split,
    validate_rolling_split, save_report,
)
from .splitter import (
    build_primary_split, build_rolling_splits,
    save_primary_split, save_rolling_splits,
)
from .state import build_holdings_asof


def main():
    parser = argparse.ArgumentParser(
        description="Build FAR-Trans splits for RATGR thesis."
    )
    parser.add_argument(
        "--processed-dir", default=str(PROCESSED_DIR),
        help="Directory containing the six cleaned CSV files.",
    )
    parser.add_argument(
        "--output-dir", default=str(SPLITS_DIR),
        help="Directory to write splits into.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = output_dir.parent / "reports"
    report_path = report_dir / "data_quality_report.json"

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print("[1/6] Loading cleaned data...")
    loader = FARTransLoader(args.processed_dir)
    data = loader.load_all()
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows, {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # 2. Validate
    # ------------------------------------------------------------------
    print("\n[2/6] Validating dataset...")
    report = validate_dataset(data)

    for w in report["warnings"]:
        print(f"  WARNING: {w}")

    if not report["ok"]:
        print("\n  DATASET VALIDATION FAILED:")
        for e in report["errors"]:
            print(f"  ERROR: {e}")
        save_report(report, report_path)
        print(f"\n  Report saved: {report_path}")
        print("  Fix the errors above before building splits.")
        sys.exit(1)

    print("  Validation passed.")

    # ------------------------------------------------------------------
    # 3. Primary split
    # ------------------------------------------------------------------
    print("\n[3/6] Building primary split (Train/Validation/Test)...")
    primary = build_primary_split(
        data["transactions"], data["customers"],
        data["assets"], data["close_prices"],
    )
    save_primary_split(primary, output_dir / "primary")
    print(f"  Train:      {len(primary.train):>8,} txn  "
          f"({primary.train['customerID'].nunique():,} customers, "
          f"{primary.train['ISIN'].nunique():,} assets)")
    print(f"  Validation: {len(primary.validation):>8,} txn  "
          f"({primary.validation['customerID'].nunique():,} customers, "
          f"{primary.validation['ISIN'].nunique():,} assets)")
    print(f"  Test:       {len(primary.test):>8,} txn  "
          f"({primary.test['customerID'].nunique():,} customers, "
          f"{primary.test['ISIN'].nunique():,} assets)")
    print(f"  Snapshots:  customer={len(primary.customer_snapshot_train):,}/"
          f"{len(primary.customer_snapshot_validation):,}  "
          f"asset={len(primary.asset_snapshot_train):,}/"
          f"{len(primary.asset_snapshot_validation):,}  "
          f"prices={len(primary.prices_upto_train_end):,}/"
          f"{len(primary.prices_upto_validation_end):,}")

    # ------------------------------------------------------------------
    # 4. Rolling splits
    # ------------------------------------------------------------------
    print("\n[4/6] Building 5 auxiliary rolling splits...")
    rolling = build_rolling_splits(
        data["transactions"], data["customers"],
        data["assets"], data["close_prices"],
    )
    save_rolling_splits(rolling, output_dir / "rolling")
    for i, item in enumerate(rolling, 1):
        print(f"  rolling {i} (cutoff={item['cutoff'].date()}): "
              f"train={len(item['train']):,}  test={len(item['test']):,}")

    # ------------------------------------------------------------------
    # 5. Holdings state
    # ------------------------------------------------------------------
    print("\n[5/6] Computing holdings state...")
    tx = data["transactions"]

    # Holdings at train_end (for validation candidate filtering)
    h_train = build_holdings_asof(tx, PRIMARY_TRAIN_END)
    h_train_file = output_dir / "primary" / "holdings_asof_train_end.csv"
    h_train.to_csv(h_train_file, index=False)
    n_held_train = h_train["currently_held"].sum()

    # Holdings at validation_end (for test candidate filtering)
    h_val = build_holdings_asof(tx, PRIMARY_VAL_END)
    h_val_file = output_dir / "primary" / "holdings_asof_validation_end.csv"
    h_val.to_csv(h_val_file, index=False)
    n_held_val = h_val["currently_held"].sum()

    print(f"  Holdings at train_end:      {n_held_train:,} held pairs  "
          f"(of {len(h_train):,} total)")
    print(f"  Holdings at validation_end: {n_held_val:,} held pairs  "
          f"(of {len(h_val):,} total)")

    # Holdings for each rolling cutoff
    for i, item in enumerate(rolling, 1):
        h_r = build_holdings_asof(tx, item["cutoff"])
        h_r_file = (output_dir / "rolling"
                     / f"rolling_{i:02d}_{item['cutoff'].date()}"
                     / "holdings_asof_cutoff.csv")
        h_r.to_csv(h_r_file, index=False)

    # ------------------------------------------------------------------
    # 5b. ID mapping
    # ------------------------------------------------------------------
    cust_map, asset_map = build_id_mapping(tx)
    print(f"  ID mapping:  {len(cust_map):,} customers, {len(asset_map):,} assets")

    import json
    mapping_path = output_dir / "id_mapping.json"
    mapping_path.write_text(json.dumps({
        "customer_to_idx": cust_map,
        "asset_to_idx": asset_map,
    }, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # 6. Leakage checks
    # ------------------------------------------------------------------
    print("\n[6/6] Running leakage checks...")
    leakage_ok = True

    primary_check = validate_primary_split(primary)
    if not primary_check["ok"]:
        leakage_ok = False
        for issue in primary_check["issues"]:
            print(f"  PRIMARY LEAKAGE: {issue}")
    else:
        print("  Primary split: no leakage detected.")

    report["primary_leakage"] = primary_check

    rolling_checks = []
    for i, item in enumerate(rolling, 1):
        rc = validate_rolling_split(item)
        rolling_checks.append(rc)
        if not rc["ok"]:
            leakage_ok = False
            for issue in rc["issues"]:
                print(f"  ROLLING {i} LEAKAGE: {issue}")

    if all(rc["ok"] for rc in rolling_checks):
        print("  Rolling splits: no leakage detected.")

    report["rolling_leakage"] = rolling_checks

    # ------------------------------------------------------------------
    # Save report (both to data/reports/ and root reports/ directory)
    # ------------------------------------------------------------------
    save_report(report, report_path)
    root_report_path = output_dir.parent.parent / "reports" / "data_quality_report.json"
    save_report(report, root_report_path)

    print(f"\nQuality report: {report_path}")
    print(f"Primary output: {output_dir / 'primary'}")
    print(f"Rolling output: {output_dir / 'rolling'}")
    print(f"ID mapping:     {mapping_path}")

    if leakage_ok:
        print("\n[OK] Dataset pipeline completed successfully. No leakage found.")
    else:
        print("\n[FAIL] Leakage detected! Review the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
