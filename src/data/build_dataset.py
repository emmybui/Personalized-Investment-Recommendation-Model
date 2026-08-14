"""Run the reproducible FAR-Trans Checkpoint 1 pipeline.

One command cleans the six core files, validates them, creates point-in-time
splits/snapshots, reconstructs holdings, builds temporal graph events, and
exports baseline-ready indexed interactions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .clean import run_cleaning
from .config import (
    PRIMARY_TRAIN_END,
    PRIMARY_VAL_END,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
)
from .graph import build_interaction_snapshot, build_temporal_graph_events
from .loader import FARTransLoader, encode_transactions, fit_id_mapping
from .splitter import (
    build_primary_split,
    build_rolling_splits,
    save_primary_split,
    save_rolling_splits,
)
from .state import build_holdings_asof
from .validator import (
    save_report,
    validate_dataset,
    validate_primary_split,
    validate_rolling_split,
)


def _cold_start_stats(frame: pd.DataFrame, mapping) -> dict:
    unknown_customer = ~frame["customerID"].astype(str).isin(mapping.customer_to_idx)
    unknown_asset = ~frame["ISIN"].astype(str).isin(mapping.asset_to_idx)
    return {
        "rows": int(len(frame)),
        "unknown_customer_rows": int(unknown_customer.sum()),
        "unknown_asset_rows": int(unknown_asset.sum()),
        "unknown_any_rows": int((unknown_customer | unknown_asset).sum()),
        "unknown_customers": int(frame.loc[unknown_customer, "customerID"].nunique()),
        "unknown_assets": int(frame.loc[unknown_asset, "ISIN"].nunique()),
    }


def _split_stats(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "customers": int(frame["customerID"].nunique()),
        "assets": int(frame["ISIN"].nunique()),
        "start": str(frame["timestamp"].min()) if not frame.empty else None,
        "end": str(frame["timestamp"].max()) if not frame.empty else None,
    }


def _save_baseline_interactions(train: pd.DataFrame, mapping, path: Path) -> int:
    buys = train.loc[train["transactionType"].eq("Buy")].copy()
    encoded = encode_transactions(buys, mapping)
    interactions = (
        encoded.groupby(["customer_idx", "asset_idx"], as_index=False)
        .agg(
            weight=("transactionID", "size"),
            total_value=("totalValue", "sum"),
            last_timestamp=("timestamp", "max"),
        )
        .sort_values(["customer_idx", "asset_idx"], kind="stable")
    )
    interactions.to_csv(path, index=False)
    return len(interactions)


def build_checkpoint(
    *,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = SPLITS_DIR,
    reports_dir: Path = REPORTS_DIR,
    skip_clean: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cleaning_report = None
    if not skip_clean:
        print("[1/7] Cleaning six FAR-Trans files...")
        cleaning_report = run_cleaning(
            raw_dir,
            processed_dir,
            reports_dir / "cleaning_report.json",
        )
    else:
        print("[1/7] Cleaning skipped by request.")

    print("[2/7] Loading and validating cleaned tables...")
    data = FARTransLoader(processed_dir=processed_dir, splits_dir=output_dir).load_all()
    report = validate_dataset(data)
    if not report["ok"]:
        save_report(report, reports_dir / "data_quality_report.json")
        raise ValueError("Dataset validation failed: " + "; ".join(report["errors"]))
    save_report(report, reports_dir / "data_quality_report.json")

    print("[3/7] Building primary point-in-time split...")
    primary = build_primary_split(
        data["transactions"],
        data["customers"],
        data["assets"],
        data["close_prices"],
    )
    primary_dir = output_dir / "primary"
    save_primary_split(primary, primary_dir)

    print("[4/7] Building auxiliary rolling snapshots...")
    rolling = build_rolling_splits(
        data["transactions"],
        data["customers"],
        data["assets"],
        data["close_prices"],
    )
    save_rolling_splits(rolling, output_dir / "rolling")

    print("[5/7] Reconstructing point-in-time holdings...")
    holdings_train = build_holdings_asof(data["transactions"], PRIMARY_TRAIN_END)
    holdings_validation = build_holdings_asof(
        data["transactions"], PRIMARY_VAL_END
    )
    holdings_train.to_csv(primary_dir / "holdings_asof_train_end.csv", index=False)
    holdings_validation.to_csv(
        primary_dir / "holdings_asof_validation_end.csv", index=False
    )
    for index, item in enumerate(rolling, 1):
        folder = (
            output_dir
            / "rolling"
            / f"rolling_{index:02d}_{item['cutoff'].date()}"
        )
        build_holdings_asof(data["transactions"], item["cutoff"]).to_csv(
            folder / "holdings_asof_cutoff.csv", index=False
        )

    print("[6/7] Building train-fitted graph and baseline artifacts...")
    mapping = fit_id_mapping(primary.train, fitted_on="primary_train")
    mapping.save(output_dir / "id_mapping.json")
    graph = build_temporal_graph_events(
        primary.train,
        as_of=primary.train_end,
        customer_to_idx=mapping.customer_to_idx,
        asset_to_idx=mapping.asset_to_idx,
    )
    graph.frame.to_csv(primary_dir / "train_graph_events.csv", index=False)
    graph_snapshot = build_interaction_snapshot(
        primary.train,
        primary.train_end,
        customer_to_idx=mapping.customer_to_idx,
        asset_to_idx=mapping.asset_to_idx,
    )
    graph_snapshot.to_csv(primary_dir / "train_graph_snapshot.csv", index=False)
    baseline_pairs = _save_baseline_interactions(
        primary.train, mapping, primary_dir / "train_baseline_interactions.csv"
    )

    print("[7/7] Verifying leakage and writing checkpoint report...")
    primary_check = validate_primary_split(primary)
    rolling_checks = [validate_rolling_split(item) for item in rolling]
    leakage_ok = primary_check["ok"] and all(item["ok"] for item in rolling_checks)
    covered_rows = len(primary.train) + len(primary.validation) + len(primary.test)
    coverage_ok = covered_rows == len(data["transactions"])
    report.update(
        {
            "ok": bool(report["ok"] and leakage_ok and coverage_ok),
            "cleaning": cleaning_report,
            "primary_split": {
                "train": _split_stats(primary.train),
                "validation": _split_stats(primary.validation),
                "test": _split_stats(primary.test),
                "coverage": {
                    "clean_transactions": int(len(data["transactions"])),
                    "assigned_rows": int(covered_rows),
                    "unassigned_rows": int(len(data["transactions"]) - covered_rows),
                    "ok": coverage_ok,
                },
            },
            "cold_start_against_train_mapping": {
                "validation": _cold_start_stats(primary.validation, mapping),
                "test": _cold_start_stats(primary.test, mapping),
            },
            "temporal_graph": {
                "events": int(len(graph.frame)),
                "snapshot_edges": int(len(graph_snapshot)),
                "customers": mapping.num_customers,
                "assets": mapping.num_assets,
                "as_of": str(primary.train_end),
            },
            "baseline": {
                "train_buy_pairs": int(baseline_pairs),
                "mapping_fitted_on": mapping.fitted_on,
            },
            "primary_leakage": primary_check,
            "rolling_leakage": rolling_checks,
        }
    )
    save_report(report, reports_dir / "checkpoint1_report.json")

    summary_source = primary_dir / "primary_split_summary.csv"
    (reports_dir / "primary_split_summary.csv").write_text(
        summary_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    rolling_summary = output_dir / "rolling" / "rolling_split_summary.csv"
    (reports_dir / "rolling_split_summary.csv").write_text(
        rolling_summary.read_text(encoding="utf-8"), encoding="utf-8"
    )

    if not report["ok"]:
        raise ValueError("Leakage checks failed; see checkpoint1_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAR-Trans Checkpoint 1")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--skip-clean", action="store_true")
    args = parser.parse_args()
    result = build_checkpoint(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        skip_clean=args.skip_clean,
    )
    print(json.dumps(result["primary_split"], indent=2))
    print("[OK] Checkpoint 1 completed without temporal leakage.")


if __name__ == "__main__":
    main()
