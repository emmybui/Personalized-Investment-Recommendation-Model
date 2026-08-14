"""Clean the six FAR-Trans core CSV files without temporal leakage.

The source dataset reuses ``transactionID`` across customers.  Therefore the
transaction key is ``(customerID, transactionID)``; treating transactionID as
globally unique removes valid events and investors.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR


RAW_FILES = {
    "customers": "customer_information.csv",
    "assets": "asset_information.csv",
    "markets": "markets.csv",
    "close_prices": "close_prices.csv",
    "limit_prices": "limit_prices.csv",
    "transactions": "transactions.csv",
}

CLEAN_FILES = {
    name: filename.replace(".csv", "_clean.csv")
    for name, filename in RAW_FILES.items()
}

KNOWN_ASSET_CONFLICT_ISINS = frozenset(
    {"IE00B66F4759", "IE00BCRY6003", "LU0055631609"}
)


class DataConflictError(ValueError):
    """Raised when the source contains ambiguous rows for a logical key."""


def _read_csv(path: Path, *, dtype=None, parse_dates=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FAR-Trans file: {path}. See data/raw/README.txt."
        )
    return pd.read_csv(path, dtype=dtype, parse_dates=parse_dates)


def load_raw_tables(input_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load all six core files with ID and date types fixed at read time."""
    root = Path(input_dir)
    return {
        "customers": _read_csv(
            root / RAW_FILES["customers"],
            dtype={"customerID": "string"},
            parse_dates=["lastQuestionnaireDate", "timestamp"],
        ),
        "assets": _read_csv(
            root / RAW_FILES["assets"],
            dtype={"ISIN": "string", "marketID": "string"},
            parse_dates=["timestamp"],
        ),
        "markets": _read_csv(
            root / RAW_FILES["markets"],
            dtype={"exchangeID": "string", "marketID": "string"},
        ),
        "close_prices": _read_csv(
            root / RAW_FILES["close_prices"],
            dtype={"ISIN": "string", "closePrice": "float64"},
            parse_dates=["timestamp"],
        ),
        "limit_prices": _read_csv(
            root / RAW_FILES["limit_prices"],
            dtype={"ISIN": "string"},
            parse_dates=["minDate", "maxDate"],
        ),
        "transactions": _read_csv(
            root / RAW_FILES["transactions"],
            dtype={
                "customerID": "string",
                "ISIN": "string",
                "transactionID": "string",
                "marketID": "string",
            },
            parse_dates=["timestamp"],
        ),
    }


def _drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def _raise_on_conflicts(
    df: pd.DataFrame, key: list[str], table: str
) -> pd.DataFrame:
    """Remove exact duplicates and reject same-key rows with other differences."""
    clean = _drop_exact_duplicates(df)
    duplicate = clean.duplicated(key, keep=False)
    if duplicate.any():
        examples = clean.loc[duplicate, key].drop_duplicates().head(5)
        raise DataConflictError(
            f"{table}: conflicting rows for key {key}; examples="
            f"{examples.to_dict('records')}"
        )
    return clean


def _require(df: pd.DataFrame, columns: list[str], table: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{table}: missing columns {sorted(missing)}")


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "customerID",
        "customerType",
        "riskLevel",
        "investmentCapacity",
        "lastQuestionnaireDate",
        "timestamp",
    ]
    _require(df, required, "customer_information")
    out = _raise_on_conflicts(
        df.dropna(subset=["customerID", "timestamp"]),
        ["customerID", "timestamp"],
        "customer_information",
    )

    risk = out["riskLevel"].astype("string")
    out["is_risk_predicted"] = risk.str.startswith("Predicted_", na=False)
    out["riskLevel"] = (
        risk.str.replace("Predicted_", "", regex=False)
        .replace("Not_Available", "Unknown")
        .fillna("Unknown")
    )

    capacity = out["investmentCapacity"].astype("string")
    out["is_capacity_predicted"] = capacity.str.startswith(
        "Predicted_", na=False
    )
    out["investmentCapacity"] = (
        capacity.str.replace("Predicted_", "", regex=False)
        .replace({"GT300K": "CAP_GT300K", "Not_Available": "Unknown"})
        .fillna("Unknown")
    )
    out["customerType"] = out["customerType"].astype("string").fillna("Unknown")
    return out.sort_values(
        ["customerID", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def clean_assets(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "ISIN",
        "assetName",
        "assetShortName",
        "assetCategory",
        "assetSubCategory",
        "marketID",
        "sector",
        "industry",
        "timestamp",
    ]
    _require(df, required, "asset_information")
    out = _drop_exact_duplicates(df.dropna(subset=["ISIN", "timestamp"]))

    duplicate = out.duplicated(["ISIN", "timestamp"], keep=False)
    if duplicate.any():
        conflict_ids = set(out.loc[duplicate, "ISIN"])
        unknown = conflict_ids - KNOWN_ASSET_CONFLICT_ISINS
        if unknown:
            raise DataConflictError(
                "asset_information: unknown conflicting (ISIN, timestamp) "
                f"keys for {sorted(unknown)[:5]}"
            )
        warnings.warn(
            "asset_information: resolving the three documented dual "
            "Stock/MTF classifications by preferring MTF.",
            stacklevel=2,
        )
        out["_category_priority"] = out["assetCategory"].eq("MTF").astype(int)
        out = (
            out.sort_values(
                ["ISIN", "timestamp", "_category_priority"], kind="stable"
            )
            .drop_duplicates(["ISIN", "timestamp"], keep="last")
            .drop(columns="_category_priority")
        )

    for column in [
        "assetName",
        "assetShortName",
        "assetCategory",
        "assetSubCategory",
        "marketID",
        "sector",
        "industry",
    ]:
        out[column] = out[column].astype("string").fillna("Unknown")
    return out.sort_values(["ISIN", "timestamp"], kind="stable").reset_index(
        drop=True
    )


def clean_markets(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, ["marketID"], "markets")
    out = _raise_on_conflicts(
        df.dropna(subset=["marketID"]), ["marketID"], "markets"
    )
    for column in out.select_dtypes(include=["object", "string"]).columns:
        out[column] = out[column].astype("string").fillna("Unknown")
    return out.sort_values("marketID", kind="stable").reset_index(drop=True)


def clean_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, ["ISIN", "timestamp", "closePrice"], "close_prices")
    out = df.dropna(subset=["ISIN", "timestamp", "closePrice"])
    invalid = out["closePrice"] <= 0
    if invalid.any():
        warnings.warn(
            f"close_prices: dropping {int(invalid.sum())} non-positive prices",
            stacklevel=2,
        )
        out = out.loc[~invalid]
    out = _raise_on_conflicts(out, ["ISIN", "timestamp"], "close_prices")
    return out.sort_values(["ISIN", "timestamp"], kind="stable").reset_index(
        drop=True
    )


def clean_limit_prices(df: pd.DataFrame) -> pd.DataFrame:
    _require(
        df,
        ["ISIN", "minDate", "maxDate", "priceMinDate", "priceMaxDate", "profitability"],
        "limit_prices",
    )
    out = df.dropna(subset=["ISIN", "minDate", "maxDate"])
    invalid = out["minDate"] > out["maxDate"]
    if invalid.any():
        warnings.warn(
            f"limit_prices: dropping {int(invalid.sum())} invalid date ranges",
            stacklevel=2,
        )
        out = out.loc[~invalid]
    out = _raise_on_conflicts(out, ["ISIN"], "limit_prices")
    return out.sort_values("ISIN", kind="stable").reset_index(drop=True)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "customerID",
        "ISIN",
        "transactionID",
        "transactionType",
        "timestamp",
        "totalValue",
        "units",
        "channel",
        "marketID",
    ]
    _require(df, required, "transactions")
    out = df.dropna(
        subset=[
            "customerID",
            "ISIN",
            "transactionID",
            "transactionType",
            "timestamp",
            "totalValue",
            "units",
        ]
    )
    out = out.loc[out["transactionType"].isin(["Buy", "Sell"])]
    out = out.loc[(out["units"] > 0) & (out["totalValue"] >= 0)]

    # transactionID is only unique within a customer in FAR-Trans.
    out = _raise_on_conflicts(
        out, ["customerID", "transactionID"], "transactions"
    )
    return out.sort_values(
        ["timestamp", "customerID", "transactionID", "ISIN"], kind="stable"
    ).reset_index(drop=True)


CLEANERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "customers": clean_customers,
    "assets": clean_assets,
    "markets": clean_markets,
    "close_prices": clean_close_prices,
    "limit_prices": clean_limit_prices,
    "transactions": clean_transactions,
}


def run_cleaning(
    input_dir: Path | str = RAW_DIR,
    output_dir: Path | str = PROCESSED_DIR,
    report_path: Path | str | None = None,
) -> dict:
    """Clean all core tables, save them, and return a reproducibility report."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_tables(input_dir)
    cleaned = {name: CLEANERS[name](table) for name, table in raw.items()}

    # Enforce cross-table integrity after the table-local rules.  The source
    # contains one price/limit record for an ISIN absent from asset metadata;
    # it cannot be represented as a graph node and is removed from processed
    # model inputs while remaining untouched in raw data.
    asset_ids = set(cleaned["assets"]["ISIN"])
    for name in ["close_prices", "limit_prices"]:
        orphan = ~cleaned[name]["ISIN"].isin(asset_ids)
        if orphan.any():
            warnings.warn(
                f"{name}: dropping {int(orphan.sum())} rows whose ISIN is "
                "absent from asset_information",
                stacklevel=2,
            )
            cleaned[name] = cleaned[name].loc[~orphan].reset_index(drop=True)

    transactions = cleaned["transactions"]
    reference_checks = {
        "customerID": set(cleaned["customers"]["customerID"]),
        "ISIN": asset_ids,
        "marketID": set(cleaned["markets"]["marketID"]),
    }
    for column, known in reference_checks.items():
        unknown = ~transactions[column].isin(known)
        if unknown.any():
            raise DataConflictError(
                f"transactions: {int(unknown.sum())} rows have unknown {column}"
            )

    for name, table in cleaned.items():
        table.to_csv(output_dir / CLEAN_FILES[name], index=False)

    tx = cleaned["transactions"]
    report = {
        "ok": True,
        "transaction_key": ["customerID", "transactionID"],
        "tables": {
            name: {
                "input_rows": int(len(raw[name])),
                "output_rows": int(len(cleaned[name])),
                "removed_rows": int(len(raw[name]) - len(cleaned[name])),
                "missing_cells_after_cleaning": int(
                    cleaned[name].isna().sum().sum()
                ),
            }
            for name in RAW_FILES
        },
        "transactions": {
            "unique_composite_keys": int(
                tx[["customerID", "transactionID"]].drop_duplicates().shape[0]
            ),
            "globally_reused_transaction_ids": int(
                tx.loc[tx["transactionID"].duplicated(keep=False), "transactionID"].nunique()
            ),
            "min_timestamp": str(tx["timestamp"].min()),
            "max_timestamp": str(tx["timestamp"].max()),
        },
    }

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the six FAR-Trans CSVs")
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report", type=Path, default=Path("reports/cleaning_report.json"))
    args = parser.parse_args()
    report = run_cleaning(args.input_dir, args.output_dir, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
