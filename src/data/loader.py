"""Central data loader for all downstream experiments.

Usage:
    from src.data import FARTransLoader

    loader = FARTransLoader()
    data = loader.load_all()                   # 6 cleaned tables
    primary = loader.load_primary_split()       # train/val/test + snapshots
    rolling = loader.load_rolling_split(1)      # rolling window 1
"""

from pathlib import Path
import pandas as pd

from .config import PROCESSED_DIR, SPLITS_DIR, CORE_FILES, ROLLING_CUTOFFS

REQUIRED_COLUMNS = {
    "customers": {
        "customerID", "customerType", "riskLevel", "investmentCapacity",
        "lastQuestionnaireDate", "timestamp"
    },
    "assets": {
        "ISIN", "assetName", "assetShortName", "assetCategory",
        "assetSubCategory", "marketID", "sector", "industry", "timestamp"
    },
    "markets": {
        "exchangeID", "marketID", "name", "description", "country",
        "tradingDays", "tradingHours", "marketClass"
    },
    "close_prices": {"ISIN", "timestamp", "closePrice"},
    "limit_prices": {
        "ISIN", "minDate", "maxDate", "priceMinDate",
        "priceMaxDate", "profitability"
    },
    "transactions": {
        "customerID", "ISIN", "transactionID", "transactionType",
        "totalValue", "units", "channel", "marketID", "timestamp"
    },
}


class FARTransLoader:
    """Single entry point for all downstream experiments.

    The loader reads only processed data.  It does not perform cleaning.
    """

    def __init__(self, processed_dir=PROCESSED_DIR, splits_dir=SPLITS_DIR):
        self.processed_dir = Path(processed_dir)
        self.splits_dir = Path(splits_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, filename: str) -> Path:
        candidates = [
            self.processed_dir / filename,
            self.processed_dir / "cleaned" / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Cannot find {filename}. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    @staticmethod
    def _check_columns(df, name):
        missing = REQUIRED_COLUMNS[name] - set(df.columns)
        if missing:
            raise ValueError(f"{name}: missing columns: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Raw table loaders
    # ------------------------------------------------------------------

    def load_customers(self):
        df = pd.read_csv(
            self._resolve(CORE_FILES["customers"]),
            dtype={"customerID": "string"},
            parse_dates=["lastQuestionnaireDate", "timestamp"],
        )
        self._check_columns(df, "customers")
        return df

    def load_assets(self):
        df = pd.read_csv(
            self._resolve(CORE_FILES["assets"]),
            dtype={"ISIN": "string", "marketID": "string"},
            parse_dates=["timestamp"],
        )
        self._check_columns(df, "assets")
        return df

    def load_markets(self):
        df = pd.read_csv(
            self._resolve(CORE_FILES["markets"]),
            dtype={"exchangeID": "string", "marketID": "string"},
        )
        self._check_columns(df, "markets")
        return df

    def load_close_prices(self):
        df = pd.read_csv(
            self._resolve(CORE_FILES["close_prices"]),
            dtype={"ISIN": "string", "closePrice": "float64"},
            parse_dates=["timestamp"],
        )
        self._check_columns(df, "close_prices")
        return df

    def load_limit_prices(self):
        df = pd.read_csv(
            self._resolve(CORE_FILES["limit_prices"]),
            dtype={"ISIN": "string"},
            parse_dates=["minDate", "maxDate"],
        )
        self._check_columns(df, "limit_prices")
        return df

    def load_transactions(self):
        df = pd.read_csv(
            self._resolve(CORE_FILES["transactions"]),
            dtype={
                "customerID": "string",
                "ISIN": "string",
                "transactionID": "string",
                "marketID": "string",
            },
            parse_dates=["timestamp"],
        )
        self._check_columns(df, "transactions")
        return df.sort_values(
            ["timestamp", "transactionID"], kind="stable"
        ).reset_index(drop=True)

    def load_all(self):
        return {
            "customers": self.load_customers(),
            "assets": self.load_assets(),
            "markets": self.load_markets(),
            "close_prices": self.load_close_prices(),
            "limit_prices": self.load_limit_prices(),
            "transactions": self.load_transactions(),
        }

    # ------------------------------------------------------------------
    # Split loaders (for model training in Week 4-7)
    # ------------------------------------------------------------------

    def _load_split_csv(self, path, parse_dates=None, dtypes=None):
        """Load a CSV from a split directory with optional type hints."""
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        return pd.read_csv(path, parse_dates=parse_dates or [],
                           dtype=dtypes or {})

    _TX_DTYPES = {"customerID": "string", "ISIN": "string",
                  "transactionID": "string", "marketID": "string"}
    _SNAP_DTYPES = {"customerID": "string", "ISIN": "string",
                    "marketID": "string"}

    def load_primary_split(self):
        """Load the primary Train/Validation/Test split with snapshots.

        Returns a dict with keys:
            train, validation, test,
            train_customer_snapshot, train_asset_snapshot, train_prices,
            validation_customer_snapshot, validation_asset_snapshot,
            validation_prices.
        """
        d = self.splits_dir / "primary"
        return {
            "train": self._load_split_csv(
                d / "train_transactions.csv",
                parse_dates=["timestamp"], dtypes=self._TX_DTYPES),
            "validation": self._load_split_csv(
                d / "validation_transactions.csv",
                parse_dates=["timestamp"], dtypes=self._TX_DTYPES),
            "test": self._load_split_csv(
                d / "test_transactions.csv",
                parse_dates=["timestamp"], dtypes=self._TX_DTYPES),
            "train_customer_snapshot": self._load_split_csv(
                d / "train_customer_snapshot.csv",
                parse_dates=["timestamp", "snapshot_asof_date"]),
            "train_asset_snapshot": self._load_split_csv(
                d / "train_asset_snapshot.csv",
                parse_dates=["timestamp", "snapshot_asof_date"]),
            "train_prices": self._load_split_csv(
                d / "train_prices.csv",
                parse_dates=["timestamp"]),
            "validation_customer_snapshot": self._load_split_csv(
                d / "validation_customer_snapshot.csv",
                parse_dates=["timestamp", "snapshot_asof_date"]),
            "validation_asset_snapshot": self._load_split_csv(
                d / "validation_asset_snapshot.csv",
                parse_dates=["timestamp", "snapshot_asof_date"]),
            "validation_prices": self._load_split_csv(
                d / "validation_prices.csv",
                parse_dates=["timestamp"]),
        }

    def load_rolling_split(self, split_id):
        """Load rolling split 1..5 with snapshots.

        Returns a dict with keys:
            train, test, customer_snapshot, asset_snapshot, prices.
        """
        if not 1 <= split_id <= len(ROLLING_CUTOFFS):
            raise ValueError(
                f"split_id must be 1..{len(ROLLING_CUTOFFS)}, got {split_id}")
        cutoff = ROLLING_CUTOFFS[split_id - 1]
        d = self.splits_dir / "rolling" / f"rolling_{split_id:02d}_{cutoff}"
        return {
            "cutoff": cutoff,
            "train": self._load_split_csv(
                d / "train_transactions.csv",
                parse_dates=["timestamp"], dtypes=self._TX_DTYPES),
            "test": self._load_split_csv(
                d / "test_transactions.csv",
                parse_dates=["timestamp"], dtypes=self._TX_DTYPES),
            "customer_snapshot": self._load_split_csv(
                d / "customer_snapshot.csv",
                parse_dates=["timestamp", "snapshot_asof_date"]),
            "asset_snapshot": self._load_split_csv(
                d / "asset_snapshot.csv",
                parse_dates=["timestamp", "snapshot_asof_date"]),
            "prices": self._load_split_csv(
                d / "prices_upto_t.csv",
                parse_dates=["timestamp"]),
        }


# -----------------------------------------------------------------------
# Shared ID Mapping  (needed by Popularity, BPR, LightGCN, TGN)
# -----------------------------------------------------------------------

def build_id_mapping(transactions):
    """Create deterministic integer mappings for customer and asset IDs.

    Returns (customer_to_idx, asset_to_idx) where each is a dict
    mapping string ID to int index.  The same mapping must be used
    across all baselines and the final TGN model.
    """
    customer_ids = sorted(transactions["customerID"].unique())
    asset_ids = sorted(transactions["ISIN"].unique())
    return (
        {cid: idx for idx, cid in enumerate(customer_ids)},
        {isin: idx for idx, isin in enumerate(asset_ids)},
    )
