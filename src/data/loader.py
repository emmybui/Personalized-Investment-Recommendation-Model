"""Central data loader for all downstream experiments.

Usage:
    from src.data import FARTransLoader

    loader = FARTransLoader()
    data = loader.load_all()                   # 6 cleaned tables
    primary = loader.load_primary_split()       # train/val/test + snapshots
    rolling = loader.load_rolling_split(1)      # rolling window 1
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .config import (
    CORE_FILES,
    EVENT_ORDER,
    PROCESSED_DIR,
    ROLLING_CUTOFFS,
    SPLITS_DIR,
)

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
        return df.sort_values(EVENT_ORDER, kind="stable").reset_index(drop=True)

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

    def load_primary_split(self, include_snapshots=True):
        """Load the primary Train/Validation/Test split with snapshots.

        Returns a dict with keys:
            train, validation, test,
            train_customer_snapshot, train_asset_snapshot, train_prices,
            validation_customer_snapshot, validation_asset_snapshot,
            validation_prices.
        """
        d = self.splits_dir / "primary"
        result = {
            "train": self._load_split_csv(
                d / "train_transactions.csv",
                parse_dates=["timestamp"], dtypes=self._TX_DTYPES),
            "validation": self._load_split_csv(
                d / "validation_transactions.csv",
                parse_dates=["timestamp"], dtypes=self._TX_DTYPES),
            "test": self._load_split_csv(
                d / "test_transactions.csv",
                parse_dates=["timestamp"], dtypes=self._TX_DTYPES),
        }
        if not include_snapshots:
            return result
        result.update({
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
        })
        return result

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

    def _primary_transaction_path(self, split: str) -> Path:
        names = {
            "train": "train_transactions.csv",
            "validation": "validation_transactions.csv",
            "test": "test_transactions.csv",
        }
        if split not in names:
            raise ValueError(f"split must be one of {sorted(names)}, got {split!r}")
        path = self.splits_dir / "primary" / names[split]
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        return path

    def iter_transaction_batches(
        self,
        split: str = "train",
        *,
        batch_size: int = 65_536,
        mapping=None,
        drop_unknown: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Stream a primary transaction split without loading it all in RAM.

        When ``mapping`` is supplied, integer ``customer_idx``/``asset_idx``
        columns are added.  Validation/test should use a mapping fitted on
        train.  Unknown cold-start rows raise by default so they cannot vanish
        silently from evaluation.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        path = self._primary_transaction_path(split)
        reader = pd.read_csv(
            path,
            dtype=self._TX_DTYPES,
            parse_dates=["timestamp"],
            chunksize=batch_size,
        )
        for batch in reader:
            batch = batch.sort_values(EVENT_ORDER, kind="stable").reset_index(drop=True)
            if mapping is not None:
                batch = encode_transactions(
                    batch, mapping, drop_unknown=drop_unknown
                )
            yield batch

    def fit_train_mapping(self):
        """Fit deterministic node IDs from train only (default leakage policy)."""
        train = self._load_split_csv(
            self._primary_transaction_path("train"),
            parse_dates=["timestamp"],
            dtypes=self._TX_DTYPES,
        )
        return fit_id_mapping(train)

    def load_baseline_interactions(
        self,
        split: str = "train",
        *,
        mapping=None,
        buys_only: bool = True,
        drop_unknown: bool = False,
    ):
        """Return indexed COO-style arrays ready for recommendation baselines."""
        mapping = mapping or self.fit_train_mapping()
        frame = self._load_split_csv(
            self._primary_transaction_path(split),
            parse_dates=["timestamp"],
            dtypes=self._TX_DTYPES,
        )
        if buys_only:
            frame = frame.loc[frame["transactionType"].eq("Buy")].copy()
        encoded = encode_transactions(frame, mapping, drop_unknown=drop_unknown)
        grouped = (
            encoded.groupby(["customer_idx", "asset_idx"], as_index=False)
            .agg(weight=("transactionID", "size"), last_timestamp=("timestamp", "max"))
            .sort_values(["customer_idx", "asset_idx"], kind="stable")
        )
        return BaselineInteractions(
            row=grouped["customer_idx"].to_numpy(dtype=np.int64),
            col=grouped["asset_idx"].to_numpy(dtype=np.int64),
            weight=grouped["weight"].to_numpy(dtype=np.float32),
            last_timestamp=grouped["last_timestamp"].to_numpy(),
            shape=(mapping.num_customers, mapping.num_assets),
            mapping=mapping,
        )

    def load_temporal_graph_events(
        self,
        split: str = "train",
        *,
        mapping=None,
        drop_unknown: bool = False,
    ):
        """Load one split as the same indexed event stream used by TGN."""
        from .graph import build_temporal_graph_events

        mapping = mapping or self.fit_train_mapping()
        frame = self._load_split_csv(
            self._primary_transaction_path(split),
            parse_dates=["timestamp"],
            dtypes=self._TX_DTYPES,
        )
        return build_temporal_graph_events(
            frame,
            customer_to_idx=mapping.customer_to_idx,
            asset_to_idx=mapping.asset_to_idx,
            drop_unknown=drop_unknown,
        )


# -----------------------------------------------------------------------
# Shared ID Mapping  (needed by Popularity, BPR, LightGCN, TGN)
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class IDMapping:
    customer_to_idx: dict[str, int]
    asset_to_idx: dict[str, int]
    fitted_on: str = "train"

    @property
    def num_customers(self) -> int:
        return len(self.customer_to_idx)

    @property
    def num_assets(self) -> int:
        return len(self.asset_to_idx)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fitted_on": self.fitted_on,
                    "customer_to_idx": self.customer_to_idx,
                    "asset_to_idx": self.asset_to_idx,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class BaselineInteractions:
    """Sparse COO-compatible user--asset interaction arrays."""

    row: np.ndarray
    col: np.ndarray
    weight: np.ndarray
    last_timestamp: np.ndarray
    shape: tuple[int, int]
    mapping: IDMapping

    def to_scipy_csr(self):
        try:
            from scipy.sparse import coo_matrix
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install scipy to create a CSR matrix.") from exc
        return coo_matrix((self.weight, (self.row, self.col)), shape=self.shape).tocsr()


def fit_id_mapping(transactions: pd.DataFrame, fitted_on: str = "train") -> IDMapping:
    """Fit deterministic IDs on the supplied history (train by default)."""
    customer_ids = sorted(transactions["customerID"].dropna().astype(str).unique())
    asset_ids = sorted(transactions["ISIN"].dropna().astype(str).unique())
    return IDMapping(
        customer_to_idx={cid: idx for idx, cid in enumerate(customer_ids)},
        asset_to_idx={isin: idx for idx, isin in enumerate(asset_ids)},
        fitted_on=fitted_on,
    )


def encode_transactions(
    transactions: pd.DataFrame,
    mapping: IDMapping,
    *,
    drop_unknown: bool = False,
) -> pd.DataFrame:
    """Apply a train-fitted mapping and make cold-start handling explicit."""
    frame = transactions.copy()
    frame["customer_idx"] = frame["customerID"].astype(str).map(
        mapping.customer_to_idx
    )
    frame["asset_idx"] = frame["ISIN"].astype(str).map(mapping.asset_to_idx)
    unknown = frame["customer_idx"].isna() | frame["asset_idx"].isna()
    if unknown.any() and not drop_unknown:
        unknown_customers = frame.loc[
            frame["customer_idx"].isna(), "customerID"
        ].nunique()
        unknown_assets = frame.loc[frame["asset_idx"].isna(), "ISIN"].nunique()
        raise ValueError(
            f"{int(unknown.sum())} cold-start rows are outside the train mapping "
            f"({unknown_customers} customers, {unknown_assets} assets). "
            "Set drop_unknown=True only for an explicitly reported warm-start metric."
        )
    frame = frame.loc[~unknown].copy()
    frame[["customer_idx", "asset_idx"]] = frame[
        ["customer_idx", "asset_idx"]
    ].astype("int64")
    return frame.reset_index(drop=True)


def build_id_mapping(transactions):
    """Backward-compatible tuple API; mappings are fitted on supplied rows."""
    mapping = fit_id_mapping(transactions)
    return mapping.customer_to_idx, mapping.asset_to_idx
