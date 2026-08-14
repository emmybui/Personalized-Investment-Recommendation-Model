"""Leakage-safe temporal investor--asset graph construction.

The graph is an event stream rather than one graph aggregated over the entire
dataset.  Calling the builders with ``as_of=t`` guarantees that every edge and
edge feature was observable at or before ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

from .config import EVENT_ORDER


EVENT_COLUMNS = [
    "event_id",
    "customerID",
    "ISIN",
    "transactionID",
    "src_customer_idx",
    "dst_asset_idx",
    "src_node_idx",
    "dst_node_idx",
    "timestamp",
    "unix_timestamp",
    "event_type",
    "is_buy",
    "signed_units",
    "total_value",
]


@dataclass(frozen=True)
class TemporalGraphEvents:
    """Ordered bipartite events and the train-fitted node mappings."""

    frame: pd.DataFrame
    customer_to_idx: dict[str, int]
    asset_to_idx: dict[str, int]
    as_of: pd.Timestamp | None

    @property
    def num_customers(self) -> int:
        return len(self.customer_to_idx)

    @property
    def num_assets(self) -> int:
        return len(self.asset_to_idx)

    @property
    def num_nodes(self) -> int:
        return self.num_customers + self.num_assets

    def to_pyg_temporal_data(self):
        """Convert to ``torch_geometric.data.TemporalData`` when installed.

        PyTorch and PyG are optional so the cleaning/baseline pipeline remains
        lightweight.  Message features are event-local and need no future-fit
        normalization: ``[is_buy, signed_log_units, log_total_value]``.
        """
        try:
            import torch
            from torch_geometric.data import TemporalData
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install the graph extra (torch and torch-geometric) to use "
                "to_pyg_temporal_data()."
            ) from exc

        frame = self.frame
        signed_log_units = np.sign(frame["signed_units"].to_numpy()) * np.log1p(
            np.abs(frame["signed_units"].to_numpy())
        )
        messages = np.column_stack(
            [
                frame["is_buy"].to_numpy(dtype=np.float32),
                signed_log_units.astype(np.float32),
                np.log1p(frame["total_value"].to_numpy()).astype(np.float32),
            ]
        )
        return TemporalData(
            src=torch.as_tensor(frame["src_node_idx"].to_numpy(), dtype=torch.long),
            dst=torch.as_tensor(frame["dst_node_idx"].to_numpy(), dtype=torch.long),
            t=torch.as_tensor(frame["unix_timestamp"].to_numpy(), dtype=torch.long),
            msg=torch.as_tensor(messages, dtype=torch.float32),
            y=torch.as_tensor(frame["is_buy"].to_numpy(), dtype=torch.long),
        )


def _mapping(values: pd.Series) -> dict[str, int]:
    return {value: idx for idx, value in enumerate(sorted(values.astype(str).unique()))}


def build_temporal_graph_events(
    transactions: pd.DataFrame,
    *,
    as_of=None,
    customer_to_idx: Mapping[str, int] | None = None,
    asset_to_idx: Mapping[str, int] | None = None,
    drop_unknown: bool = False,
) -> TemporalGraphEvents:
    """Build a deterministic temporal bipartite event stream.

    If mappings are omitted they are learned only from the already filtered
    history (``timestamp <= as_of``).  For validation/test, pass mappings fitted
    on train; unknown cold-start nodes then raise unless ``drop_unknown=True``.
    """
    required = {
        "customerID",
        "ISIN",
        "transactionID",
        "transactionType",
        "timestamp",
        "units",
        "totalValue",
    }
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions: missing graph columns {sorted(missing)}")

    frame = transactions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    cutoff = pd.Timestamp(as_of) if as_of is not None else None
    if cutoff is not None:
        frame = frame.loc[frame["timestamp"] <= cutoff].copy()
    frame = frame.sort_values(EVENT_ORDER, kind="stable").reset_index(drop=True)

    customer_map = dict(customer_to_idx or _mapping(frame["customerID"]))
    asset_map = dict(asset_to_idx or _mapping(frame["ISIN"]))
    frame["src_customer_idx"] = frame["customerID"].astype(str).map(customer_map)
    frame["dst_asset_idx"] = frame["ISIN"].astype(str).map(asset_map)
    unknown = frame["src_customer_idx"].isna() | frame["dst_asset_idx"].isna()
    if unknown.any() and not drop_unknown:
        raise ValueError(
            f"Graph contains {int(unknown.sum())} events with IDs absent from "
            "the supplied train mapping. Use drop_unknown=True only when the "
            "cold-start evaluation policy explicitly excludes them."
        )
    frame = frame.loc[~unknown].copy()

    frame["src_customer_idx"] = frame["src_customer_idx"].astype("int64")
    frame["dst_asset_idx"] = frame["dst_asset_idx"].astype("int64")
    frame["src_node_idx"] = frame["src_customer_idx"]
    frame["dst_node_idx"] = len(customer_map) + frame["dst_asset_idx"]
    frame["event_id"] = (
        frame["customerID"].astype(str)
        + ":"
        + frame["transactionID"].astype(str)
    )
    frame["event_type"] = frame["transactionType"].astype(str)
    frame["is_buy"] = frame["event_type"].eq("Buy").astype("int8")
    frame["signed_units"] = frame["units"].astype(float).where(
        frame["is_buy"].eq(1), -frame["units"].astype(float)
    )
    frame["total_value"] = frame["totalValue"].astype(float)
    frame["unix_timestamp"] = (
        frame["timestamp"].astype("int64") // 1_000_000_000
    )
    return TemporalGraphEvents(
        frame=frame[EVENT_COLUMNS].reset_index(drop=True),
        customer_to_idx=customer_map,
        asset_to_idx=asset_map,
        as_of=cutoff,
    )


def build_interaction_snapshot(
    transactions: pd.DataFrame,
    as_of,
    *,
    customer_to_idx: Mapping[str, int] | None = None,
    asset_to_idx: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Aggregate event edges into a graph snapshot using history only."""
    events = build_temporal_graph_events(
        transactions,
        as_of=as_of,
        customer_to_idx=customer_to_idx,
        asset_to_idx=asset_to_idx,
    ).frame
    if events.empty:
        return pd.DataFrame(
            columns=[
                "src_customer_idx",
                "dst_asset_idx",
                "event_count",
                "buy_count",
                "sell_count",
                "net_units",
                "total_value",
                "last_timestamp",
            ]
        )
    events = events.assign(
        buy_count=events["is_buy"], sell_count=1 - events["is_buy"]
    )
    return (
        events.groupby(["src_customer_idx", "dst_asset_idx"], as_index=False)
        .agg(
            event_count=("event_id", "size"),
            buy_count=("buy_count", "sum"),
            sell_count=("sell_count", "sum"),
            net_units=("signed_units", "sum"),
            total_value=("total_value", "sum"),
            last_timestamp=("timestamp", "max"),
        )
        .sort_values(["src_customer_idx", "dst_asset_idx"], kind="stable")
        .reset_index(drop=True)
    )


def iter_event_windows(
    events: TemporalGraphEvents, frequency: str = "ME"
) -> Iterator[tuple[pd.Period, pd.DataFrame]]:
    """Yield chronological event windows (monthly by default)."""
    frame = events.frame.copy()
    if frame.empty:
        return
    periods = frame["timestamp"].dt.to_period(frequency.rstrip("E"))
    for period, group in frame.groupby(periods, sort=True):
        yield period, group.reset_index(drop=True)
