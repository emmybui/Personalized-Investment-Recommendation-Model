from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class PointInTimeState:
    """Reusable candidate state computed once for one cutoff."""

    as_of: pd.Timestamp
    asset_universe: frozenset[str]
    held_by_customer: Mapping[str, frozenset[str]]

    def candidates(self, customer_id: str) -> set[str]:
        return set(self.asset_universe - self.held_by_customer.get(customer_id, frozenset()))


def build_holdings_asof(transactions: pd.DataFrame, as_of):
    """Reconstruct net units held by each customer/asset at as_of.

    BUY adds units; SELL subtracts units. Only events <= as_of are used.
    The result is the state needed later to enforce the thesis requirement
    that recommendations exclude assets currently held by an investor.
    """
    as_of = pd.Timestamp(as_of)
    hist = transactions.loc[transactions["timestamp"] <= as_of].copy()

    if hist.empty:
        return pd.DataFrame(
            columns=["customerID", "ISIN", "net_units", "currently_held"]
        )

    hist["signed_units"] = hist["units"].astype(float)
    # Case-insensitive match for 'Sell'
    hist.loc[hist["transactionType"].astype(str).str.strip().str.capitalize() == "Sell", "signed_units"] *= -1

    holdings = (
        hist.groupby(["customerID", "ISIN"], as_index=False)["signed_units"]
        .sum()
        .rename(columns={"signed_units": "net_units"})
    )

    holdings["currently_held"] = holdings["net_units"] > 0
    return holdings


def build_point_in_time_state(
    transactions: pd.DataFrame, as_of, eligible_assets=None
) -> PointInTimeState:
    """Precompute holdings and candidates for efficient batched evaluation."""
    cutoff = pd.Timestamp(as_of)
    holdings = build_holdings_asof(transactions, cutoff)
    held = holdings.loc[holdings["currently_held"]]
    held_by_customer = {
        str(customer_id): frozenset(group["ISIN"].astype(str))
        for customer_id, group in held.groupby("customerID", sort=False)
    }

    if isinstance(eligible_assets, pd.DataFrame):
        available = eligible_assets
        if "timestamp" in available.columns:
            available = available.loc[available["timestamp"] <= cutoff]
        universe = frozenset(available["ISIN"].dropna().astype(str).unique())
    elif eligible_assets is not None:
        universe = frozenset(str(asset) for asset in eligible_assets)
    else:
        history = transactions.loc[transactions["timestamp"] <= cutoff]
        universe = frozenset(history["ISIN"].dropna().astype(str).unique())

    return PointInTimeState(cutoff, universe, held_by_customer)


def currently_held_assets(transactions: pd.DataFrame, customer_id, as_of):
    """Return the set of ISINs currently held by a customer at as_of cutoff."""
    holdings = build_holdings_asof(transactions, as_of)
    return set(
        holdings.loc[
            (holdings["customerID"] == customer_id)
            & holdings["currently_held"],
            "ISIN",
        ]
    )


def get_candidate_assets(transactions: pd.DataFrame, customer_id, eligible_assets=None, as_of=None):
    """Return eligible candidate ISINs for recommendation to customer_id at cutoff as_of.

    Formula (Point-in-Time candidate selection):
        C(u, t) = A(t) - H(u, t)

    Where:
        A(t) = eligible_assets known/available at time t (e.g. ISINs from asset_snapshot@t)
        H(u, t) = assets currently held by customer u at time t (net units > 0)

    If eligible_assets is a DataFrame (asset_snapshot), its ISIN column is extracted.
    If eligible_assets is None, A(t) is derived from transaction history up to as_of
    to strictly prevent future universe leakage.
    """
    if as_of is None:
        raise ValueError("as_of is required for leakage-safe candidate generation")
    state = build_point_in_time_state(transactions, as_of, eligible_assets)
    return state.candidates(str(customer_id))
