import pandas as pd


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
    as_of = pd.Timestamp(as_of) if as_of is not None else None

    # Derive point-in-time asset universe A(t) if DataFrame or None passed
    if isinstance(eligible_assets, pd.DataFrame):
        if "timestamp" in eligible_assets.columns and as_of is not None:
            snap = eligible_assets.loc[eligible_assets["timestamp"] <= as_of]
        else:
            snap = eligible_assets
        asset_universe = set(snap["ISIN"].dropna().unique())
    elif eligible_assets is not None:
        asset_universe = set(eligible_assets)
    else:
        # Fallback: extract assets from transactions up to as_of
        if as_of is not None:
            hist_tx = transactions.loc[transactions["timestamp"] <= as_of]
        else:
            hist_tx = transactions
        asset_universe = set(hist_tx["ISIN"].dropna().unique())

    held = currently_held_assets(transactions, customer_id, as_of)
    return asset_universe - held
