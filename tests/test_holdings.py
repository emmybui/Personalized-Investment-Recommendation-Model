"""Unit tests for holdings state reconstruction and point-in-time candidate generator.

Covered edge cases:
1. BUY 100 -> net_units = 100, currently_held = True
2. BUY 100 + SELL 100 -> net_units = 0, currently_held = False
3. BUY 100 + SELL 40 -> net_units = 60, currently_held = True
4. SELL before BUY (net negative) -> net_units <= 0, currently_held = False
5. Holdings temporal leakage: BUY in 2020, SELL in 2022 -> at cutoff 2021, asset STILL held
6. Candidate Universe Cutoff: Asset A in 2020, Asset B in 2022 -> at cutoff 2021, candidates include A but NOT B
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.state import build_holdings_asof, currently_held_assets, get_candidate_assets


def test_holdings_edge_cases():
    tx_data = pd.DataFrame([
        # Customer C1, Asset A1: BUY 100 (2020-01-01)
        {"customerID": "C1", "ISIN": "A1", "transactionType": "Buy", "units": 100.0, "timestamp": pd.Timestamp("2020-01-01")},
        # SELL A1 in 2022 (future relative to cutoff 2021-06-01)
        {"customerID": "C1", "ISIN": "A1", "transactionType": "Sell", "units": 100.0, "timestamp": pd.Timestamp("2022-01-01")},
        
        # Customer C1, Asset A2: BUY 100 + SELL 100 (both in 2021)
        {"customerID": "C1", "ISIN": "A2", "transactionType": "Buy", "units": 100.0, "timestamp": pd.Timestamp("2021-01-01")},
        {"customerID": "C1", "ISIN": "A2", "transactionType": "Sell", "units": 100.0, "timestamp": pd.Timestamp("2021-02-01")},
        
        # Customer C1, Asset A3: BUY 100 + SELL 40 (net 60)
        {"customerID": "C1", "ISIN": "A3", "transactionType": "Buy", "units": 100.0, "timestamp": pd.Timestamp("2021-01-01")},
        {"customerID": "C1", "ISIN": "A3", "transactionType": "Sell", "units": 40.0, "timestamp": pd.Timestamp("2021-02-01")},
        
        # Customer C1, Asset A4: SELL 100 before BUY (negative net)
        {"customerID": "C1", "ISIN": "A4", "transactionType": "Sell", "units": 100.0, "timestamp": pd.Timestamp("2021-01-01")},

        # Asset B1 created in 2022 (future asset)
        {"customerID": "C2", "ISIN": "B1", "transactionType": "Buy", "units": 50.0, "timestamp": pd.Timestamp("2022-03-01")},
    ])

    cutoff = "2021-06-01"
    holdings = build_holdings_asof(tx_data, cutoff)

    # 1 & 5. BUY in 2020, SELL in 2022 -> at cutoff 2021-06-01, A1 MUST STILL BE HELD (100 units)
    h_a1 = holdings.loc[(holdings["customerID"] == "C1") & (holdings["ISIN"] == "A1")]
    assert not h_a1.empty
    assert h_a1["net_units"].values[0] == 100.0, f"Expected 100.0, got {h_a1['net_units'].values[0]}"
    assert h_a1["currently_held"].values[0] == True

    # 2. BUY 100 + SELL 100 -> net 0, currently_held = False
    h_a2 = holdings.loc[(holdings["customerID"] == "C1") & (holdings["ISIN"] == "A2")]
    assert not h_a2.empty
    assert h_a2["net_units"].values[0] == 0.0
    assert h_a2["currently_held"].values[0] == False

    # 3. BUY 100 + SELL 40 -> net 60, currently_held = True
    h_a3 = holdings.loc[(holdings["customerID"] == "C1") & (holdings["ISIN"] == "A3")]
    assert not h_a3.empty
    assert h_a3["net_units"].values[0] == 60.0
    assert h_a3["currently_held"].values[0] == True

    # 4. SELL 100 before BUY -> net -100, currently_held = False
    h_a4 = holdings.loc[(holdings["customerID"] == "C1") & (holdings["ISIN"] == "A4")]
    assert not h_a4.empty
    assert h_a4["net_units"].values[0] == -100.0
    assert h_a4["currently_held"].values[0] == False

    # Currently held set for C1 at 2021-06-01 should be {'A1', 'A3'}
    held_set = currently_held_assets(tx_data, "C1", cutoff)
    assert held_set == {"A1", "A3"}, f"Expected {{'A1', 'A3'}}, got {held_set}"

    # 6. Point-in-time Candidate Universe test:
    # Asset universe at 2021-06-01 consists ONLY of known assets up to cutoff: {'A1', 'A2', 'A3', 'A4'}
    # B1 was created in 2022, so it MUST NOT be in candidate set at 2021-06-01 cutoff!
    candidates = get_candidate_assets(tx_data, "C1", eligible_assets=None, as_of=cutoff)
    assert "B1" not in candidates, f"Future asset B1 leaked into candidate set at {cutoff}!"
    assert candidates == {"A2", "A4"}, f"Expected {{'A2', 'A4'}}, got {candidates}"

    print("[PASS] All holdings and point-in-time candidate edge-case tests passed!")


if __name__ == "__main__":
    test_holdings_edge_cases()
