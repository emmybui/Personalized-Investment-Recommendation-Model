import unittest

import pandas as pd

from src.data.state import (
    build_holdings_asof,
    build_point_in_time_state,
    currently_held_assets,
    get_candidate_assets,
)


class HoldingsTests(unittest.TestCase):
    def setUp(self):
        self.transactions = pd.DataFrame(
            [
                {"customerID": "C1", "ISIN": "A1", "transactionType": "Buy", "units": 100.0, "timestamp": pd.Timestamp("2020-01-01")},
                {"customerID": "C1", "ISIN": "A1", "transactionType": "Sell", "units": 100.0, "timestamp": pd.Timestamp("2022-01-01")},
                {"customerID": "C1", "ISIN": "A2", "transactionType": "Buy", "units": 100.0, "timestamp": pd.Timestamp("2021-01-01")},
                {"customerID": "C1", "ISIN": "A2", "transactionType": "Sell", "units": 100.0, "timestamp": pd.Timestamp("2021-02-01")},
                {"customerID": "C1", "ISIN": "A3", "transactionType": "Buy", "units": 100.0, "timestamp": pd.Timestamp("2021-01-01")},
                {"customerID": "C1", "ISIN": "A3", "transactionType": "Sell", "units": 40.0, "timestamp": pd.Timestamp("2021-02-01")},
                {"customerID": "C1", "ISIN": "A4", "transactionType": "Sell", "units": 100.0, "timestamp": pd.Timestamp("2021-01-01")},
                {"customerID": "C2", "ISIN": "B1", "transactionType": "Buy", "units": 50.0, "timestamp": pd.Timestamp("2022-03-01")},
            ]
        )
        self.cutoff = "2021-06-01"

    def test_holdings_use_only_events_before_cutoff(self):
        holdings = build_holdings_asof(self.transactions, self.cutoff)
        units = holdings.set_index(["customerID", "ISIN"])["net_units"]
        self.assertEqual(units.loc[("C1", "A1")], 100.0)
        self.assertEqual(units.loc[("C1", "A2")], 0.0)
        self.assertEqual(units.loc[("C1", "A3")], 60.0)
        self.assertEqual(units.loc[("C1", "A4")], -100.0)
        self.assertEqual(
            currently_held_assets(self.transactions, "C1", self.cutoff),
            {"A1", "A3"},
        )

    def test_future_asset_is_not_a_candidate(self):
        candidates = get_candidate_assets(
            self.transactions, "C1", as_of=self.cutoff
        )
        self.assertEqual(candidates, {"A2", "A4"})
        self.assertNotIn("B1", candidates)

    def test_reusable_point_in_time_state(self):
        state = build_point_in_time_state(self.transactions, self.cutoff)
        self.assertEqual(state.candidates("C1"), {"A2", "A4"})
        self.assertEqual(
            state.candidates("UNKNOWN"), {"A1", "A2", "A3", "A4"}
        )


if __name__ == "__main__":
    unittest.main()
