import pandas as pd
import unittest

from src.data.graph import (
    build_interaction_snapshot,
    build_temporal_graph_events,
)
from src.data.loader import encode_transactions, fit_id_mapping
from src.data.config import PRIMARY_TEST_END
from src.data.snapshot import get_customer_snapshot_asof


def _events():
    return pd.DataFrame(
        [
            {
                "customerID": "C1",
                "ISIN": "A1",
                "transactionID": "2",
                "transactionType": "Buy",
                "timestamp": pd.Timestamp("2020-01-02"),
                "units": 10.0,
                "totalValue": 100.0,
            },
            {
                "customerID": "C1",
                "ISIN": "A1",
                "transactionID": "1",
                "transactionType": "Sell",
                "timestamp": pd.Timestamp("2020-01-03"),
                "units": 4.0,
                "totalValue": 50.0,
            },
            {
                "customerID": "C2",
                "ISIN": "FUTURE",
                "transactionID": "3",
                "transactionType": "Buy",
                "timestamp": pd.Timestamp("2022-01-01"),
                "units": 1.0,
                "totalValue": 20.0,
            },
        ]
    )


class TemporalGraphTests(unittest.TestCase):
    def test_customer_snapshot_never_uses_future_profile(self):
        profiles = pd.DataFrame(
            [
                {"customerID": "C1", "timestamp": pd.Timestamp("2020-01-01"), "risk": "Low"},
                {"customerID": "C1", "timestamp": pd.Timestamp("2021-01-01"), "risk": "High"},
            ]
        )
        snapshot = get_customer_snapshot_asof(profiles, "2020-06-30")
        self.assertEqual(snapshot.loc[0, "risk"], "Low")
        self.assertLessEqual(
            snapshot.loc[0, "timestamp"], snapshot.loc[0, "snapshot_asof_date"]
        )

    def test_temporal_graph_filters_future_and_offsets_asset_nodes(self):
        graph = build_temporal_graph_events(_events(), as_of="2020-12-31")
        self.assertEqual(len(graph.frame), 2)
        self.assertNotIn("FUTURE", graph.asset_to_idx)
        self.assertLessEqual(
            graph.frame["timestamp"].max(), pd.Timestamp("2020-12-31")
        )
        self.assertTrue(
            (graph.frame["dst_node_idx"] >= graph.num_customers).all()
        )
        self.assertEqual(graph.frame["signed_units"].tolist(), [10.0, -4.0])

    def test_interaction_snapshot_uses_only_history(self):
        snapshot = build_interaction_snapshot(_events(), "2020-12-31")
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot.loc[0, "event_count"], 2)
        self.assertEqual(snapshot.loc[0, "net_units"], 6.0)

    def test_train_mapping_makes_cold_start_policy_explicit(self):
        events = _events()
        train = events.loc[events["timestamp"] < "2021-01-01"]
        mapping = fit_id_mapping(train)
        with self.assertRaisesRegex(ValueError, "cold-start"):
            encode_transactions(events, mapping)
        warm = encode_transactions(events, mapping, drop_unknown=True)
        self.assertEqual(len(warm), 2)

    def test_primary_protocol_includes_last_transaction_day(self):
        self.assertEqual(
            pd.Timestamp(PRIMARY_TEST_END), pd.Timestamp("2022-11-30 23:59:59")
        )


if __name__ == "__main__":
    unittest.main()
