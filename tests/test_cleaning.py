import pandas as pd
import unittest

from src.data.clean import DataConflictError, clean_assets, clean_transactions


def _transaction(customer, transaction_id, timestamp="2022-01-01", units=1.0):
    return {
        "customerID": customer,
        "ISIN": "A1",
        "transactionID": transaction_id,
        "transactionType": "Buy",
        "timestamp": pd.Timestamp(timestamp),
        "totalValue": 10.0,
        "units": units,
        "channel": "Branch",
        "marketID": "M1",
    }


class CleaningTests(unittest.TestCase):
    def test_transaction_id_is_scoped_to_customer(self):
        source = pd.DataFrame(
            [_transaction("C1", "42"), _transaction("C2", "42")]
        )
        cleaned = clean_transactions(source)
        self.assertEqual(len(cleaned), 2)
        self.assertFalse(
            cleaned.duplicated(["customerID", "transactionID"]).any()
        )

    def test_conflicting_composite_transaction_key_fails(self):
        source = pd.DataFrame(
            [
                _transaction("C1", "42", units=1),
                _transaction("C1", "42", units=2),
            ]
        )
        with self.assertRaises(DataConflictError):
            clean_transactions(source)

    def test_asset_missing_metadata_becomes_unknown(self):
        source = pd.DataFrame(
            [
                {
                    "ISIN": "A1",
                    "assetName": None,
                    "assetShortName": None,
                    "assetCategory": "MTF",
                    "assetSubCategory": None,
                    "marketID": "M1",
                    "sector": None,
                    "industry": None,
                    "timestamp": pd.Timestamp("2020-01-01"),
                }
            ]
        )
        cleaned = clean_assets(source)
        self.assertEqual(cleaned.loc[0, "sector"], "Unknown")
        self.assertEqual(cleaned.loc[0, "industry"], "Unknown")
        self.assertEqual(cleaned.loc[0, "assetName"], "Unknown")
        self.assertEqual(cleaned.isna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
