from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Proposal primary protocol:
# train: up to 2021
# validation: 2022 Q1-Q2
# test: 2022 Q3-Q4
PRIMARY_TRAIN_END = "2021-12-31 23:59:59"
PRIMARY_VAL_START = "2022-01-01 00:00:00"
PRIMARY_VAL_END = "2022-06-30 23:59:59"
PRIMARY_TEST_START = "2022-07-01 00:00:00"
PRIMARY_TEST_END = "2022-11-29 23:59:59"

# Five auxiliary 6-month evaluation cutoffs.
# Keep these separate from the primary protocol.
ROLLING_CUTOFFS = [
    "2020-01-01",
    "2020-07-01",
    "2021-01-01",
    "2021-07-01",
    "2022-01-01",
]
ROLLING_HORIZON_MONTHS = 6

CUSTOMER_FILE = "customer_information_clean.csv"
ASSET_FILE = "asset_information_clean.csv"
MARKETS_FILE = "markets_clean.csv"
CLOSE_PRICES_FILE = "close_prices_clean.csv"
LIMIT_PRICES_FILE = "limit_prices_clean.csv"
TRANSACTIONS_FILE = "transactions_clean.csv"

CORE_FILES = {
    "customers": CUSTOMER_FILE,
    "assets": ASSET_FILE,
    "markets": MARKETS_FILE,
    "close_prices": CLOSE_PRICES_FILE,
    "limit_prices": LIMIT_PRICES_FILE,
    "transactions": TRANSACTIONS_FILE,
}
