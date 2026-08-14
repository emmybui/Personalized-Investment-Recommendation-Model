# Data Protocol — RATGR Thesis

## Dataset

- **Name**: FAR-Trans (Financial Asset Recommendation Transactions)
- **Transaction date range**: 2018-01-02 to 2022-11-30
- **Source**: 6 CSV files (customer_information, asset_information, markets, close_prices, limit_prices, transactions)

## Pipeline Architecture

```
raw/                        (6 original CSVs)
  |
  v
clean.py                    (cleaning ONLY)
  |
  v
processed/                  (6 cleaned CSVs)
  |
  v
python -m src.data.build_dataset
  |
  v
splits/                     (primary + rolling + holdings + ID mapping)
reports/                    (data_quality_report.json)
```

> **Rule**: `clean.py` does NOT create splits.  `build_dataset.py` does NOT clean.

## Primary Protocol

| Phase | Date Range | Purpose |
|-------|-----------|---------|
| **Train** | 2018-01-02 — 2021-12-31 | Model training |
| **Validation** | 2022-01-01 — 2022-06-30 | Hyperparameter tuning |
| **Test** | 2022-07-01 — 2022-11-30 | Final evaluation (open LAST) |

## Auxiliary Rolling Protocol

5 windows × 6 months each.  These are robustness evaluations, NOT replacements for the primary protocol.

| Window | Cutoff | Test End |
|--------|--------|----------|
| 1 | 2020-01-01 | 2020-07-01 |
| 2 | 2020-07-01 | 2021-01-01 |
| 3 | 2021-01-01 | 2021-07-01 |
| 4 | 2021-07-01 | 2022-01-01 |
| 5 | 2022-01-01 | 2022-07-01 |

## Point-in-Time Rule

Features at time `t` must use **only** information available at or before `t`.

- Customer snapshot at `t`: most recent record per customerID where `timestamp <= t`
- Asset snapshot at `t`: most recent record per ISIN where `timestamp <= t`
- Price features at `t`: close_prices where `timestamp <= t`
- Holdings at `t`: net(BUY - SELL) units where transaction `timestamp <= t`

## Candidate Rule & Point-in-Time Candidate Universe

The candidate pool at cutoff `t` is strictly bounded by point-in-time asset availability to prevent future-universe leakage:

```
C(u, t) = A(t) - H(u, t)
```

Where:
- `A(t)`: Eligible/known assets at time `t` (derived from `asset_snapshot` at time `t`). Assets created in future periods after `t` are excluded from `A(t)`.
- `H(u, t)`: Assets currently held by customer `u` at time `t` (`net_units > 0`).
- `BUY` → `+units`
- `SELL` → `-units`
- `net_units > 0` → currently held (excluded from recommendations)
- `net_units <= 0` → not currently held (eligible candidate)

> **Important**: "Currently held" is NOT the same as "ever interacted".
> Example: BUY A in 2020, SELL A in 2021 → At cutoff 2021-07-01, net_units = 0 → A is NOT currently held → A IS an eligible candidate.

### Negative Net Holdings Policy (SELL before BUY or missing history)

Negative net holdings (`net_units < 0`) are treated as `currently_held = False` (`net_units <= 0`). No short-position recommendation is modeled. Such cases are logged as data-quality statistics during pipeline execution.

## Transaction Ordering

```
ORDER BY timestamp ASC, customerID ASC, transactionID ASC, ISIN ASC  (stable sort)
```

This is critical for TGN event stream determinism.

## Duplicate Policy

| Type | Action |
|------|--------|
| Exact duplicate (all columns identical) | Remove automatically |
| Conflicting temporal key (same key, different content) | **ERROR** — pipeline stops |
| Exception: 3 known FAR-Trans asset dual-classifications | Whitelisted, prefer MTF deterministically |

For transactions the logical key is (`customerID`, `transactionID`). FAR-Trans
reuses `transactionID` across customers; de-duplicating it globally removes
valid events.

## Test Policy

- Test data is NEVER used during training or hyperparameter tuning.
- Flow: Train model → tune on Validation → freeze → evaluate on Test.
- Validation predictions use features available at `train_end`.
- Test predictions use features available at `validation_end`.

## Pair-Overlap Policy

| Mode | Behavior |
|------|----------|
| **Thesis mode** (default) | No pair-overlap removal. Use holdings-based candidate exclusion instead. |
| **Benchmark mode** (optional) | Remove (customer, asset) pairs that appear in both train and test, for comparability with FAR-Trans paper. |

## Graph Event Schema (Pu ↔ Meo)

```
GraphEvent:
  src_customer: customerID (string)
  dst_asset: ISIN (string)
  timestamp: datetime
  event_type: "Buy" | "Sell"
  units: float
  totalValue: float
  marketID: string

Edge semantics:
  Buy  → customer --[BUY]--> asset
  Sell → customer --[SELL]--> asset
```

## Shared ID Mapping

All models (Popularity, BPR, LightGCN, TGN) use the same Train-fitted integer mapping:

```python
from src.data.loader import fit_id_mapping

mapping = fit_id_mapping(train_transactions)
```

The mapping is saved at `data/splits/id_mapping.json` during pipeline execution.

> **ID Mapping Rule**: the default mapping is fit on Train only. Validation/Test
> IDs outside that mapping are cold-start cases and must be reported explicitly;
> they must not be silently added to a full-data mapping. Original string IDs
> are preserved in generated tables.

## limit_prices

`profitability` in limit_prices must **NOT** be used as a model input at time `t` unless it is recomputed from information available before `t`. The pipeline loads limit_prices for reference but does not include it in split features.
