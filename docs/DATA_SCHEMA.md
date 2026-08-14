# FAR-Trans schema used by RATGR

Core data files are the six files named in the thesis proposal:

1. customer_information
2. asset_information
3. markets
4. close_prices
5. limit_prices
6. transactions

## Keys

### customer_information
- Entity key: `customerID`
- Temporal key: (`customerID`, `timestamp`)
- Risk attributes: `riskLevel`, `investmentCapacity`
- Other profile fields: `customerType`, `lastQuestionnaireDate`

### asset_information
- Entity key: `ISIN`
- Temporal key: (`ISIN`, `timestamp`)
- Attributes: category/subcategory, market, sector, industry

### markets
- Reference key: `marketID`
- Exchange key: `exchangeID`
- This is treated as reference data in the current cleaning code.

### close_prices
- Temporal key: (`ISIN`, `timestamp`)
- Numeric signal: `closePrice`

### limit_prices
- Asset key: `ISIN`
- Date range: `minDate`, `maxDate`
- `profitability` must NOT be used as a model input at an as-of time unless it is recomputed from information available before that time.

### transactions
- Event key: (`customerID`, `transactionID`)
- Investor key: `customerID`
- Asset key: `ISIN`
- Event type: `transactionType` (`Buy` / `Sell`)
- Event time: `timestamp`
- Numeric fields: `totalValue`, `units`
- Context: `channel`, `marketID`

## Important invariant

For customer and asset snapshots, there must not be multiple conflicting rows
for the same (`ID`, `timestamp`). If such rows exist, the pipeline must stop rather
than silently choosing one.

`transactionID` is not globally unique in FAR-Trans. It is reused across
customers, while (`customerID`, `transactionID`) is unique. Global
transactionID de-duplication is therefore data loss, not cleaning.
