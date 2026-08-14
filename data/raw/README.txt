FAR-Trans local data directory
==============================

Download FAR-Trans from:
https://doi.org/10.5525/gla.researchdata.1658

Place exactly these six core model files in this directory:

  customer_information.csv
  asset_information.csv
  markets.csv
  close_prices.csv
  limit_prices.csv
  transactions.csv

The files are ignored by Git and are not redistributed by this repository.
questionnaires.csv contains questionnaire text rather than model observations,
so it is outside the six-file Checkpoint 1 pipeline.

Run from the project root:

  python -m src.data.build_dataset

FAR-Trans is published under CC BY 4.0. Cite the dataset/paper as requested by
its authors when using it in research.
