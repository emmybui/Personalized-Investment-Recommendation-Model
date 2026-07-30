"""
far_trans_cleaning.py
======================
Pipeline lam sach + chuan bi du lieu FAR-Trans cho khoa luan
"Xay dung mo hinh khuyen nghi tai san dau tu ca nhan hoa dua tren mang do thi
thoi gian va ho so rui ro cua nha dau tu".

INPUT MONG DOI (dat trong thu muc INPUT_DIR, giu nguyen ten file goc tu bo du lieu):
    - customer_information.csv
    - asset_information.csv
    - markets.csv
    - close_prices.csv
    - limit_prices.csv
    - transactions.csv
(File questionnaires.csv KHONG duoc xu ly o day vi no la van ban cau hoi,
 khong phai bang du lieu dang CSV, va khong can cho pipeline do thi/mo hinh.)

OUTPUT:
    OUTPUT_DIR/cleaned/                      -> ban da lam sach cua ca 6 file gom (dung cho EDA, kiem tra)
    OUTPUT_DIR/split_XX_<ngay_t>/            -> 1 thu muc cho MOI moc thoi gian t trong 5 moc da thong nhat,
                                                 chua du lieu train/test + snapshot "point-in-time" tuong ung

CACH CHAY:
    python far_trans_cleaning.py --input_dir /path/toi/FAR-Trans --output_dir /path/toi/output

Neu chua co du lieu that, co the chay voi --self_test de pipeline tu tao du lieu gia
(dung dinh dang cot) va chay thu toan bo cac buoc, giup ban chac chan code khong loi
truoc khi dung tren du lieu that.
"""

import argparse
import os
import shutil
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. CAU HINH CHUNG
# ---------------------------------------------------------------------------

# 5 moc thoi gian danh gia (Delta t = 6 thang moi moc).
# Ly do chon 5 moc nay (thay vi 61 moc nhu paper goc):
#   - Van phu du cac giai doan thi truong khac nhau (COVID crash 3/2020, chien tranh
#     Nga-Ukraine dau 2022) trong khi giam tai compute rat nhieu cho mot nguoi lam solo.
#   - Warm-up ~2 nam truoc moc dau tien (2018-01-02 -> 2020-01-01) de giam cold-start,
#     vi >50% nha dau tu trong dataset chi co <=3 giao dich trong ca 5 nam.
SPLIT_DATES = [
    "2020-01-01",
    "2020-07-01",
    "2021-01-01",
    "2021-07-01",
    "2022-01-01",
]
DELTA_MONTHS = 6  # do dai cua so test sau moi moc t (khop voi ROI@10 / nDCG@10 cua paper goc)

# Nguong loc: giu lai nha dau tu / tai san co it nhat bao nhieu giao dich trong tap train.
# Dat = 1 nghia la khong loc gat gao (giu ca long-tail), ban co the tang len sau khi EDA
# neu thay mo hinh bi nhieu boi cac nha dau tu qua it du lieu.
MIN_TRAIN_TRANSACTIONS = 1

# Co loc "chi giu nha dau tu active o CA train VA test" giong cach paper goc lam khong.
# True  = giong paper goc, so sanh so lieu cong bo duoc truc tiep.
# False = giu toan bo, phu hop neu ban muon danh gia rieng kha nang cold-start.
FILTER_ACTIVE_IN_BOTH_WINDOWS = True


# ---------------------------------------------------------------------------
# 2. HAM DOC DU LIEU (LOAD) -- ep kieu du lieu ngay tu luc doc de tranh loi ngam
# ---------------------------------------------------------------------------

def load_customer_information(path):
    """Doc customer_information.csv.
    Luu y quan trong (theo README goc): 1 customerID co the xuat hien NHIEU DONG,
    moi dong la 1 lan thong tin duoc cap nhat (khac timestamp). Ham nay CHUA gop
    lai thanh 1 dong/khach hang -- viec gop theo "point-in-time" duoc lam rieng
    trong ham get_customer_snapshot_asof() ben duoi, vi moi moc thoi gian t se
    can 1 phien ban snapshot khac nhau.
    """
    df = pd.read_csv(
        path,
        dtype={
            "customerID": str,
            "customerType": "category",
            "riskLevel": "category",
            "investmentCapacity": "category",
        },
        parse_dates=["lastQuestionnaireDate", "timestamp"],
    )
    return df


def load_asset_information(path):
    """Doc asset_information.csv. Cung co van de nhieu dong/1 ISIN theo thoi gian
    giong customer_information.csv -> xu ly point-in-time o get_asset_snapshot_asof().
    """
    df = pd.read_csv(
        path,
        dtype={
            "ISIN": str,
            "assetName": str,
            "assetShortName": str,
            "assetCategory": "category",   # Stock / Bond / MTF
            "assetSubCategory": "category",
            "marketID": str,
            "sector": "category",
            "industry": "category",
        },
        parse_dates=["timestamp"],
    )
    return df


def load_markets(path):
    """Doc markets.csv. Day la bang tham chieu tinh (khong co timestamp cap nhat
    theo README), nen khong can xu ly point-in-time."""
    df = pd.read_csv(
        path,
        dtype={
            "exchangeID": str,
            "marketID": str,
            "name": str,
            "description": str,
            "country": "category",
            "tradingDays": str,
            "tradingHours": str,
            "marketClass": "category",
        },
    )
    return df


def load_close_prices(path):
    """Doc close_prices.csv (ISIN, timestamp, closePrice)."""
    df = pd.read_csv(
        path,
        dtype={"ISIN": str, "closePrice": float},
        parse_dates=["timestamp"],
    )
    return df


def load_limit_prices(path):
    """Doc limit_prices.csv (thong ke gia dau/cuoi chuoi thoi gian cho tung tai san)."""
    df = pd.read_csv(
        path,
        dtype={"ISIN": str, "priceMinDate": float, "priceMaxDate": float, "profitability": float},
        parse_dates=["minDate", "maxDate"],
    )
    return df


def load_transactions(path):
    """Doc transactions.csv (nhat ky giao dich mua/ban)."""
    df = pd.read_csv(
        path,
        dtype={
            "customerID": str,
            "ISIN": str,
            "transactionID": str,
            "transactionType": "category",  # Buy / Sell
            "totalValue": float,
            "units": float,
            "channel": "category",
            "marketID": str,
        },
        parse_dates=["timestamp"],
    )
    return df


# ---------------------------------------------------------------------------
# 3. HAM LAM SACH (CLEANING) THEO TUNG BANG
#    Luu y: paper goc FAR-Trans da tu lam sach phan lon van de nghiem trong
#    (trung gia, stock split, outlier gia...) truoc khi cong bo dataset, nen
#    o day chi lam them 3 viec: (a) chuan hoa nhan category cho de dung trong
#    model, (b) loai cac dong ro rang hong/thieu key, (c) kiem tra lai (sanity
#    check) de phat hien neu ban tai ve mot ban da bi loi/thieu.
# ---------------------------------------------------------------------------

def clean_customer_information(df):
    """Lam sach customer_information.csv.

    Cac buoc:
    1. Bo dong trung lap hoan toan.
    2. Bo dong thieu customerID (khong the dung neu khong co key).
    3. Chuan hoa riskLevel: gop "Predicted_Aggressive" -> "Aggressive", v.v.
       Ly do gop: nhan Predicted_* va nhan that (do khach hang tu tra loi bang
       hoi MiFID) VE MAT Y NGHIA la cung 1 gia tri rui ro, chi khac o cach co
       duoc. Neu de rieng, model se coi "Aggressive" va "Predicted_Aggressive"
       la 2 category khac nhau -> lam loang du lieu mot cach khong can thiet.
       Ta giu lai thong tin "co phai la du doan hay khong" bang 1 cot rieng
       (is_risk_predicted) de model van co the hoc duoc do tin cay cua nhan
       neu can.
    4. Tuong tu cho investmentCapacity.
    5. Gan "Not_Available" -> "Unknown" (mot category rieng, KHONG phai NaN,
       vi "khong co du lieu" cung la 1 tin hieu ma model co the hoc duoc,
       khac voi de trong/NaN de rot xuong buoc imputation ngoai y muon).
    """
    df = df.drop_duplicates().copy()
    n_before = len(df)
    df = df.dropna(subset=["customerID"])
    if len(df) < n_before:
        warnings.warn(f"[customer_information] Da bo {n_before - len(df)} dong thieu customerID.")

    # --- chuan hoa riskLevel ---
    df["is_risk_predicted"] = df["riskLevel"].astype(str).str.startswith("Predicted_")
    df["riskLevel"] = (
        df["riskLevel"].astype(str).str.replace("Predicted_", "", regex=False)
    )
    df["riskLevel"] = df["riskLevel"].replace({"Not_Available": "Unknown"})
    df["riskLevel"] = df["riskLevel"].astype("category")

    # --- chuan hoa investmentCapacity ---
    # Luu y nho: theo README, gia tri du doan cho capacity la "Predicted_GT300K"
    # (thieu chu "CAP_" so voi cac muc con lai) -- xu ly rieng truong hop nay.
    df["is_capacity_predicted"] = df["investmentCapacity"].astype(str).str.startswith("Predicted_")
    cap = df["investmentCapacity"].astype(str).str.replace("Predicted_", "", regex=False)
    cap = cap.replace({"GT300K": "CAP_GT300K"})  # dua ve dung format voi nhan khong-du-doan
    cap = cap.replace({"Not_Available": "Unknown"})
    df["investmentCapacity"] = cap.astype("category")

    df = df.sort_values(["customerID", "timestamp"]).reset_index(drop=True)
    return df


def clean_asset_information(df):
    """Lam sach asset_information.csv.

    Cac buoc:
    1. Bo dong trung lap hoan toan.
    2. Bo dong thieu ISIN hoac assetCategory (khong xac dinh duoc loai tai san
       thi khong dua vao do thi duoc).
    3. Dien "Unknown" cho sector/industry con thieu (nhieu tai san khong co
       nganh theo README -- "when available") thay vi de NaN, vi day se la
       mot node "Sector" rieng trong do thi (Unknown cung la 1 nhom hop le).
    4. Sap xep theo ISIN, timestamp de phuc vu buoc snapshot point-in-time.
    """
    df = df.drop_duplicates().copy()
    n_before = len(df)
    df = df.dropna(subset=["ISIN", "assetCategory"])
    if len(df) < n_before:
        warnings.warn(f"[asset_information] Da bo {n_before - len(df)} dong thieu ISIN/assetCategory.")

    df["sector"] = df["sector"].astype(str).replace({"nan": "Unknown"}).astype("category")
    df["industry"] = df["industry"].astype(str).replace({"nan": "Unknown"}).astype("category")

    df = df.sort_values(["ISIN", "timestamp"]).reset_index(drop=True)
    return df


def clean_markets(df):
    """Lam sach markets.csv. Day la bang tinh, chi can bo trung va kiem tra khoa chinh."""
    df = df.drop_duplicates().copy()
    dup_ids = df["marketID"].duplicated().sum()
    if dup_ids > 0:
        warnings.warn(
            f"[markets] Phat hien {dup_ids} marketID bi lap voi thong tin khac nhau -- "
            f"can kiem tra thu cong, hien tai script GIU LAI dong xuat hien dau tien."
        )
        df = df.drop_duplicates(subset=["marketID"], keep="first")
    return df.reset_index(drop=True)


def clean_close_prices(df):
    """Lam sach close_prices.csv.

    Paper goc da xu ly hau het loi ve gia (trung gia, stock split, outlier),
    nhung van nen kiem tra lai vi day la input truc tiep cho ca 2 nhanh model
    (TCN market feature + tinh ROI/Sharpe/Max Drawdown danh gia sau nay):
    1. Bo dong gia <= 0 (khong hop le ve mat tai chinh) -- canh bao neu gap,
       vi ly thuyet la khong con sau buoc lam sach cua paper.
    2. Bo dong trung hoan toan (ISIN, timestamp, closePrice giong het nhau).
    3. Neu van con nhieu gia cho CUNG 1 cap (ISIN, timestamp) (khong nen co),
       giu dong CUOI CUNG trong file va canh bao ro rang -- KHONG tu lay
       trung binh vi co the che giau loi du lieu that su.
    4. Sap xep theo ISIN, timestamp (bat buoc de tinh return/technical
       indicator dung thu tu sau nay).
    """
    df = df.drop_duplicates().copy()

    n_before = len(df)
    df = df[df["closePrice"] > 0]
    if len(df) < n_before:
        warnings.warn(f"[close_prices] Da bo {n_before - len(df)} dong co gia <= 0.")

    dup_key = df.duplicated(subset=["ISIN", "timestamp"], keep=False)
    if dup_key.sum() > 0:
        warnings.warn(
            f"[close_prices] Phat hien {dup_key.sum()} dong co cung (ISIN, timestamp) "
            f"nhung gia khac nhau -- giu dong CUOI CUNG, ban NEN kiem tra lai thu cong."
        )
        df = df.drop_duplicates(subset=["ISIN", "timestamp"], keep="last")

    df = df.sort_values(["ISIN", "timestamp"]).reset_index(drop=True)
    return df


def clean_limit_prices(df):
    """Lam sach limit_prices.csv. Bo dong minDate > maxDate (khong hop le)."""
    df = df.drop_duplicates().copy()
    n_before = len(df)
    df = df[df["minDate"] <= df["maxDate"]]
    if len(df) < n_before:
        warnings.warn(f"[limit_prices] Da bo {n_before - len(df)} dong co minDate > maxDate.")
    return df.reset_index(drop=True)


def clean_transactions(df):
    """Lam sach transactions.csv.

    Theo dung cach paper goc mo ta da lam voi du lieu tho cua ho (Section 3.3.1):
    1. Bo giao dich thieu customerID (khong gan duoc voi ai).
    2. Chi giu transactionType la "Buy" hoac "Sell" (loai gia tri la).
    3. Bo giao dich units <= 0 hoac totalValue < 0 (khong hop le).
    4. Bo transactionID bi trung (neu co, coi la loi ghi log 2 lan).
    5. Sap xep theo timestamp tang dan -- QUAN TRONG vi day se la thu tu
       "chuoi su kien" (event stream) dua thang vao TGN.
    """
    df = df.drop_duplicates().copy()

    n_before = len(df)
    df = df.dropna(subset=["customerID"])
    if len(df) < n_before:
        warnings.warn(f"[transactions] Da bo {n_before - len(df)} dong thieu customerID.")

    n_before = len(df)
    df = df[df["transactionType"].isin(["Buy", "Sell"])]
    if len(df) < n_before:
        warnings.warn(f"[transactions] Da bo {n_before - len(df)} dong co transactionType khong hop le.")

    n_before = len(df)
    df = df[(df["units"] > 0) & (df["totalValue"] >= 0)]
    if len(df) < n_before:
        warnings.warn(f"[transactions] Da bo {n_before - len(df)} dong co units<=0 hoac totalValue<0.")

    n_before = len(df)
    df = df.drop_duplicates(subset=["transactionID"], keep="first")
    if len(df) < n_before:
        warnings.warn(f"[transactions] Da bo {n_before - len(df)} dong transactionID bi trung.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 4. HAM SNAPSHOT "POINT-IN-TIME"
#    Day la phan xu ly truc tiep cho van de ban hoi: customer_information.csv
#    va asset_information.csv co THE co nhieu dong / 1 ID (theo cac lan cap
#    nhat khac nhau). Neu xay dung feature cho node tai moc thoi gian t ma
#    LO lay nham 1 dong co timestamp SAU t, thi model se "nhin thay tuong
#    lai" (data leakage) -- vi du: biet truoc mot khach hang sap chuyen tu
#    "Conservative" sang "Aggressive" 2 thang sau moc du doan.
#
#    Nguyen tac chung: voi moi ID, chi lay dong CO TIMESTAMP GAN NHAT nhung
#    KHONG VUOT QUA moc as_of_date.
# ---------------------------------------------------------------------------

def get_customer_snapshot_asof(customer_df, as_of_date):
    """Voi moi customerID, lay dong thong tin moi nhat tinh den truoc as_of_date.

    Neu 1 khach hang chua co ban ghi nao truoc as_of_date (vi du: khach hang
    moi tham gia sau moc t), khach hang do se KHONG xuat hien trong snapshot
    nay -- day la hanh vi dung, vi tai moc t ta chua he biet gi ve ho ca.
    """
    as_of_date = pd.Timestamp(as_of_date)
    mask = customer_df["timestamp"] <= as_of_date
    valid = customer_df.loc[mask]
    # sort roi lay dong CUOI CUNG cho moi customerID = dong co timestamp lon nhat
    # nhung van <= as_of_date (nho da loc o buoc mask ben tren).
    snapshot = (
        valid.sort_values(["customerID", "timestamp"])
        .drop_duplicates(subset=["customerID"], keep="last")
        .reset_index(drop=True)
    )
    snapshot["snapshot_asof_date"] = as_of_date
    return snapshot


def get_asset_snapshot_asof(asset_df, as_of_date):
    """Tuong tu get_customer_snapshot_asof() nhung cho tai san (theo ISIN)."""
    as_of_date = pd.Timestamp(as_of_date)
    mask = asset_df["timestamp"] <= as_of_date
    valid = asset_df.loc[mask]
    snapshot = (
        valid.sort_values(["ISIN", "timestamp"])
        .drop_duplicates(subset=["ISIN"], keep="last")
        .reset_index(drop=True)
    )
    snapshot["snapshot_asof_date"] = as_of_date
    return snapshot


# ---------------------------------------------------------------------------
# 5. XAY DUNG TUNG SPLIT TRAIN/TEST THEO MOC THOI GIAN t
# ---------------------------------------------------------------------------

def build_split(
    transactions_df,
    customer_df,
    asset_df,
    close_prices_df,
    t,
    delta_months=DELTA_MONTHS,
    min_train_transactions=MIN_TRAIN_TRANSACTIONS,
    filter_active_in_both_windows=FILTER_ACTIVE_IN_BOTH_WINDOWS,
):
    """Xay dung 1 bo du lieu train/test tai moc thoi gian t.

    Cac buoc (theo dung thu tu de tranh leakage o BAT KY buoc nao):
    1. train = tat ca giao dich co timestamp <= t
       test  = giao dich co t < timestamp <= t + delta_months
    2. Neu 1 cap (customerID, ISIN) xuat hien o CA train va test, chi giu lai
       o train (giong cach paper goc xu ly, muc 5.2.1) -- vi neu khach hang
       da tung mua tai san do truoc t roi, viec ho "mua lai" trong test khong
       con la mot khuyen nghi moi/huu ich de danh gia nua.
    3. (Tuy chon, bat theo filter_active_in_both_windows) Chi giu khach hang
       co it nhat 1 giao dich trong CA train VA test -- day la cach paper
       goc lam de ROI-based va nDCG-based metric duoc tinh tren cung 1 tap
       khach hang/tai san, moi so sanh cong bang. Neu tat co nay, ban se
       giu duoc ca cac khach hang "cold-start" trong test.
    4. Snapshot ho so khach hang & thong tin tai san tinh DEN TRUOC t (dung
       2 ham point-in-time o tren) -- KHONG duoc dung thong tin sau t.
    5. Cat chuoi gia (close_prices) chi giu <= t -- de moi chi bao ky thuat
       (RSI, MACD, volatility...) tinh sau nay chi dung du lieu qua khu,
       khong "nhin thay" gia trong tuong lai.
    """
    t = pd.Timestamp(t)
    test_end = t + pd.DateOffset(months=delta_months)

    train_txn = transactions_df[transactions_df["timestamp"] <= t].copy()
    test_txn = transactions_df[
        (transactions_df["timestamp"] > t) & (transactions_df["timestamp"] <= test_end)
    ].copy()

    # --- buoc 2: bo cap (customer, asset) bi lap giua train va test ---
    # Dung 1 cot key gop (vectorized) thay vi apply() theo tung dong -- transactions.csv
    # co ~388K dong nen apply() se rat cham, con ghep chuoi + isin() thi nhanh hon nhieu.
    train_txn["_pair_key"] = train_txn["customerID"].astype(str) + "||" + train_txn["ISIN"].astype(str)
    test_txn["_pair_key"] = test_txn["customerID"].astype(str) + "||" + test_txn["ISIN"].astype(str)
    train_pairs = set(train_txn["_pair_key"])
    before = len(test_txn)
    test_txn = test_txn[~test_txn["_pair_key"].isin(train_pairs)]
    removed_overlap = before - len(test_txn)
    train_txn = train_txn.drop(columns="_pair_key")
    test_txn = test_txn.drop(columns="_pair_key")

    # --- buoc 3 (tuy chon): loc theo hoat dong toi thieu ---
    train_counts = train_txn["customerID"].value_counts()
    eligible_customers = set(train_counts[train_counts >= min_train_transactions].index)

    if filter_active_in_both_windows:
        test_customers = set(test_txn["customerID"].unique())
        eligible_customers &= test_customers

    train_txn = train_txn[train_txn["customerID"].isin(eligible_customers)]
    test_txn = test_txn[test_txn["customerID"].isin(eligible_customers)]

    # Tai san: chi giu tai san co gia du lieu it nhat den het cua so test
    # (khong loai bo tai san chua tung giao dich -- do la nhom "cold-start"
    # can cho de danh gia kha nang tan dung node Market/Sector cua do thi).
    assets_with_price_until_test_end = set(
        close_prices_df.loc[close_prices_df["timestamp"] <= test_end, "ISIN"].unique()
    )

    # --- buoc 4: snapshot ho so tai t ---
    customer_snapshot = get_customer_snapshot_asof(customer_df, t)
    asset_snapshot = get_asset_snapshot_asof(asset_df, t)
    asset_snapshot = asset_snapshot[asset_snapshot["ISIN"].isin(assets_with_price_until_test_end)]

    # --- buoc 5: cat chuoi gia ---
    prices_upto_t = close_prices_df[close_prices_df["timestamp"] <= t].copy()

    summary = {
        "t": t,
        "test_end": test_end,
        "n_train_transactions": len(train_txn),
        "n_test_transactions": len(test_txn),
        "n_customers_train": train_txn["customerID"].nunique(),
        "n_customers_test": test_txn["customerID"].nunique(),
        "n_assets_train": train_txn["ISIN"].nunique(),
        "n_assets_test": test_txn["ISIN"].nunique(),
        "n_train_test_pairs_removed_as_overlap": removed_overlap,
    }

    return {
        "train_transactions": train_txn.reset_index(drop=True),
        "test_transactions": test_txn.reset_index(drop=True),
        "customer_snapshot": customer_snapshot,
        "asset_snapshot": asset_snapshot,
        "prices_upto_t": prices_upto_t.reset_index(drop=True),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 6. PIPELINE CHINH: doc -> lam sach -> luu ban sach -> tao 5 split -> luu
# ---------------------------------------------------------------------------

def run_pipeline(input_dir, output_dir):
    """Chay toan bo pipeline: doc 6 file goc, lam sach, luu ban sach, roi
    tao va luu 5 bo train/test theo cac moc thoi gian trong SPLIT_DATES.
    """
    print(f"\n[1/3] Dang doc du lieu tu: {input_dir}")
    customer_df = load_customer_information(os.path.join(input_dir, "customer_information.csv"))
    asset_df = load_asset_information(os.path.join(input_dir, "asset_information.csv"))
    markets_df = load_markets(os.path.join(input_dir, "markets.csv"))
    prices_df = load_close_prices(os.path.join(input_dir, "close_prices.csv"))
    limit_prices_df = load_limit_prices(os.path.join(input_dir, "limit_prices.csv"))
    transactions_df = load_transactions(os.path.join(input_dir, "transactions.csv"))
    print(
        f"    customer_information: {len(customer_df)} dong | "
        f"asset_information: {len(asset_df)} dong | "
        f"markets: {len(markets_df)} dong"
    )
    print(
        f"    close_prices: {len(prices_df)} dong | "
        f"limit_prices: {len(limit_prices_df)} dong | "
        f"transactions: {len(transactions_df)} dong"
    )

    print("\n[2/3] Dang lam sach du lieu...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        customer_clean = clean_customer_information(customer_df)
        asset_clean = clean_asset_information(asset_df)
        markets_clean = clean_markets(markets_df)
        prices_clean = clean_close_prices(prices_df)
        limit_prices_clean = clean_limit_prices(limit_prices_df)
        transactions_clean = clean_transactions(transactions_df)
        for w in caught:
            print(f"    [canh bao] {w.message}")

    cleaned_dir = os.path.join(output_dir, "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)
    customer_clean.to_csv(os.path.join(cleaned_dir, "customer_information_clean.csv"), index=False)
    asset_clean.to_csv(os.path.join(cleaned_dir, "asset_information_clean.csv"), index=False)
    markets_clean.to_csv(os.path.join(cleaned_dir, "markets_clean.csv"), index=False)
    prices_clean.to_csv(os.path.join(cleaned_dir, "close_prices_clean.csv"), index=False)
    limit_prices_clean.to_csv(os.path.join(cleaned_dir, "limit_prices_clean.csv"), index=False)
    transactions_clean.to_csv(os.path.join(cleaned_dir, "transactions_clean.csv"), index=False)
    print(f"    Da luu ban sach vao: {cleaned_dir}")

    print(f"\n[3/3] Dang tao {len(SPLIT_DATES)} bo train/test theo cac moc thoi gian...")
    all_summaries = []
    for i, t in enumerate(SPLIT_DATES, start=1):
        split = build_split(
            transactions_clean, customer_clean, asset_clean, prices_clean, t
        )
        split_dir = os.path.join(output_dir, f"split_{i:02d}_{t}")
        os.makedirs(split_dir, exist_ok=True)
        split["train_transactions"].to_csv(os.path.join(split_dir, "train_transactions.csv"), index=False)
        split["test_transactions"].to_csv(os.path.join(split_dir, "test_transactions.csv"), index=False)
        split["customer_snapshot"].to_csv(os.path.join(split_dir, "customer_snapshot.csv"), index=False)
        split["asset_snapshot"].to_csv(os.path.join(split_dir, "asset_snapshot.csv"), index=False)
        split["prices_upto_t"].to_csv(os.path.join(split_dir, "prices_upto_t.csv"), index=False)

        s = split["summary"]
        print(
            f"    split {i} (t={t}): train={s['n_train_transactions']} giao dich "
            f"({s['n_customers_train']} khach, {s['n_assets_train']} tai san) | "
            f"test={s['n_test_transactions']} giao dich "
            f"({s['n_customers_test']} khach, {s['n_assets_test']} tai san) | "
            f"da bo {s['n_train_test_pairs_removed_as_overlap']} cap trung train/test"
        )
        all_summaries.append(s)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(output_dir, "split_summary.csv"), index=False)
    print(f"\nHoan tat. Bang tom tat cac split: {os.path.join(output_dir, 'split_summary.csv')}")
    return summary_df


# ---------------------------------------------------------------------------
# 7. SINH DU LIEU GIA (SELF-TEST) -- CHI DE KIEM TRA PIPELINE CHAY DUNG,
#    KHONG PHAI DU LIEU THAT. Dung khi ban muon chay thu script truoc khi
#    co file that trong tay, hoac de debug nhanh.
# ---------------------------------------------------------------------------

def generate_synthetic_data(target_dir, n_customers=200, n_assets=30, n_transactions=3000, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(target_dir, exist_ok=True)

    date_range = pd.date_range("2018-01-02", "2022-11-29", freq="D")

    # customer_information.csv -- co CO Y moi customer xuat hien 1-3 lan
    # (nhieu ban ghi theo thoi gian) de test dung logic point-in-time.
    risk_levels = ["Conservative", "Income", "Balanced", "Aggressive",
                    "Predicted_Conservative", "Predicted_Income",
                    "Predicted_Balanced", "Predicted_Aggressive", "Not_Available"]
    caps = ["CAP_LT_30K", "CAP_30K_80K", "CAP_80K_300K", "CAP_GT300K",
            "Predicted_CAP_LT_30K", "Predicted_CAP_30K_80K",
            "Predicted_CAP_80K_300K", "Predicted_GT300K", "Not_Available"]
    cust_types = ["Mass", "Premium", "Professional", "Legal entity", "Inactive"]

    cust_rows = []
    for c in range(n_customers):
        cid = f"CUST{c:05d}"
        n_updates = rng.integers(1, 4)
        update_dates = sorted(rng.choice(date_range, size=n_updates, replace=False))
        for d in update_dates:
            cust_rows.append({
                "customerID": cid,
                "customerType": rng.choice(cust_types),
                "riskLevel": rng.choice(risk_levels),
                "investmentCapacity": rng.choice(caps),
                "lastQuestionnaireDate": d,
                "timestamp": d,
            })
    pd.DataFrame(cust_rows).to_csv(os.path.join(target_dir, "customer_information.csv"), index=False)

    # asset_information.csv
    categories = ["Stock", "Bond", "MTF"]
    isins = [f"ISIN{a:04d}" for a in range(n_assets)]
    asset_rows = []
    for isin in isins:
        n_updates = rng.integers(1, 3)
        update_dates = sorted(rng.choice(date_range, size=n_updates, replace=False))
        for d in update_dates:
            asset_rows.append({
                "ISIN": isin,
                "assetName": f"Asset {isin}",
                "assetShortName": isin,
                "assetCategory": rng.choice(categories),
                "assetSubCategory": "SubcatA",
                "marketID": f"MKT{rng.integers(0, 5):02d}",
                "sector": rng.choice(["Tech", "Finance", "Energy", None]),
                "industry": rng.choice(["Software", "Banking", "Oil", None]),
                "timestamp": d,
            })
    pd.DataFrame(asset_rows).to_csv(os.path.join(target_dir, "asset_information.csv"), index=False)

    # markets.csv
    market_rows = [{
        "exchangeID": f"EXC{m:02d}", "marketID": f"MKT{m:02d}", "name": f"Market {m}",
        "description": "desc", "country": "GR", "tradingDays": "Mon,Tue,Wed,Thu,Fri",
        "tradingHours": "09:00-17:00", "marketClass": "Regulated",
    } for m in range(5)]
    pd.DataFrame(market_rows).to_csv(os.path.join(target_dir, "markets.csv"), index=False)

    # close_prices.csv -- random walk gia cho moi ISIN
    price_rows = []
    for isin in isins:
        price = rng.uniform(10, 100)
        for d in date_range[::5]:  # gia moi 5 ngay cho gon, du de test
            price *= (1 + rng.normal(0, 0.02))
            price = max(price, 0.5)
            price_rows.append({"ISIN": isin, "timestamp": d, "closePrice": round(price, 4)})
    pd.DataFrame(price_rows).to_csv(os.path.join(target_dir, "close_prices.csv"), index=False)

    # limit_prices.csv
    limit_rows = []
    for isin in isins:
        sub = [r for r in price_rows if r["ISIN"] == isin]
        limit_rows.append({
            "ISIN": isin,
            "minDate": sub[0]["timestamp"], "maxDate": sub[-1]["timestamp"],
            "priceMinDate": sub[0]["closePrice"], "priceMaxDate": sub[-1]["closePrice"],
            "profitability": sub[-1]["closePrice"] / sub[0]["closePrice"] - 1,
        })
    pd.DataFrame(limit_rows).to_csv(os.path.join(target_dir, "limit_prices.csv"), index=False)

    # transactions.csv
    cust_ids = [f"CUST{c:05d}" for c in range(n_customers)]
    txn_dates = pd.date_range("2018-01-02", "2022-11-29", freq="D")
    txn_rows = []
    for i in range(n_transactions):
        txn_rows.append({
            "customerID": rng.choice(cust_ids),
            "ISIN": rng.choice(isins),
            "transactionID": f"TXN{i:06d}",
            "transactionType": rng.choice(["Buy", "Sell"]),
            "timestamp": rng.choice(txn_dates),
            "totalValue": round(rng.uniform(100, 5000), 2),
            "units": round(rng.uniform(1, 100), 2),
            "channel": rng.choice(["Internet Banking", "Phone Banking", "Branch"]),
            "marketID": f"MKT{rng.integers(0, 5):02d}",
        })
    pd.DataFrame(txn_rows).to_csv(os.path.join(target_dir, "transactions.csv"), index=False)

    print(f"Da sinh du lieu gia (self-test) vao: {target_dir}")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lam sach + chia split FAR-Trans")
    parser.add_argument("--input_dir", type=str, default="./FAR-Trans",
                         help="Thu muc chua 6 file CSV goc cua FAR-Trans")
    parser.add_argument("--output_dir", type=str, default="./FAR-Trans-processed",
                         help="Thu muc se ghi ket qua ra")
    parser.add_argument("--self_test", action="store_true",
                         help="Sinh du lieu gia va chay thu toan bo pipeline (khong can du lieu that)")
    args = parser.parse_args()

    if args.self_test:
        synth_dir = "./_far_trans_synthetic"
        if os.path.exists(synth_dir):
            shutil.rmtree(synth_dir)
        generate_synthetic_data(synth_dir)
        run_pipeline(synth_dir, args.output_dir)
    else:
        run_pipeline(args.input_dir, args.output_dir)
