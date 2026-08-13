"""
far_trans_cleaning.py
======================
Pipeline lam sach + chuan bi du lieu FAR-Trans
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

# Nguong loc: giu lai nha dau tu / tai san co it nhat bao nhieu giao dich trong tap train.
# Dat = 1 nghia la khong loc gat gao (giu ca long-tail), ban co the tang len sau khi EDA
# neu thay mo hinh bi nhieu boi cac nha dau tu qua it du lieu.
MIN_TRAIN_TRANSACTIONS = 1

# Co loc "chi giu nha dau tu active o CA train VA test" giong cach paper goc lam khong.
# True  = giong paper goc, so sanh so lieu cong bo duoc truc tiep.
# False = giu toan bo -- MAC DINH cho thesis primary protocol.
# RED 03 fix: True dung thong tin tuong lai (ai co giao dich trong test),
# chi bat True trong benchmark-reproduction mode.
FILTER_ACTIVE_IN_BOTH_WINDOWS = False


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
    """Lam sach markets.csv. Day la bang tinh, chi can bo trung va kiem tra khoa chinh.

    RED 07 fix: phan biet exact duplicate (xoa an toan) va conflicting
    duplicate (cung marketID nhung noi dung khac -- bao loi de kiem tra thu cong).
    """
    n_before = len(df)
    df = df.drop_duplicates().copy()  # exact duplicates removed
    n_exact = n_before - len(df)
    if n_exact > 0:
        warnings.warn(f"[markets] Da bo {n_exact} dong trung lap hoan toan (exact duplicate).")

    dup_ids = df["marketID"].duplicated(keep=False)
    if dup_ids.any():
        conflict_df = df.loc[dup_ids].sort_values("marketID")
        n_conflict = len(conflict_df)
        sample = conflict_df["marketID"].unique()[:5].tolist()
        warnings.warn(
            f"[markets] CONFLICTING: {n_conflict} dong co cung marketID nhung noi dung "
            f"khac nhau (KHONG phai exact duplicate). marketID mau: {sample}. "
            f"Hien tai GIU dong DAU TIEN -- can kiem tra thu cong."
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
    # RED 06 fix: phan biet exact duplicate vs conflicting duplicate.
    n_before = len(df)
    df = df.drop_duplicates().copy()  # exact duplicates (tat ca cot giong) -> xoa an toan
    n_exact = n_before - len(df)
    if n_exact > 0:
        warnings.warn(f"[close_prices] Da bo {n_exact} dong trung lap hoan toan (exact duplicate).")

    n_before = len(df)
    df = df[df["closePrice"] > 0]
    if len(df) < n_before:
        warnings.warn(f"[close_prices] Da bo {n_before - len(df)} dong co gia <= 0.")

    # Sau khi bo exact duplicate, neu con trung (ISIN, timestamp) thi CHAC CHAN gia khac nhau.
    dup_key = df.duplicated(subset=["ISIN", "timestamp"], keep=False)
    if dup_key.sum() > 0:
        conflict_df = df.loc[dup_key].sort_values(["ISIN", "timestamp"])
        n_pairs = conflict_df[["ISIN", "timestamp"]].drop_duplicates().shape[0]
        warnings.warn(
            f"[close_prices] CONFLICTING: {dup_key.sum()} dong thuoc {n_pairs} cap "
            f"(ISIN, timestamp) co GIA KHAC NHAU (khong phai exact duplicate). "
            f"Giu dong CUOI CUNG -- can kiem tra thu cong."
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

    # RED 08 fix: deterministic order cho TGN event stream.
    df = df.sort_values(["timestamp", "transactionID"], kind="stable").reset_index(drop=True)
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

# ---------------------------------------------------------------------------
# 5. PIPELINE LAM SACH: doc raw -> lam sach -> luu vao processed/
#    Split/snapshot/validation logic da duoc chuyen sang src/data/splitter.py
#    va src/data/build_dataset.py. clean.py CHI lam sach du lieu.
#
#    Flow chuan:
#      raw/  -->  clean.py  -->  processed/  -->  build_dataset.py  -->  splits/
# ---------------------------------------------------------------------------

def run_cleaning(input_dir, output_dir):
    """Doc 6 file goc tu input_dir, lam sach, luu vao output_dir.

    Day la BUOC DUY NHAT cua clean.py. Moi logic ve split, snapshot,
    pair-overlap, va validation deu nam trong src/data/.
    """
    print(f"\n[1/2] Dang doc du lieu tu: {input_dir}")
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

    print("\n[2/2] Dang lam sach du lieu...")
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

    os.makedirs(output_dir, exist_ok=True)
    customer_clean.to_csv(os.path.join(output_dir, "customer_information_clean.csv"), index=False)
    asset_clean.to_csv(os.path.join(output_dir, "asset_information_clean.csv"), index=False)
    markets_clean.to_csv(os.path.join(output_dir, "markets_clean.csv"), index=False)
    prices_clean.to_csv(os.path.join(output_dir, "close_prices_clean.csv"), index=False)
    limit_prices_clean.to_csv(os.path.join(output_dir, "limit_prices_clean.csv"), index=False)
    transactions_clean.to_csv(os.path.join(output_dir, "transactions_clean.csv"), index=False)

    print(f"\nHoan tat. Da luu 6 file sach vao: {output_dir}")
    print("Buoc tiep theo: chay  python -m src.data.build_dataset  de tao splits + validation.")


# ---------------------------------------------------------------------------
# 6. SINH DU LIEU GIA (SELF-TEST)
# ---------------------------------------------------------------------------

def generate_synthetic_data(target_dir, n_customers=200, n_assets=30, n_transactions=3000, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(target_dir, exist_ok=True)

    date_range = pd.date_range("2018-01-02", "2022-11-29", freq="D")

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
                "customerID": cid, "customerType": rng.choice(cust_types),
                "riskLevel": rng.choice(risk_levels),
                "investmentCapacity": rng.choice(caps),
                "lastQuestionnaireDate": d, "timestamp": d,
            })
    pd.DataFrame(cust_rows).to_csv(os.path.join(target_dir, "customer_information.csv"), index=False)

    categories = ["Stock", "Bond", "MTF"]
    isins = [f"ISIN{a:04d}" for a in range(n_assets)]
    asset_rows = []
    for isin in isins:
        n_updates = rng.integers(1, 3)
        update_dates = sorted(rng.choice(date_range, size=n_updates, replace=False))
        for d in update_dates:
            asset_rows.append({
                "ISIN": isin, "assetName": f"Asset {isin}", "assetShortName": isin,
                "assetCategory": rng.choice(categories), "assetSubCategory": "SubcatA",
                "marketID": f"MKT{rng.integers(0, 5):02d}",
                "sector": rng.choice(["Tech", "Finance", "Energy", None]),
                "industry": rng.choice(["Software", "Banking", "Oil", None]),
                "timestamp": d,
            })
    pd.DataFrame(asset_rows).to_csv(os.path.join(target_dir, "asset_information.csv"), index=False)

    market_rows = [{
        "exchangeID": f"EXC{m:02d}", "marketID": f"MKT{m:02d}", "name": f"Market {m}",
        "description": "desc", "country": "GR", "tradingDays": "Mon,Tue,Wed,Thu,Fri",
        "tradingHours": "09:00-17:00", "marketClass": "Regulated",
    } for m in range(5)]
    pd.DataFrame(market_rows).to_csv(os.path.join(target_dir, "markets.csv"), index=False)

    price_rows = []
    for isin in isins:
        price = rng.uniform(10, 100)
        for d in date_range[::5]:
            price *= (1 + rng.normal(0, 0.02))
            price = max(price, 0.5)
            price_rows.append({"ISIN": isin, "timestamp": d, "closePrice": round(price, 4)})
    pd.DataFrame(price_rows).to_csv(os.path.join(target_dir, "close_prices.csv"), index=False)

    limit_rows = []
    for isin in isins:
        sub = [r for r in price_rows if r["ISIN"] == isin]
        limit_rows.append({
            "ISIN": isin, "minDate": sub[0]["timestamp"], "maxDate": sub[-1]["timestamp"],
            "priceMinDate": sub[0]["closePrice"], "priceMaxDate": sub[-1]["closePrice"],
            "profitability": sub[-1]["closePrice"] / sub[0]["closePrice"] - 1,
        })
    pd.DataFrame(limit_rows).to_csv(os.path.join(target_dir, "limit_prices.csv"), index=False)

    cust_ids = [f"CUST{c:05d}" for c in range(n_customers)]
    txn_dates = pd.date_range("2018-01-02", "2022-11-29", freq="D")
    txn_rows = []
    for i in range(n_transactions):
        txn_rows.append({
            "customerID": rng.choice(cust_ids), "ISIN": rng.choice(isins),
            "transactionID": f"TXN{i:06d}", "transactionType": rng.choice(["Buy", "Sell"]),
            "timestamp": rng.choice(txn_dates),
            "totalValue": round(rng.uniform(100, 5000), 2),
            "units": round(rng.uniform(1, 100), 2),
            "channel": rng.choice(["Internet Banking", "Phone Banking", "Branch"]),
            "marketID": f"MKT{rng.integers(0, 5):02d}",
        })
    pd.DataFrame(txn_rows).to_csv(os.path.join(target_dir, "transactions.csv"), index=False)
    print(f"Da sinh du lieu gia (self-test) vao: {target_dir}")


# ---------------------------------------------------------------------------
# 7. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lam sach du lieu FAR-Trans (CHI cleaning, khong split)"
    )
    parser.add_argument("--input_dir", type=str, default="../raw",
                         help="Thu muc chua 6 file CSV goc")
    parser.add_argument("--output_dir", type=str, default=".",
                         help="Thu muc se ghi 6 file _clean.csv")
    parser.add_argument("--self_test", action="store_true",
                         help="Sinh du lieu gia va chay thu cleaning")
    args = parser.parse_args()

    if args.self_test:
        synth_dir = "./_far_trans_synthetic"
        if os.path.exists(synth_dir):
            shutil.rmtree(synth_dir)
        generate_synthetic_data(synth_dir)
        run_cleaning(synth_dir, args.output_dir)
    else:
        run_cleaning(args.input_dir, args.output_dir)
