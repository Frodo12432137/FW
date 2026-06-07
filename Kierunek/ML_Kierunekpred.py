# ML_Kierunek_pred_final.py
# Inferencja korekty kierunku wiatru — pełny skrypt z bezpiecznym generowaniem lagów (1h..6h przy 15-min)
# Wymaga artefaktów: model.pth, scaler.joblib, features.json w ARTIFACTS_DIR

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import joblib
import pyodbc
import torch
import torch.nn as nn

# ============================
# KONFIGURACJA DOMYŚLNA
# ============================
SQL_PROGNOZA_PATH = r"D:\app\weatherCorrection_fv\SQL\pogodaJankins.sql"
SQL_WYKONANIE_PATH = r"D:\app\weatherCorrection_fv\SQL\wykonanie.sql"
ARTIFACTS_DIR = r"D:\app\weatherCorrection_fv\Kierunek\artifacts"
OUTPUT_CSV = r"D:\app\weatherCorrection_fv\Kierunek\prognoza_kierunek_korekta.csv"

USE_15MIN = True
LAG_SHIFTS = list(range(4, 300)) if USE_15MIN else list(range(1, 25))  # 4..24 => 1h..6h dla 15-min

ROLLING_FEATURES = [
        ("kierunekWiatru", "kierunek_srednia16",90),     
        ("predkoscWiatru", "predkosc_srednia16", 90),     
        ("temperatura",    "temperatura_srednia16", 90),
        ("kierunekWiatru", "kierunek_srednia48", 96),     
        ("predkoscWiatru", "predkosc_srednia48", 96),    
        ("temperatura",    "temperatura_srednia48", 96),
        ("kierunekWiatru", "kierunek_srednia54", 100),     
        ("predkoscWiatru", "predkosc_srednia54", 100),     
        ("temperatura",    "temperatura_srednia54", 100) ,  
        ("kierunekWiatru", "kierunek_srednia16", 102),     
        ("predkoscWiatru", "predkosc_srednia16", 102),     
        ("temperatura",    "temperatura_srednia16", 102),
        ("kierunekWiatru", "kierunek_srednia48", 102),     
        ("predkoscWiatru", "predkosc_srednia48", 102),    
        ("temperatura",    "temperatura_srednia48", 102),
        ("kierunekWiatru", "kierunek_srednia54", 110),     
        ("predkoscWiatru", "predkosc_srednia54", 110),     
        ("temperatura",    "temperatura_srednia54", 110) ,
    ]

REPLACE_DICT = {
    'Galicja (Hnatkowice – Orzechowce)': 'GAL',
    'Kamieńsk': 'KAM',
    'Karnice': 'KAR',
    'Karnice II': 'KAR2',
    'Kisielice': 'KIS',
    'Kisielice II': 'KIS2',
    'Karwice': 'KRW',
    'Jagniątkowo (Lake Ostrowo)': 'LOS',
    'Lotnisko': 'LOT',
    'Malbork (Koniecwałd)': 'MAL',
    'Pelplin': 'PEL',
    'Resko I': 'RES',
    'Resko II': 'RES2',
    'Rybice': 'RYB',
    'Skoczykłody': 'SKO',
    'Starza': 'STA',
    'Wojciechowo': 'WOJ',
    'Zalesie': 'ZAL',
    'Żuromin': 'ZUR'
}

CONN_PROGNOZA = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "Server={MISDWHPRD.GKPGE.PL};"
    "DATABASE=PGESA_MarketAnalytics;"
    "Trusted_Connection=yes;"
)

CONN_WYKONANIE = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "Server={MISDWHPRD.GKPGE.PL};"
    "DATABASE=PGEEO_DDS;"
    "Trusted_Connection=yes;"
)

# ============================
# Model + loader
# ============================
class NeuralNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


def load_artifacts(artifacts_dir: str):
    arte = Path(artifacts_dir)
    features = json.loads((arte / "features.json").read_text(encoding="utf-8"))["x_cols"]
    scaler = joblib.load(arte / "scaler.joblib")
    model = NeuralNet(input_dim=len(features))
    state = torch.load(arte / "model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, scaler, features

# ============================
# SQL helper
# ============================
def _query2df(sql_path: str, connection_string: str, **kwargs) -> pd.DataFrame:
    sql_query = Path(sql_path).read_text(encoding="utf-8")
    for key, value in kwargs.items():
        sql_query = re.sub(rf"SET @{key} =.*", rf"SET @{key} = '{value}'", sql_query)
    conn = pyodbc.connect(connection_string)
    try:
        df = pd.read_sql_query(sql_query, conn)
    finally:
        conn.close()
    return df

# ============================
# Forecast preparation
# ============================
def prepare_forecast(sql_prognoza_path: str) -> pd.DataFrame:
    prognoza = _query2df(sql_prognoza_path, CONN_PROGNOZA)

    if "punkt" in prognoza.columns:
        prognoza = prognoza.rename(columns={"punkt": "lokalizacja"})
    prognoza["lokalizacja"] = prognoza["lokalizacja"].replace(REPLACE_DICT)

    prognoza["dataGodzinaCET"] = pd.to_datetime(prognoza["dataGodzinaCET"], errors="coerce")
    prognoza = prognoza.sort_values(["lokalizacja", "dataGodzinaCET"]).reset_index(drop=True)

    prognoza["godzina"] = prognoza["dataGodzinaCET"].dt.hour
    prognoza["miesiac"] = prognoza["dataGodzinaCET"].dt.month

    for col in ["predkoscWiatru", "temperatura", "kierunekWiatru"]:
        if col in prognoza.columns:
            prognoza[col] = prognoza[col].astype(str).str.replace(",", ".", regex=False).str.strip()
            prognoza[col] = pd.to_numeric(prognoza[col], errors="coerce")

    for base_col, roll_col, window in ROLLING_FEATURES:
        if base_col in prognoza.columns:
            prognoza[roll_col] = prognoza.groupby("lokalizacja")[base_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

    return prognoza

# ============================
# Wykonanie + lagi dla kąta (angular diff)
# ============================
def angle_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b + 180) % 360 - 180

def prepare_wykonanie_and_lags(sql_wykonanie_path: str, prognoza_df: pd.DataFrame) -> pd.DataFrame:
    try:
        wykonanie = _query2df(sql_wykonanie_path, CONN_WYKONANIE)
    except Exception:
        return pd.DataFrame(0.0, index=prognoza_df.index, columns=[f"diff_lag_{s}" for s in LAG_SHIFTS])

    if 'punkt' in wykonanie.columns:
        wykonanie['punkt'] = wykonanie['punkt'].replace(REPLACE_DICT)
        wykonanie = wykonanie.rename(columns={'punkt': 'lokalizacja'})
    elif 'lokalizacja' in wykonanie.columns:
        wykonanie['lokalizacja'] = wykonanie['lokalizacja'].replace(REPLACE_DICT)

    if 'DataiCzasOdczytu' in wykonanie.columns:
        wykonanie['DataiCzasOdczytu'] = pd.to_datetime(wykonanie['DataiCzasOdczytu'], errors='coerce')
        wykonanie['dataGodzinaCET'] = wykonanie['DataiCzasOdczytu'].dt.floor('15min' if USE_15MIN else 'h')
    elif 'dataGodzinaCET' in wykonanie.columns:
        wykonanie['dataGodzinaCET'] = pd.to_datetime(wykonanie['dataGodzinaCET'], errors='coerce')
        wykonanie['dataGodzinaCET'] = wykonanie['dataGodzinaCET'].dt.floor('15min' if USE_15MIN else 'h')
    else:
        return pd.DataFrame(0.0, index=prognoza_df.index, columns=[f"diff_lag_{s}" for s in LAG_SHIFTS])

    if 'kierunekWiatru' in wykonanie.columns:
        wykonanie['kierunekWiatru_wykonanie'] = wykonanie['kierunekWiatru'].astype(str).str.replace(',', '.', regex=False).str.strip()
        wykonanie['kierunekWiatru_wykonanie'] = pd.to_numeric(wykonanie['kierunekWiatru_wykonanie'], errors='coerce')
    elif 'kierunekWiatruLokalizacja_wykonanie' in wykonanie.columns:
        wykonanie['kierunekWiatru_wykonanie'] = wykonanie['kierunekWiatruLokalizacja_wykonanie'].astype(str).str.replace(',', '.', regex=False).str.strip()
        wykonanie['kierunekWiatru_wykonanie'] = pd.to_numeric(wykonanie['kierunekWiatru_wykonanie'], errors='coerce')
    else:
        return pd.DataFrame(0.0, index=prognoza_df.index, columns=[f"diff_lag_{s}" for s in LAG_SHIFTS])

    hist = pd.merge(
        prognoza_df[['lokalizacja', 'dataGodzinaCET', 'kierunekWiatru']],
        wykonanie[['lokalizacja', 'dataGodzinaCET', 'kierunekWiatru_wykonanie']],
        on=['lokalizacja', 'dataGodzinaCET'],
        how='inner'
    )

    if hist.empty:
        return pd.DataFrame(0.0, index=prognoza_df.index, columns=[f"diff_lag_{s}" for s in LAG_SHIFTS])

    hist['diff'] = angle_diff(hist['kierunekWiatru_wykonanie'].fillna(0.0), hist['kierunekWiatru'].fillna(0.0))

    # deduplicate and sort
    hist = hist.sort_values(['lokalizacja', 'dataGodzinaCET']).groupby(['lokalizacja','dataGodzinaCET'], as_index=False).last()
    hist = hist.sort_values(['lokalizacja', 'dataGodzinaCET'])

    for s in LAG_SHIFTS:
        hist[f"diff_lag_{s}"] = hist.groupby('lokalizacja')['diff'].shift(s)

    lag_cols = [f"diff_lag_{s}" for s in LAG_SHIFTS]
    hist_lags = hist[['lokalizacja', 'dataGodzinaCET'] + lag_cols]

    full_idx = prognoza_df[['lokalizacja', 'dataGodzinaCET']].drop_duplicates().sort_values(['lokalizacja','dataGodzinaCET'])
    full = pd.merge(full_idx, hist_lags, on=['lokalizacja','dataGodzinaCET'], how='left')
    full = full.drop_duplicates(subset=['lokalizacja','dataGodzinaCET']).sort_values(['lokalizacja','dataGodzinaCET'])

    if lag_cols:
        full[lag_cols] = full.groupby('lokalizacja')[lag_cols].ffill().fillna(0.0)
    else:
        return pd.DataFrame(0.0, index=prognoza_df.index, columns=[])

    merged_with_forecast = prognoza_df.reset_index().merge(
        full[['lokalizacja','dataGodzinaCET'] + lag_cols],
        on=['lokalizacja','dataGodzinaCET'],
        how='left'
    )

    merged_with_forecast[lag_cols] = merged_with_forecast[lag_cols].fillna(0.0)

    # create result preserving order of prognoza_df.reset_index()
    lag_values = merged_with_forecast[lag_cols].to_numpy()
    if lag_values.shape[0] != len(prognoza_df):
        # fallback: try merge via index mapping and ffill
        tmp = prognoza_df.reset_index()[['index','lokalizacja','dataGodzinaCET']]
        merged2 = tmp.merge(full[['lokalizacja','dataGodzinaCET'] + lag_cols], on=['lokalizacja','dataGodzinaCET'], how='left')
        merged2[lag_cols] = merged2[lag_cols].ffill().fillna(0.0)
        lag_values = merged2[lag_cols].to_numpy()
        if lag_values.shape[0] != len(prognoza_df):
            lag_values = np.zeros((len(prognoza_df), len(lag_cols)), dtype=float)

    result = pd.DataFrame(lag_values, index=prognoza_df.index, columns=lag_cols)
    return result

# ============================
# Align features, normalize angle, dedup
# ============================
def align_features(df: pd.DataFrame, features: list):
    df_ohe = pd.get_dummies(df, columns=["lokalizacja"], drop_first=True)
    for col in features:
        if col not in df_ohe.columns:
            df_ohe[col] = 0.0
    X = df_ohe[features].copy()
    mask = ~X.isna().any(axis=1)
    return X.loc[mask].copy(), df.loc[mask].copy()

def normalize_angle(deg: pd.Series) -> pd.Series:
    return (deg % 360 + 360) % 360

def _dedup_within_location(df: pd.DataFrame, time_col: str, loc_col: str) -> pd.DataFrame:
    num_cols = df.select_dtypes(include="number").columns
    id_cols = [loc_col, time_col]
    other_cols = [c for c in df.columns if c not in id_cols + num_cols.tolist()]
    if len(num_cols):
        g_num = df.groupby([loc_col, time_col])[num_cols].mean()
    else:
        g_num = None
    if other_cols:
        g_first = (
            df.sort_values([loc_col, time_col])
              .drop_duplicates([loc_col, time_col], keep="first")
              .set_index([loc_col, time_col])[other_cols]
        )
        g = pd.concat([g_num, g_first], axis=1) if g_num is not None else g_first
    else:
        g = g_num
    return g.reset_index()

# ============================
# Main
# ============================
def main():
    model, scaler, features = load_artifacts(ARTIFACTS_DIR)

    df_fore = prepare_forecast(SQL_PROGNOZA_PATH)
    df_fore = _dedup_within_location(df_fore, time_col="dataGodzinaCET", loc_col="lokalizacja")

    print("DEBUG: rows forecast:", len(df_fore))
    print("DEBUG: unique (lokalizacja,time):", df_fore[['lokalizacja','dataGodzinaCET']].drop_duplicates().shape[0])

    lag_df = prepare_wykonanie_and_lags(SQL_WYKONANIE_PATH, df_fore)
    for col in lag_df.columns:
        df_fore[col] = lag_df[col].values

    X, df_aligned = align_features(df_fore, features)

    print("DEBUG: features expected:", len(features))
    print("DEBUG: X.shape:", X.shape)

    X_scaled_arr = scaler.transform(X.values.astype(np.float64))
    X_scaled = pd.DataFrame(X_scaled_arr, columns=features, index=X.index)

    with torch.no_grad():
        x_t = torch.from_numpy(X_scaled.values.astype(np.float32))
        korekta = model(x_t).cpu().numpy().reshape(-1)

    out = df_aligned.copy().reset_index(drop=True)
    out["korekta_modelu"] = korekta

    if "kierunekWiatru" not in out.columns:
        raise ValueError("Brak kolumny 'kierunekWiatru' — nie mogę policzyć korekty.")

    out["skorygowany_kierunekWiatru"] = normalize_angle(out["kierunekWiatru"] + out["korekta_modelu"])

    final_cols = [
        "dataGodzinaCET",
        "lokalizacja",
        "kierunekWiatru",
        "korekta_modelu",
        "skorygowany_kierunekWiatru"
    ]
    result = out[final_cols]

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] Zapisano prognozę kierunku z korektą (CSV): {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
