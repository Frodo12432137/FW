# predict_temperatura_final.py
# Inferencja korekty temperatury — bezpieczne generowanie lagów (1h..6h przy 15-min) i predykcja
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
# KONFIGURACJA
# ============================
SQL_PROGNOZA_PATH = r"D:\app\weatherCorrection_fv\SQL\pogodaJankins.sql"
SQL_WYKONANIE_PATH = r"D:\app\weatherCorrection_fv\SQL\wykonanie.sql"
ARTIFACTS_DIR = r"D:\app\weatherCorrection_fv\Temperatura\artifacts"
OUTPUT_CSV = r"D:\app\weatherCorrection_fv\Temperatura\prognoza_temperatura_korekta.csv"

USE_15MIN = True
LAG_SHIFTS = list(range(4, 300)) if USE_15MIN else list(range(1, 25))  # 4..24 => 1h..6h

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


def load_artifacts(dir_path: str):
    arte = Path(dir_path)
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
    sql = Path(sql_path).read_text(encoding="utf-8")
    for k, v in kwargs.items():
        sql = re.sub(rf"SET @{k} =.*", rf"SET @{k} = '{v}'", sql)
    conn = pyodbc.connect(connection_string)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()

# ============================
# Forecast preparation
# ============================
def prepare_forecast(sql_path: str) -> pd.DataFrame:
    df = _query2df(sql_path, CONN_PROGNOZA)

    if 'punkt' in df:
        df['punkt'] = df['punkt'].replace(REPLACE_DICT)
        df = df.rename(columns={'punkt': 'lokalizacja'})
    elif 'lokalizacja' in df:
        df['lokalizacja'] = df['lokalizacja'].replace(REPLACE_DICT)
    else:
        raise ValueError("Brak kolumny 'punkt' lub 'lokalizacja'")

    if 'czasDanychZrodlaCET' in df:
        df['czasDanychZrodlaCET'] = pd.to_datetime(df['czasDanychZrodlaCET'], errors='coerce')
    if 'execId' in df:
        df['execId'] = df['execId']

    df['dataGodzinaCET'] = pd.to_datetime(df['dataGodzinaCET'], errors='coerce')
    if df.empty:
        raise ValueError("Brak danych prognozy")

    df = df.sort_values(['lokalizacja', 'dataGodzinaCET']).reset_index(drop=True)
    df['godzina'] = df['dataGodzinaCET'].dt.hour
    df['miesiac'] = df['dataGodzinaCET'].dt.month

    for c in ["predkoscWiatru", "temperatura", "kierunekWiatru"]:
        if c in df:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False).str.strip()
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for base, roll_col, win in ROLLING_FEATURES:
        if base in df:
            df[roll_col] = df.groupby('lokalizacja')[base].transform(lambda x: x.rolling(win, min_periods=1).mean())

    return df

# ============================
# Prepare wykonanie + lagi (temperatura)
# ============================
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

    if 'temperaturaLokalizacja_wykonanie' in wykonanie.columns:
        wykonanie['temperatura_wykonanie'] = wykonanie['temperaturaLokalizacja_wykonanie']
    elif 'temperatura' in wykonanie.columns:
        wykonanie['temperatura_wykonanie'] = wykonanie['temperatura']
    else:
        return pd.DataFrame(0.0, index=prognoza_df.index, columns=[f"diff_lag_{s}" for s in LAG_SHIFTS])

    wykonanie['temperatura_wykonanie'] = wykonanie['temperatura_wykonanie'].astype(str).str.replace(',', '.', regex=False).str.strip()
    wykonanie['temperatura_wykonanie'] = pd.to_numeric(wykonanie['temperatura_wykonanie'], errors='coerce')

    hist = pd.merge(
        prognoza_df[['lokalizacja', 'dataGodzinaCET', 'temperatura']],
        wykonanie[['lokalizacja', 'dataGodzinaCET', 'temperatura_wykonanie']],
        on=['lokalizacja', 'dataGodzinaCET'],
        how='inner'
    )

    if hist.empty:
        return pd.DataFrame(0.0, index=prognoza_df.index, columns=[f"diff_lag_{s}" for s in LAG_SHIFTS])

    hist['diff'] = hist['temperatura_wykonanie'].astype(float) - hist['temperatura'].astype(float)
    hist = hist.sort_values(['lokalizacja', 'dataGodzinaCET']).groupby(['lokalizacja','dataGodzinaCET'], as_index=False).last()
    hist = hist.sort_values(['lokalizacja', 'dataGodzinaCET'])

    for s in LAG_SHIFTS:
        hist[f"diff_lag_{s}"] = hist.groupby('lokalizacja')['diff'].shift(s)

    lag_cols = [f"diff_lag_{s}" for s in LAG_SHIFTS]
    hist_lags = hist[['lokalizacja', 'dataGodzinaCET'] + lag_cols]

    full_idx = prognoza_df[['lokalizacja', 'dataGodzinaCET']].drop_duplicates().sort_values(['lokalizacja', 'dataGodzinaCET'])
    full = pd.merge(full_idx, hist_lags, on=['lokalizacja', 'dataGodzinaCET'], how='left')
    full = full.drop_duplicates(subset=['lokalizacja', 'dataGodzinaCET']).sort_values(['lokalizacja', 'dataGodzinaCET'])

    if lag_cols:
        full[lag_cols] = full.groupby('lokalizacja')[lag_cols].ffill().fillna(0.0)
    else:
        return pd.DataFrame(0.0, index=prognoza_df.index, columns=[])

    merged_with_forecast = prognoza_df.reset_index().merge(
        full[['lokalizacja', 'dataGodzinaCET'] + lag_cols],
        on=['lokalizacja', 'dataGodzinaCET'],
        how='left'
    )

    merged_with_forecast[lag_cols] = merged_with_forecast[lag_cols].fillna(0.0)
    lag_values = merged_with_forecast[lag_cols].to_numpy()
    if lag_values.shape[0] != len(prognoza_df):
        tmp = prognoza_df.reset_index()[['index','lokalizacja','dataGodzinaCET']]
        merged2 = tmp.merge(full[['lokalizacja','dataGodzinaCET'] + lag_cols], on=['lokalizacja','dataGodzinaCET'], how='left')
        merged2[lag_cols] = merged2[lag_cols].ffill().fillna(0.0)
        lag_values = merged2[lag_cols].to_numpy()
        if lag_values.shape[0] != len(prognoza_df):
            lag_values = np.zeros((len(prognoza_df), len(lag_cols)), dtype=float)

    result = pd.DataFrame(lag_values, index=prognoza_df.index, columns=lag_cols)
    return result

# ============================
# Dedupe, align, predict
# ============================
def _dedup(df: pd.DataFrame, time_col, loc_col) -> pd.DataFrame:
    nums = df.select_dtypes(include='number').columns.tolist()
    ids  = [loc_col, time_col]
    others = [c for c in df if c not in ids+nums]
    g_num = df.groupby([loc_col,time_col])[nums].mean() if nums else None
    if others:
        g_first = (df.sort_values([loc_col,time_col])
                     .drop_duplicates([loc_col,time_col],keep='first')
                     .set_index([loc_col,time_col])[others])
        combined = pd.concat([g_num,g_first],axis=1) if g_num is not None else g_first
    else:
        combined = g_num
    return combined.reset_index()

def align_features(df: pd.DataFrame, features: list):
    tmp = df.copy()
    tmp['lokalizacja_str'] = tmp['lokalizacja'].astype(str)
    ohe = pd.get_dummies(tmp, columns=['lokalizacja'], drop_first=True)
    for f in features:
        if f not in ohe:
            ohe[f] = 0.0
    X = ohe[features].copy()
    mask = ~X.isna().any(axis=1)
    return X.loc[mask].copy(), tmp.loc[mask].copy()

def main():
    model, scaler, features = load_artifacts(ARTIFACTS_DIR)
    df_raw  = prepare_forecast(SQL_PROGNOZA_PATH)
    df_dedup= _dedup(df_raw, time_col="dataGodzinaCET", loc_col="lokalizacja")

    lag_df = prepare_wykonanie_and_lags(SQL_WYKONANIE_PATH, df_dedup)
    for col in lag_df.columns:
        df_dedup[col] = lag_df[col].values

    X, df_ok = align_features(df_dedup, features)
    X_scaled = scaler.transform(X.values.astype(np.float64))
    with torch.no_grad():
        korekta = model(torch.from_numpy(X_scaled).float()).cpu().numpy().reshape(-1)

    out = df_ok.copy()
    out['korekta_modelu'] = korekta
    if 'temperatura' not in out:
        raise ValueError("Brak kolumny 'temperatura' w prognozie")
    out['skorygowana_temperatura'] = out['temperatura'] + out['korekta_modelu']

    if 'execId' not in out:
        out['execId'] = ""
    if 'czasDanychZrodlaCET' not in out:
        out['czasDanychZrodlaCET'] = ''

    try:
        out['czasDanychZrodlaCET'] = pd.to_datetime(out['czasDanychZrodlaCET'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        out['czasDanychZrodlaCET'] = out['czasDanychZrodlaCET'].fillna('')
    except Exception:
        out['czasDanychZrodlaCET'] = out['czasDanychZrodlaCET'].astype(str).fillna('')

    final_cols = [
        'dataGodzinaCET',
        'czasDanychZrodlaCET',
        'execId',
        'lokalizacja',
        'temperatura',
        'korekta_modelu',
        'skorygowana_temperatura',
        'idPunkt'
    ]
    final_df = out[final_cols]

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Zapisano prognoze temperatury z korekta (CSV): {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
