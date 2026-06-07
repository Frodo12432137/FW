import json
from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import pyodbc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


USE_15MIN = True

SQL_PROGNOZA_PATH = r"D:\app\weatherCorrection_fv\SQL\pogodaprognoza.sql"
SQL_WYKONANIE_PATH = r"D:\app\weatherCorrection_fv\SQL\wykonanie.sql"
OUTPUT_DIR = r"D:\app\weatherCorrection_fv\Temperatura\artifacts"

RANDOM_STATE = 42
EPOCHS = 2000
LR = 0.009
WEIGHT_DECAY = 0.0001
TEST_SIZE = 0.2

if USE_15MIN:
    FLOOR_UNIT = "15min"
    VALID_MINUTES = [0, 15, 30, 45]
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
    RES_LABEL = "15min"
else:
    FLOOR_UNIT = "h"
    VALID_MINUTES = [0]
    ROLLING_FEATURES = [
        ("kierunekWiatru", "kierunek_srednia4", 12),
        ("predkoscWiatru", "predkosc_srednia12", 4),
        ("temperatura",    "temperatura_srednia24", 24),
    ]
    RES_LABEL = "hourly"

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


def _query2df(sql_path: str, connection_string: str, **kwargs) -> pd.DataFrame:
    sql_query = open(sql_path, mode='rt', encoding='utf-8').read()
    for key, value in kwargs.items():
        sql_query = re.sub(rf"SET @{key} =.*", rf"SET @{key} = '{value}'", sql_query)
    conn = pyodbc.connect(connection_string)
    try:
        df = pd.read_sql_query(sql_query, conn)
    finally:
        conn.close()
    return df

def load_and_prepare(sql_prognoza: str, sql_wykonanie: str) -> pd.DataFrame:
    prognoza = _query2df(sql_prognoza, CONN_PROGNOZA)
    wykonanie = _query2df(sql_wykonanie, CONN_WYKONANIE)

    if 'punkt' in prognoza.columns:
        prognoza['punkt'] = prognoza['punkt'].replace(REPLACE_DICT)
        prognoza = prognoza.rename(columns={'punkt': 'lokalizacja'})
    elif 'lokalizacja' in prognoza.columns:
        prognoza['lokalizacja'] = prognoza['lokalizacja'].replace(REPLACE_DICT)

    if 'lokalizacja' in wykonanie.columns:
        wykonanie['lokalizacja'] = wykonanie['lokalizacja'].replace(REPLACE_DICT)

    prognoza['dataGodzinaCET'] = pd.to_datetime(prognoza['dataGodzinaCET']).dt.floor(FLOOR_UNIT)

    if 'DataiCzasOdczytu' in wykonanie.columns:
        wykonanie['DataiCzasOdczytu'] = pd.to_datetime(wykonanie['DataiCzasOdczytu'])
        wykonanie['dataGodzinaCET'] = wykonanie['DataiCzasOdczytu'].dt.floor(FLOOR_UNIT)
        if USE_15MIN:
            wykonanie = wykonanie[
                wykonanie['DataiCzasOdczytu'].dt.minute.isin(VALID_MINUTES)
                & (wykonanie['DataiCzasOdczytu'].dt.second == 0)
            ]
        else:
            wykonanie = wykonanie[
                (wykonanie['DataiCzasOdczytu'].dt.minute == 0)
                & (wykonanie['DataiCzasOdczytu'].dt.second == 0)
            ]
    elif 'dataGodzinaCET' in wykonanie.columns:
        wykonanie['dataGodzinaCET'] = pd.to_datetime(wykonanie['dataGodzinaCET']).dt.floor(FLOOR_UNIT)

    merged = pd.merge(prognoza, wykonanie, on=['dataGodzinaCET', 'lokalizacja'], how='inner')

    columns_to_convert = [
        "predkoscWiatru",
        "predkoscWiatruLokalizacja_wykonanie",
        "temperaturaLokalizacja_wykonanie",
        "temperatura",
        "kierunekWiatruLokalizacja_wykonanie",
        "kierunekWiatru",
    ]
    for col in columns_to_convert:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).str.replace(',', '.', regex=False).str.strip()
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    merged = merged.sort_values(['lokalizacja', 'dataGodzinaCET'])
    for base_col, roll_col, window in ROLLING_FEATURES:
        if base_col in merged.columns:
            merged[roll_col] = merged.groupby('lokalizacja')[base_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

    # cechy czasowe
    merged['godzina'] = merged['dataGodzinaCET'].dt.hour
    merged['miesiac'] = merged['dataGodzinaCET'].dt.month

    # >>> TEMPERATURE DIFF <<<
    merged['diff'] = merged["temperaturaLokalizacja_wykonanie"] - merged["temperatura"]

    # Dodajemy lagi dla temperatury (1h..6h przy 15-min => shifty 4..24)
    if USE_15MIN:
        LAGS = list(range(4, 300))  # 4..24
    else:
        LAGS = list(range(1, 25))  # 1..24 godzin

    merged = merged.sort_values(['lokalizacja', 'dataGodzinaCET'])
    if 'diff' in merged.columns:
        for lag in LAGS:
            col_name = f"diff_lag_{lag}"
            merged[col_name] = merged.groupby('lokalizacja')['diff'].shift(lag)

    # roll names used for dropna
    roll_names = [r for _, r, _ in ROLLING_FEATURES]

    # dropna: nie wymuszamy obecności wszystkich lagów (żeby nie tracić zbyt wielu wierszy)
    merged = merged.dropna(
        subset=[
            "predkoscWiatru",
            "kierunekWiatru",
            "temperatura",
            "predkoscWiatruLokalizacja_wykonanie",
            "kierunekWiatruLokalizacja_wykonanie",
            "temperaturaLokalizacja_wykonanie",
            *roll_names,
        ]
    )

    merged = pd.get_dummies(merged, columns=['lokalizacja'], drop_first=True)

    # upewnij się że godzina, miesiac i diff pozostają
    merged['godzina'] = merged['dataGodzinaCET'].dt.hour
    merged['miesiac'] = merged['dataGodzinaCET'].dt.month
    merged['diff'] = merged["temperaturaLokalizacja_wykonanie"] - merged["temperatura"]

    return merged

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
        x = self.fc5(x)
        return x

def train_model(
    df: pd.DataFrame,
    outdir: Path,
    random_state: int = 42,
    epochs: int = 2500,
    lr: float = 0.009,
    weight_decay: float = 0.0001,
    test_size: float = 0.2
):
    roll_names = [r for _, r, _ in ROLLING_FEATURES]

    # uwzględniamy lagi diff_lag_* jako cechy (jeżeli istnieją w df)
    lag_cols = [c for c in df.columns if c.startswith("diff_lag_")]

    x_cols_core = [
        "predkoscWiatru",
        "kierunekWiatru",
        "temperatura",
        "godzina",
        "miesiac",
        *roll_names,
        *lag_cols,
    ]
    loc_cols = [c for c in df.columns if c.startswith("lokalizacja_")]
    x_cols = x_cols_core + loc_cols

    X = df[x_cols].copy()
    y = df["diff"].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    nan_mask_train = ~np.isnan(x_train).any(axis=1)
    x_train = x_train[nan_mask_train]
    y_train = y_train.iloc[nan_mask_train]

    nan_mask_test = ~np.isnan(x_test).any(axis=1)
    x_test = x_test[nan_mask_test]
    y_test = y_test.iloc[nan_mask_test]

    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train.values.reshape(-1, 1)).float()
    x_test_t = torch.from_numpy(x_test).float()
    y_test_t = torch.from_numpy(y_test.values.reshape(-1, 1)).float()

    model = NeuralNet(input_dim=x_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        outputs = model(x_train_t)
        loss = criterion(outputs, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        y_pred_t = model(x_test_t)
        mse = criterion(y_pred_t, y_test_t).item()
    print(f"(MSE) na zbiorze testowym: {mse:.6f}")

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model.pth"
    scaler_path = outdir / "scaler.joblib"
    features_path = outdir / "features.json"
    metadata_path = outdir / "metadata.json"

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump({"x_cols": x_cols}, f, ensure_ascii=False, indent=2)

    metadata = {
        "random_state": random_state,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "test_size": test_size,
        "resolution": RES_LABEL,
        "rolling_features": [
            {"base": b, "name": r, "window": w} for (b, r, w) in ROLLING_FEATURES
        ],
        "replace_dict_keys": list(REPLACE_DICT.keys()),
        "model_architecture": [X.shape[1], 256, 128, 64, 32, 1],
        "mse_test": mse,
        "target": "temperatura_wykonanie_minus_prognoza",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] Zapisano:\n  - {model_path}\n  - {scaler_path}\n  - {features_path}\n  - {metadata_path}")

if __name__ == "__main__":
    df = load_and_prepare(SQL_PROGNOZA_PATH, SQL_WYKONANIE_PATH)
    train_model(
        df=df,
        outdir=Path(OUTPUT_DIR),
        random_state=RANDOM_STATE,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        test_size=TEST_SIZE,
    )
