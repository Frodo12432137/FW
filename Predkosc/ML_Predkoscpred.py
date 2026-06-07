# ML_Predkoscpred_fixed_full.py
# Inferencja korekty prędkości wiatru — pełny skrypt z odporną obsługą lagów

import json
from pathlib import Path
import re
import os
import sys

import numpy as np
import pandas as pd
import joblib
import pyodbc
import torch
import torch.nn as nn


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================
# KONFIGURACJA DOMYŚLNA
# ============================

SQL_PROGNOZA_PATH = r"D:\app\weatherCorrection_fv\SQL\pogodaJankins.sql"
SQL_WYKONANIE_PATH = r"D:\app\weatherCorrection_fv\SQL\wykonanie.sql"

OUTPUT_CSV = r"D:\app\weatherCorrection_fv\Predkosc\prognoza_predkosc_korekta.csv"
ARTIFACTS_DIR = r"D:\app\weatherCorrection_fv\Predkosc\artifacts"

USE_15MIN = True

FLOOR_UNIT = "15min" if USE_15MIN else "h"

# 4 dla danych 15-min = 1h
LAG_SHIFTS = list(range(4, 300)) if USE_15MIN else list(range(1, 25))

ROLLING_FEATURES = [
    ("kierunekWiatru", "kierunek_srednia16", 90),
    ("predkoscWiatru", "predkosc_srednia16", 90),
    ("temperatura", "temperatura_srednia16", 90),

    ("kierunekWiatru", "kierunek_srednia48", 96),
    ("predkoscWiatru", "predkosc_srednia48", 96),
    ("temperatura", "temperatura_srednia48", 96),

    ("kierunekWiatru", "kierunek_srednia54", 100),
    ("predkoscWiatru", "predkosc_srednia54", 100),
    ("temperatura", "temperatura_srednia54", 100),

    # Uwaga: poniższe wpisy nadpisują poprzednie kolumny o tych samych nazwach.
    # Zostawiam je, żeby zachować zgodność z istniejącym modelem/features.json.
    ("kierunekWiatru", "kierunek_srednia16", 102),
    ("predkoscWiatru", "predkosc_srednia16", 102),
    ("temperatura", "temperatura_srednia16", 102),

    ("kierunekWiatru", "kierunek_srednia48", 102),
    ("predkoscWiatru", "predkosc_srednia48", 102),
    ("temperatura", "temperatura_srednia48", 102),

    ("kierunekWiatru", "kierunek_srednia54", 110),
    ("predkoscWiatru", "predkosc_srednia54", 110),
    ("temperatura", "temperatura_srednia54", 110),
]

REPLACE_DICT = {
    "Galicja (Hnatkowice - Orzechowce)": "GAL",
    "Galicja (Hnatkowice – Orzechowce)": "GAL",
    "Galicja (Hnatkowice â€“ Orzechowce)": "GAL",

    "Kamieńsk": "KAM",
    "Karnice": "KAR",
    "Karnice II": "KAR2",
    "Kisielice": "KIS",
    "Kisielice II": "KIS2",
    "Karwice": "KRW",

    "Jagniątkowo (Ląk Ostrów)": "LOS",
    "Jagniątkowo (Lake Ostrowo)": "LOS",

    "Lotnisko": "LOT",

    "Malbork (Konieczwałd)": "MAL",
    "Malbork (Koniecwałd)": "MAL",
    "Malbork (KoniecwaÅ‚d)": "MAL",

    "Pelplin": "PEL",
    "Resko I": "RES",
    "Resko II": "RES2",

    "Rybitwy": "RYB",
    "Rybice": "RYB",

    "Skoczykłody": "SKO",
    "Starza": "STA",
    "Wojciechowo": "WOJ",
    "Zalesie": "ZAL",
    "Żuromin": "ZUR",
}

CONN_PROGNOZA = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=MISDWHPRD.GKPGE.PL;"
    "DATABASE=PGESA_MarketAnalytics;"
    "Trusted_Connection=yes;"
)

CONN_WYKONANIE = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=MISDWHPRD.GKPGE.PL;"
    "DATABASE=PGEEO_DDS;"
    "Trusted_Connection=yes;"
)


# ============================
# MODEL
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

    features_path = arte / "features.json"
    scaler_path = arte / "scaler.joblib"
    model_path = arte / "model.pth"

    for path in [features_path, scaler_path, model_path]:
        if not path.exists():
            raise FileNotFoundError(f"Brak wymaganego artefaktu: {path}")

    features = json.loads(
        features_path.read_text(encoding="utf-8")
    )["x_cols"]

    scaler = joblib.load(scaler_path)

    model = NeuralNet(input_dim=len(features))

    state = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(state)
    model.eval()

    print("DEBUG: załadowano liczbę cech modelu:", len(features))

    return model, scaler, features


# ============================
# SQL
# ============================

def _query2df(
    sql_path: str,
    connection_string: str,
    **kwargs,
) -> pd.DataFrame:

    sql_query = Path(sql_path).read_text(encoding="utf-8")

    for key, value in kwargs.items():
        sql_query = re.sub(
            rf"SET @{key}\s*=\s*.*",
            f"SET @{key} = '{value}'",
            sql_query,
        )

    conn = pyodbc.connect(connection_string)

    try:
        df = pd.read_sql_query(sql_query, conn)

    finally:
        conn.close()

    df.columns = [str(c).strip() for c in df.columns]

    return df


# ============================
# HELPERY
# ============================

def prepare_location(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    if "punkt" in df.columns:
        df = df.rename(columns={"punkt": "lokalizacja"})

    if "lokalizacja" not in df.columns:
        raise ValueError("Brak kolumny 'punkt' albo 'lokalizacja'.")

    df["lokalizacja"] = (
        df["lokalizacja"]
        .astype(str)
        .str.strip()
        .replace(REPLACE_DICT)
    )

    return df


def convert_numeric(
    df: pd.DataFrame,
    cols,
) -> pd.DataFrame:

    df = df.copy()

    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["godzina"] = df["dataGodzinaCET"].dt.hour
    df["miesiac"] = df["dataGodzinaCET"].dt.month

    minutes_from_midnight = (
        df["dataGodzinaCET"].dt.hour * 60
        + df["dataGodzinaCET"].dt.minute
    )

    df["sin_time"] = np.sin(
        2 * np.pi * minutes_from_midnight / 1440
    )

    df["cos_time"] = np.cos(
        2 * np.pi * minutes_from_midnight / 1440
    )

    df["sin_month"] = np.sin(
        2 * np.pi * df["miesiac"] / 12
    )

    df["cos_month"] = np.cos(
        2 * np.pi * df["miesiac"] / 12
    )

    return df


def make_empty_lags(index) -> pd.DataFrame:
    return pd.DataFrame(
        0.0,
        index=index,
        columns=[f"diff_lag_{s}" for s in LAG_SHIFTS],
    )


# ============================
# FORECAST PREPARATION
# ============================

def prepare_forecast(sql_prognoza_path: str) -> pd.DataFrame:
    prognoza = _query2df(
        sql_prognoza_path,
        CONN_PROGNOZA,
    )

    prognoza = prepare_location(prognoza)

    if "dataGodzinaCET" not in prognoza.columns:
        raise ValueError("Brak kolumny dataGodzinaCET w prognozie.")

    prognoza["dataGodzinaCET"] = pd.to_datetime(
        prognoza["dataGodzinaCET"],
        errors="coerce",
    ).dt.floor(FLOOR_UNIT)

    prognoza = prognoza.dropna(
        subset=["dataGodzinaCET"]
    )

    prognoza = convert_numeric(
        prognoza,
        [
            "predkoscWiatru",
            "temperatura",
            "kierunekWiatru",
        ],
    )

    prognoza = (
        prognoza
        .sort_values(["lokalizacja", "dataGodzinaCET"])
        .reset_index(drop=True)
    )

    print(
        "DEBUG: unikalne lokalizacje prognoza:",
        sorted(
            prognoza["lokalizacja"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    )

    prognoza = add_time_features(prognoza)

    for base_col, roll_col, window in ROLLING_FEATURES:
        if base_col in prognoza.columns:
            prognoza[roll_col] = (
                prognoza
                .groupby("lokalizacja")[base_col]
                .transform(
                    lambda x: x.rolling(
                        window,
                        min_periods=1,
                    ).mean()
                )
            )
        else:
            prognoza[roll_col] = np.nan

    return prognoza


# ============================
# WYKONANIE + LAGI
# ============================

def detect_wykonanie_speed_column(df: pd.DataFrame) -> str:
    candidates = [
        "predkoscWiatruLokalizacja_wykonanie",
        "predkoscWiatru_wykonanie",
        "predkoscWiatruLokalizacja",
        "predkoscWiatru",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Brak kolumny wykonania prędkości wiatru. "
        f"Dostępne kolumny: {df.columns.tolist()}"
    )


def prepare_wykonanie_and_lags(
    sql_wykonanie_path: str,
    prognoza_df: pd.DataFrame,
) -> pd.DataFrame:

    lag_cols = [f"diff_lag_{s}" for s in LAG_SHIFTS]

    try:
        wykonanie = _query2df(
            sql_wykonanie_path,
            CONN_WYKONANIE,
        )

    except Exception as e:
        print("WARNING: Nie udało się pobrać wykonania. LAGI = 0. Błąd:", repr(e))
        return pd.DataFrame(
            0.0,
            index=prognoza_df.index,
            columns=lag_cols,
        )

    try:
        wykonanie = prepare_location(wykonanie)

    except Exception as e:
        print("WARNING: Nie udało się przygotować lokalizacji wykonania. LAGI = 0. Błąd:", repr(e))
        return pd.DataFrame(
            0.0,
            index=prognoza_df.index,
            columns=lag_cols,
        )

    if "DataiCzasOdczytu" in wykonanie.columns:
        wykonanie["DataiCzasOdczytu"] = pd.to_datetime(
            wykonanie["DataiCzasOdczytu"],
            errors="coerce",
        )

        wykonanie["dataGodzinaCET"] = (
            wykonanie["DataiCzasOdczytu"]
            .dt.floor(FLOOR_UNIT)
        )

    elif "dataGodzinaCET" in wykonanie.columns:
        wykonanie["dataGodzinaCET"] = pd.to_datetime(
            wykonanie["dataGodzinaCET"],
            errors="coerce",
        ).dt.floor(FLOOR_UNIT)

    else:
        print("WARNING: Brak daty w wykonaniu. LAGI = 0.")
        return pd.DataFrame(
            0.0,
            index=prognoza_df.index,
            columns=lag_cols,
        )

    wykonanie = wykonanie.dropna(
        subset=["dataGodzinaCET"]
    )

    try:
        pred_wyk_col = detect_wykonanie_speed_column(wykonanie)

    except Exception as e:
        print("WARNING: Brak kolumny prędkości w wykonaniu. LAGI = 0. Błąd:", repr(e))
        return pd.DataFrame(
            0.0,
            index=prognoza_df.index,
            columns=lag_cols,
        )

    wykonanie[pred_wyk_col] = (
        wykonanie[pred_wyk_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )

    wykonanie[pred_wyk_col] = pd.to_numeric(
        wykonanie[pred_wyk_col],
        errors="coerce",
    )

    hist_base = prognoza_df[
        ["lokalizacja", "dataGodzinaCET", "predkoscWiatru"]
    ].copy()

    merged_hist = pd.merge(
        hist_base,
        wykonanie[
            ["lokalizacja", "dataGodzinaCET", pred_wyk_col]
        ],
        on=["lokalizacja", "dataGodzinaCET"],
        how="inner",
    )

    if merged_hist.empty:
        print("WARNING: Brak wspólnej historii prognoza + wykonanie. LAGI = 0.")
        return pd.DataFrame(
            0.0,
            index=prognoza_df.index,
            columns=lag_cols,
        )

    merged_hist = convert_numeric(
        merged_hist,
        [
            "predkoscWiatru",
            pred_wyk_col,
        ],
    )

    merged_hist["diff"] = (
        merged_hist[pred_wyk_col]
        - merged_hist["predkoscWiatru"]
    )

    merged_hist = (
        merged_hist
        .dropna(subset=["diff"])
        .sort_values(["lokalizacja", "dataGodzinaCET"])
        .groupby(["lokalizacja", "dataGodzinaCET"], as_index=False)
        .last()
    )

    if merged_hist.empty:
        print("WARNING: Historia po dropna diff jest pusta. LAGI = 0.")
        return pd.DataFrame(
            0.0,
            index=prognoza_df.index,
            columns=lag_cols,
        )

    lag_series = []

    for s in LAG_SHIFTS:
        lag_series.append(
            merged_hist
            .groupby("lokalizacja")["diff"]
            .shift(s)
            .rename(f"diff_lag_{s}")
        )

    merged_hist = pd.concat(
        [merged_hist, *lag_series],
        axis=1,
    )

    hist_for_merge = merged_hist[
        ["lokalizacja", "dataGodzinaCET", *lag_cols]
    ].copy()

    full_idx = (
        prognoza_df[
            ["lokalizacja", "dataGodzinaCET"]
        ]
        .drop_duplicates()
        .sort_values(["lokalizacja", "dataGodzinaCET"])
    )

    full = pd.merge(
        full_idx,
        hist_for_merge,
        on=["lokalizacja", "dataGodzinaCET"],
        how="left",
    )

    full = (
        full
        .drop_duplicates(subset=["lokalizacja", "dataGodzinaCET"])
        .sort_values(["lokalizacja", "dataGodzinaCET"])
    )

    full[lag_cols] = (
        full
        .groupby("lokalizacja")[lag_cols]
        .ffill()
        .fillna(0.0)
    )

    aligned = (
        prognoza_df
        .reset_index()
        .merge(
            full[
                ["lokalizacja", "dataGodzinaCET", *lag_cols]
            ],
            on=["lokalizacja", "dataGodzinaCET"],
            how="left",
        )
        .sort_values("index")
    )

    if len(aligned) != len(prognoza_df):
        print("WARNING: Rozjazd liczby wierszy przy dopinaniu lagów. LAGI = 0.")
        return pd.DataFrame(
            0.0,
            index=prognoza_df.index,
            columns=lag_cols,
        )

    aligned[lag_cols] = aligned[lag_cols].fillna(0.0)

    lag_df = pd.DataFrame(
        aligned[lag_cols].to_numpy(),
        index=prognoza_df.index,
        columns=lag_cols,
    )

    print("DEBUG: przygotowano lagi:", lag_df.shape)

    return lag_df


# ============================
# CECHY MODELU
# ============================

def align_features(
    df: pd.DataFrame,
    features: list,
):
    df = df.copy()

    if "lokalizacja" not in df.columns:
        raise ValueError("Brak kolumny lokalizacja przed get_dummies.")

    df_ohe = pd.get_dummies(
        df,
        columns=["lokalizacja"],
        drop_first=True,
        dtype=int,
    )

    missing_cols = [
        col for col in features
        if col not in df_ohe.columns
    ]

    if missing_cols:
        print("WARNING: Brakujące cechy uzupełniam zerami:", len(missing_cols))

        missing_df = pd.DataFrame(
            0.0,
            index=df_ohe.index,
            columns=missing_cols,
        )

        df_ohe = pd.concat(
            [df_ohe, missing_df],
            axis=1,
        )

    X = df_ohe[features].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    mask = ~X.isna().any(axis=1)

    dropped = len(X) - int(mask.sum())

    if dropped > 0:
        print("WARNING: Usunięto wiersze przez NaN w X:", dropped)

    return X.loc[mask].copy(), df.loc[mask].copy()


def _dedup_within_location(
    df: pd.DataFrame,
    time_col: str,
    loc_col: str,
) -> pd.DataFrame:

    df = df.copy()

    duplicated_count = df.duplicated(
        [loc_col, time_col]
    ).sum()

    print(
        "DEBUG: duplikaty przed dedup:",
        int(duplicated_count),
    )

    if duplicated_count == 0:
        return df.reset_index(drop=True)

    num_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    id_cols = [
        loc_col,
        time_col,
    ]

    other_cols = [
        c for c in df.columns
        if c not in id_cols + num_cols
    ]

    g_num = None

    if num_cols:
        g_num = (
            df
            .groupby(id_cols)[num_cols]
            .mean()
        )

    if other_cols:
        g_first = (
            df
            .sort_values(id_cols)
            .drop_duplicates(id_cols, keep="first")
            .set_index(id_cols)[other_cols]
        )

        if g_num is not None:
            g = pd.concat(
                [g_num, g_first],
                axis=1,
            )
        else:
            g = g_first

    else:
        g = g_num

    return g.reset_index()


# ============================
# MAIN
# ============================

def main():
    print("START ML_Predkoscpred_fixed_full")
    print("Aktualny katalog roboczy:", os.getcwd())
    print("SQL_PROGNOZA_PATH:", SQL_PROGNOZA_PATH)
    print("SQL_WYKONANIE_PATH:", SQL_WYKONANIE_PATH)
    print("ARTIFACTS_DIR:", ARTIFACTS_DIR)
    print("OUTPUT_CSV:", OUTPUT_CSV)
    print("FLOOR_UNIT:", FLOOR_UNIT)

    model, scaler, features = load_artifacts(
        ARTIFACTS_DIR,
    )

    df_fore = prepare_forecast(
        SQL_PROGNOZA_PATH,
    )

    df_fore = _dedup_within_location(
        df_fore,
        time_col="dataGodzinaCET",
        loc_col="lokalizacja",
    )

    print("DEBUG: rows forecast:", len(df_fore))

    print(
        "DEBUG: unique lokalizacja/time:",
        df_fore[
            ["lokalizacja", "dataGodzinaCET"]
        ].drop_duplicates().shape[0],
    )

    lag_df = prepare_wykonanie_and_lags(
        SQL_WYKONANIE_PATH,
        df_fore,
    )

    df_fore_with_lags = pd.concat(
        [
            df_fore.reset_index(drop=True),
            lag_df.reset_index(drop=True),
        ],
        axis=1,
    )

    X, df_aligned = align_features(
        df_fore_with_lags,
        features,
    )

    print("DEBUG: features expected:", len(features))
    print("DEBUG: X.shape:", X.shape)

    if X.empty:
        raise ValueError("Macierz X jest pusta po align_features.")

    X_scaled_arr = scaler.transform(
        X.values.astype(np.float64)
    )

    with torch.no_grad():
        x_t = torch.from_numpy(
            X_scaled_arr.astype(np.float32)
        )

        korekta = (
            model(x_t)
            .cpu()
            .numpy()
            .reshape(-1)
        )

    out = df_aligned.copy().reset_index(drop=True)

    out["korekta_modelu"] = korekta

    out["skorygowana_predkoscWiatru"] = np.clip(
        out["predkoscWiatru"] + out["korekta_modelu"],
        a_min=0,
        a_max=None,
    )

    final_cols = [
        "dataGodzinaCET",
        "lokalizacja",
        "predkoscWiatru",
        "korekta_modelu",
        "skorygowana_predkoscWiatru",
    ]

    final_df = out[final_cols].copy()

    output_path = Path(OUTPUT_CSV)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # WAŻNE:
    # Nie ustawiamy sep=";" ani decimal=",",
    # żeby pd.read_csv(path) w laczenie.py poprawnie widział kolumny.
    final_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Nie udało się utworzyć pliku: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise ValueError(
            f"Plik został utworzony, ale jest pusty: {output_path}"
        )

    test_df = pd.read_csv(
        output_path,
        nrows=1,
        encoding="utf-8-sig",
    )

    print(
        "DEBUG: kolumny po ponownym odczycie CSV:",
        test_df.columns.tolist(),
    )

    if "dataGodzinaCET" not in test_df.columns:
        raise ValueError(
            "Po zapisie CSV nadal nie widać kolumny dataGodzinaCET. "
            f"Kolumny: {test_df.columns.tolist()}"
        )

    print(
        f"[OK] Zapisano prognozę z korektą prędkości CSV: {output_path}"
    )

    print("Liczba zapisanych wierszy:", len(final_df))
    print("Rozmiar pliku bajty:", output_path.stat().st_size)
    print("KONIEC ML_Predkoscpred_fixed_full")


if __name__ == "__main__":
    main()