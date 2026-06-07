from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyodbc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent

SQL_PROGNOZA_PATH = ROOT / "SQL" / "pogodaprognoza.sql"
SQL_WYKONANIE_PATH = ROOT / "SQL" / "wykonanie.sql"
OUTPUT_DIR = ROOT / "backtest_artifacts"

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
    "Jagniątkowo (Lake Ostrowo)": "LOS",
    "Jagniątkowo (Ląk Ostrów)": "LOS",
    "Lotnisko": "LOT",
    "Malbork (Koniecwałd)": "MAL",
    "Malbork (Konieczwałd)": "MAL",
    "Malbork (KoniecwaÅ‚d)": "MAL",
    "Pelplin": "PEL",
    "Resko I": "RES",
    "Resko II": "RES2",
    "Rybice": "RYB",
    "Rybitwy": "RYB",
    "Skoczykłody": "SKO",
    "Starza": "STA",
    "Wojciechowo": "WOJ",
    "Zalesie": "ZAL",
    "Żuromin": "ZUR",
}

ROLLING_FEATURES_15MIN = [
    ("kierunekWiatru", "kierunek_srednia16", 90),
    ("predkoscWiatru", "predkosc_srednia16", 90),
    ("temperatura", "temperatura_srednia16", 90),
    ("kierunekWiatru", "kierunek_srednia48", 96),
    ("predkoscWiatru", "predkosc_srednia48", 96),
    ("temperatura", "temperatura_srednia48", 96),
    ("kierunekWiatru", "kierunek_srednia54", 100),
    ("predkoscWiatru", "predkosc_srednia54", 100),
    ("temperatura", "temperatura_srednia54", 100),
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

ROLLING_FEATURES_HOURLY = [
    ("kierunekWiatru", "kierunek_srednia4", 12),
    ("predkoscWiatru", "predkosc_srednia12", 4),
    ("temperatura", "temperatura_srednia24", 24),
]

TARGETS = {
    "predkosc": {
        "forecast_col": "predkoscWiatru",
        "actual_col": "predkoscWiatruLokalizacja_wykonanie",
        "prediction_col": "skorygowana_predkoscWiatru",
        "metric_prefix": "speed",
        "angular": False,
    },
    "temperatura": {
        "forecast_col": "temperatura",
        "actual_col": "temperaturaLokalizacja_wykonanie",
        "prediction_col": "skorygowana_temperatura",
        "metric_prefix": "temperature",
        "angular": False,
    },
    "kierunek": {
        "forecast_col": "kierunekWiatru",
        "actual_col": "kierunekWiatruLokalizacja_wykonanie",
        "prediction_col": "skorygowany_kierunekWiatru",
        "metric_prefix": "direction",
        "angular": True,
    },
}


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


@dataclass
class FoldResult:
    target: str
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    test_rows: int
    baseline_mae: float
    corrected_mae: float
    baseline_rmse: float
    corrected_rmse: float
    baseline_bias: float
    corrected_bias: float
    improvement_mae_pct: float
    improvement_rmse_pct: float


def query2df(sql_path: Path, connection_string: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    sql_query = sql_path.read_text(encoding="utf-8")
    replacements = {
        "start_date": start_date,
        "end_date": end_date,
    }
    for key, value in replacements.items():
        if value:
            sql_query = re.sub(
                rf"DECLARE\s+@{key}\s+DATETIME\s*=\s*'[^']*'",
                f"DECLARE @{key} DATETIME = '{value}'",
                sql_query,
                flags=re.IGNORECASE,
            )
            sql_query = re.sub(
                rf"SET\s+@{key}\s*=\s*'[^']*'",
                f"SET @{key} = '{value}'",
                sql_query,
                flags=re.IGNORECASE,
            )

    conn = pyodbc.connect(connection_string)
    try:
        df = pd.read_sql_query(sql_query, conn)
    finally:
        conn.close()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_location(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "punkt" in df.columns:
        df = df.rename(columns={"punkt": "lokalizacja"})
    if "lokalizacja" not in df.columns:
        raise ValueError("Brak kolumny 'punkt' albo 'lokalizacja'.")
    df["lokalizacja"] = df["lokalizacja"].astype(str).str.strip().replace(REPLACE_DICT)
    return df


def convert_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_source_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.prognoza_csv and args.wykonanie_csv:
        prognoza = pd.read_csv(args.prognoza_csv)
        wykonanie = pd.read_csv(args.wykonanie_csv)
        return prognoza, wykonanie

    prognoza = query2df(
        Path(args.sql_prognoza),
        args.conn_prognoza,
        args.start_date,
        args.end_date,
    )
    wykonanie = query2df(
        Path(args.sql_wykonanie),
        args.conn_wykonanie,
        args.start_date,
        args.end_date,
    )
    return prognoza, wykonanie


def prepare_dataset(
    prognoza_raw: pd.DataFrame,
    wykonanie_raw: pd.DataFrame,
    target_name: str,
    use_15min: bool,
) -> pd.DataFrame:
    cfg = TARGETS[target_name]
    floor_unit = "15min" if use_15min else "h"
    valid_minutes = [0, 15, 30, 45] if use_15min else [0]
    rolling_features = ROLLING_FEATURES_15MIN if use_15min else ROLLING_FEATURES_HOURLY
    lag_shifts = list(range(4, 300)) if use_15min else list(range(1, 25))

    prognoza = normalize_location(prognoza_raw)
    wykonanie = normalize_location(wykonanie_raw)

    prognoza["dataGodzinaCET"] = pd.to_datetime(prognoza["dataGodzinaCET"], errors="coerce").dt.floor(floor_unit)

    if "DataiCzasOdczytu" in wykonanie.columns:
        wykonanie["DataiCzasOdczytu"] = pd.to_datetime(wykonanie["DataiCzasOdczytu"], errors="coerce")
        wykonanie["dataGodzinaCET"] = wykonanie["DataiCzasOdczytu"].dt.floor(floor_unit)
        wykonanie = wykonanie[
            wykonanie["DataiCzasOdczytu"].dt.minute.isin(valid_minutes)
            & (wykonanie["DataiCzasOdczytu"].dt.second == 0)
        ]
    elif "dataGodzinaCET" in wykonanie.columns:
        wykonanie["dataGodzinaCET"] = pd.to_datetime(wykonanie["dataGodzinaCET"], errors="coerce").dt.floor(floor_unit)
    else:
        raise ValueError("Brak daty wykonania: DataiCzasOdczytu albo dataGodzinaCET.")

    numeric_cols = [
        "predkoscWiatru",
        "predkoscWiatruLokalizacja_wykonanie",
        "temperaturaLokalizacja_wykonanie",
        "temperatura",
        "kierunekWiatruLokalizacja_wykonanie",
        "kierunekWiatru",
    ]
    prognoza = convert_numeric(prognoza, numeric_cols)
    wykonanie = convert_numeric(wykonanie, numeric_cols)

    merged = pd.merge(
        prognoza,
        wykonanie,
        on=["dataGodzinaCET", "lokalizacja"],
        how="inner",
        suffixes=("", "_wykonanie_src"),
    )
    merged = merged.sort_values(["lokalizacja", "dataGodzinaCET"]).reset_index(drop=True)

    for base_col, roll_col, window in rolling_features:
        if base_col in merged.columns:
            merged[roll_col] = merged.groupby("lokalizacja")[base_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

    merged["godzina"] = merged["dataGodzinaCET"].dt.hour
    merged["miesiac"] = merged["dataGodzinaCET"].dt.month

    if cfg["angular"]:
        merged["diff"] = angle_diff(merged[cfg["actual_col"]], merged[cfg["forecast_col"]])
    else:
        merged["diff"] = merged[cfg["actual_col"]] - merged[cfg["forecast_col"]]

    for lag in lag_shifts:
        merged[f"diff_lag_{lag}"] = merged.groupby("lokalizacja")["diff"].shift(lag)

    roll_names = [name for _, name, _ in rolling_features]
    required = [
        "dataGodzinaCET",
        "lokalizacja",
        "predkoscWiatru",
        "kierunekWiatru",
        "temperatura",
        cfg["forecast_col"],
        cfg["actual_col"],
        "diff",
        *roll_names,
    ]
    required = list(dict.fromkeys(required))
    merged = merged.dropna(subset=[col for col in required if col in merged.columns])
    return merged


def feature_columns(df: pd.DataFrame, use_15min: bool) -> list[str]:
    rolling_features = ROLLING_FEATURES_15MIN if use_15min else ROLLING_FEATURES_HOURLY
    roll_names = [name for _, name, _ in rolling_features]
    lag_cols = sorted(
        [c for c in df.columns if c.startswith("diff_lag_")],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    loc_cols = sorted([c for c in df.columns if c.startswith("lokalizacja_")])
    return [
        "predkoscWiatru",
        "kierunekWiatru",
        "temperatura",
        "godzina",
        "miesiac",
        *roll_names,
        *lag_cols,
        *loc_cols,
    ]


def encode_features(df: pd.DataFrame, expected_features: list[str] | None = None, use_15min: bool = True):
    encoded = pd.get_dummies(df, columns=["lokalizacja"], drop_first=True, dtype=int)
    if expected_features is None:
        cols = feature_columns(encoded, use_15min)
    else:
        cols = expected_features
        for col in cols:
            if col not in encoded.columns:
                encoded[col] = 0.0

    X = encoded[cols].copy()
    mask = ~X.isna().any(axis=1)
    return X.loc[mask], df.loc[mask].copy(), cols


def angle_diff(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    return (actual - forecast + 180) % 360 - 180


def normalize_angle(value):
    return (value % 360 + 360) % 360


def train_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_15min: bool,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict, NeuralNet, StandardScaler, list[str]]:
    X_train, train_aligned, x_cols = encode_features(train_df, use_15min=use_15min)
    X_test, test_aligned, _ = encode_features(test_df, expected_features=x_cols, use_15min=use_15min)

    y_train = train_aligned["diff"].astype(float)

    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(X_train)
    x_test_np = scaler.transform(X_test)

    train_mask = ~np.isnan(x_train_np).any(axis=1) & ~y_train.isna().to_numpy()
    x_train_np = x_train_np[train_mask]
    y_train_np = y_train.to_numpy()[train_mask]

    test_mask = ~np.isnan(x_test_np).any(axis=1)
    x_test_np = x_test_np[test_mask]
    test_aligned = test_aligned.iloc[test_mask].copy()

    if len(x_train_np) == 0 or len(x_test_np) == 0:
        raise ValueError("Puste dane po odfiltrowaniu NaN w foldzie.")

    x_train_t = torch.from_numpy(x_train_np).float().to(device)
    y_train_t = torch.from_numpy(y_train_np.reshape(-1, 1)).float().to(device)

    model = NeuralNet(input_dim=x_train_np.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(x_train_t.shape[0], device=device)
        for start in range(0, x_train_t.shape[0], batch_size):
            idx = permutation[start : start + batch_size]
            outputs = model(x_train_t[idx])
            loss = criterion(outputs, y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_diff = model(torch.from_numpy(x_test_np).float().to(device)).cpu().numpy().reshape(-1)

    test_aligned["pred_diff"] = pred_diff
    return test_aligned, {"train_loss": float(loss.item())}, model, scaler, x_cols


def metrics(actual: pd.Series, forecast: pd.Series, angular: bool) -> dict[str, float]:
    if angular:
        err = angle_diff(actual, forecast)
    else:
        err = actual - forecast
    err = pd.to_numeric(err, errors="coerce").dropna()
    if err.empty:
        return {"mae": math.nan, "rmse": math.nan, "bias": math.nan}
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
    }


def make_windows(
    df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
    min_train_rows: int,
):
    start = df["dataGodzinaCET"].min()
    last = df["dataGodzinaCET"].max()
    fold = 1
    while True:
        train_start = start
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_start > last:
            break
        train_mask = (df["dataGodzinaCET"] >= train_start) & (df["dataGodzinaCET"] < train_end)
        test_mask = (df["dataGodzinaCET"] >= test_start) & (df["dataGodzinaCET"] < test_end)
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()
        if len(train_df) >= min_train_rows and not test_df.empty:
            yield fold, train_start, train_end, test_start, test_end, train_df, test_df
            fold += 1
        start = start + pd.Timedelta(days=step_days)
        if start + pd.Timedelta(days=train_days) > last:
            break


def run_target(target_name: str, source_prognoza: pd.DataFrame, source_wykonanie: pd.DataFrame, args: argparse.Namespace):
    cfg = TARGETS[target_name]
    target_dir = Path(args.output_dir) / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(source_prognoza, source_wykonanie, target_name, args.use_15min)
    if args.max_rows:
        df = df.sort_values("dataGodzinaCET").tail(args.max_rows).copy()

    all_predictions = []
    fold_results = []
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    for fold, train_start, train_end, test_start, test_end, train_df, test_df in make_windows(
        df,
        args.train_days,
        args.test_days,
        args.step_days,
        args.min_train_rows,
    ):
        print(
            f"[{target_name}] fold={fold} train={train_start:%Y-%m-%d}->{train_end:%Y-%m-%d} "
            f"test={test_start:%Y-%m-%d}->{test_end:%Y-%m-%d} rows={len(train_df)}/{len(test_df)}"
        )

        predicted, train_info, model, scaler, x_cols = train_fold(
            train_df=train_df,
            test_df=test_df,
            use_15min=args.use_15min,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=device,
        )

        if cfg["angular"]:
            predicted[cfg["prediction_col"]] = normalize_angle(predicted[cfg["forecast_col"]] + predicted["pred_diff"])
        else:
            predicted[cfg["prediction_col"]] = predicted[cfg["forecast_col"]] + predicted["pred_diff"]

        base = metrics(predicted[cfg["actual_col"]], predicted[cfg["forecast_col"]], cfg["angular"])
        corr = metrics(predicted[cfg["actual_col"]], predicted[cfg["prediction_col"]], cfg["angular"])

        fold_result = FoldResult(
            target=target_name,
            fold=fold,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_rows=len(train_df),
            test_rows=len(predicted),
            baseline_mae=base["mae"],
            corrected_mae=corr["mae"],
            baseline_rmse=base["rmse"],
            corrected_rmse=corr["rmse"],
            baseline_bias=base["bias"],
            corrected_bias=corr["bias"],
            improvement_mae_pct=100 * (base["mae"] - corr["mae"]) / base["mae"] if base["mae"] else math.nan,
            improvement_rmse_pct=100 * (base["rmse"] - corr["rmse"]) / base["rmse"] if base["rmse"] else math.nan,
        )
        fold_results.append(fold_result.__dict__ | train_info)

        predicted["target"] = target_name
        predicted["fold"] = fold
        keep_cols = [
            "target",
            "fold",
            "dataGodzinaCET",
            "lokalizacja",
            cfg["forecast_col"],
            cfg["actual_col"],
            "pred_diff",
            cfg["prediction_col"],
        ]
        all_predictions.append(predicted[keep_cols].copy())

        fold_dir = target_dir / f"fold_{fold:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fold_dir / "model.pth")
        joblib.dump(scaler, fold_dir / "scaler.joblib")
        (fold_dir / "features.json").write_text(json.dumps({"x_cols": x_cols}, ensure_ascii=False, indent=2), encoding="utf-8")

    if not fold_results:
        raise RuntimeError(f"Brak foldów dla targetu {target_name}. Zwiększ zakres dat albo zmniejsz train_days/min_train_rows.")

    metrics_df = pd.DataFrame(fold_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(target_dir / "metrics_by_fold.csv", index=False)
    predictions_df.to_csv(target_dir / "predictions.csv", index=False)

    summary = {
        "target": target_name,
        "rows": int(len(df)),
        "folds": int(len(metrics_df)),
        "baseline_mae_mean": float(metrics_df["baseline_mae"].mean()),
        "corrected_mae_mean": float(metrics_df["corrected_mae"].mean()),
        "baseline_rmse_mean": float(metrics_df["baseline_rmse"].mean()),
        "corrected_rmse_mean": float(metrics_df["corrected_rmse"].mean()),
        "improvement_mae_pct_mean": float(metrics_df["improvement_mae_pct"].mean()),
        "improvement_rmse_pct_mean": float(metrics_df["improvement_rmse_pct"].mean()),
    }
    (target_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Chronologiczny backtest modeli korekty pogody.")
    parser.add_argument("--targets", nargs="+", default=["predkosc", "temperatura", "kierunek"], choices=sorted(TARGETS))
    parser.add_argument("--start-date", default=None, help="Np. 2024-01-01 00:00:00. Podmienia @start_date w SQL.")
    parser.add_argument("--end-date", default=None, help="Np. 2026-05-31 23:45:00. Podmienia @end_date w SQL.")
    parser.add_argument("--sql-prognoza", default=str(SQL_PROGNOZA_PATH))
    parser.add_argument("--sql-wykonanie", default=str(SQL_WYKONANIE_PATH))
    parser.add_argument("--conn-prognoza", default=CONN_PROGNOZA)
    parser.add_argument("--conn-wykonanie", default=CONN_WYKONANIE)
    parser.add_argument("--prognoza-csv", default=None, help="Opcjonalnie: CSV zamiast SQL dla prognozy.")
    parser.add_argument("--wykonanie-csv", default=None, help="Opcjonalnie: CSV zamiast SQL dla wykonania.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--use-15min", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--min-train-rows", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.009)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default=None, help="cpu, cuda albo zostaw puste dla auto.")
    parser.add_argument("--max-rows", type=int, default=None, help="Szybki smoke test na ostatnich N wierszach.")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    prognoza, wykonanie = load_source_data(args)
    summaries = []
    started_at = datetime.now().isoformat(timespec="seconds")

    for target in args.targets:
        summaries.append(run_target(target, prognoza, wykonanie, args))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(Path(args.output_dir) / "summary.csv", index=False)
    run_meta = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "summaries": summaries,
    }
    (Path(args.output_dir) / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
