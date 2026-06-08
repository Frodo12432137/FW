from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyodbc
except ImportError:
    pyodbc = None


ROOT = Path(__file__).resolve().parent
SQL_PROGNOZA_PATH = ROOT / "SQL" / "pogodaprognoza.sql"
SQL_WYKONANIE_PATH = ROOT / "SQL" / "wykonanie.sql"
OUTPUT_DIR = ROOT / "weather_difference_coefficients"

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
    "Jagniątkowo": "LOS",
    "Lotnisko": "LOT",
    "Malbork (Koniecwałd)": "MAL",
    "Malbork (Konieczwałd)": "MAL",
    "Malbork": "MAL",
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

TARGETS = {
    "predkosc": {
        "forecast_col": "predkoscWiatru",
        "actual_col": "predkoscWiatruLokalizacja_wykonanie",
        "angular": False,
    },
    "temperatura": {
        "forecast_col": "temperatura",
        "actual_col": "temperaturaLokalizacja_wykonanie",
        "angular": False,
    },
    "kierunek": {
        "forecast_col": "kierunekWiatru",
        "actual_col": "kierunekWiatruLokalizacja_wykonanie",
        "angular": True,
    },
}

ID_PUNKT_BY_LOCATION = {
    "LOS": 31378,
    "KAR": 31372,
    "MAL": 31380,
    "KIS": 31374,
    "GAL": 31370,
    "RES": 31382,
    "WOJ": 31387,
    "PEL": 31381,
    "ZUR": 31389,
    "KAM": 31371,
    "LOT": 31379,
    "RES2": 31383,
    "KRW": 31377,
    "KIS2": 31375,
    "KAR2": 31373,
    "RYB": 31384,
    "STA": 31386,
    "SKO": 31385,
    "ZAL": 31388,
    "DZW": 31524,
}

SPEED_BINS = list(range(0, 38, 2))
DIRECTION_BINS = list(range(0, 361, 20))
TEMPERATURE_BINS = list(range(-32, 40, 4))


def query2df(sql_path: Path, connection_string: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if pyodbc is None:
        raise ImportError("Brak pakietu pyodbc. Zainstaluj go albo uzyj --prognoza-csv i --wykonanie-csv.")

    sql = sql_path.read_text(encoding="utf-8")

    if start_date:
        sql = re.sub(
            r"DECLARE\s+@start_date\s+DATETIME\s*=\s*'[^']*'",
            f"DECLARE @start_date DATETIME = '{start_date}'",
            sql,
            flags=re.IGNORECASE,
        )
        sql = re.sub(
            r"TRY_CONVERT\(date,\s*\[Data\]\)\s*>=\s*'[^']*'",
            f"TRY_CONVERT(date, [Data]) >= '{start_date[:10]}'",
            sql,
            flags=re.IGNORECASE,
        )

    if end_date:
        sql = re.sub(
            r"DECLARE\s+@end_date\s+DATETIME\s*=\s*'[^']*'",
            f"DECLARE @end_date DATETIME = '{end_date}'",
            sql,
            flags=re.IGNORECASE,
        )
        sql = re.sub(
            r"TRY_CONVERT\(date,\s*\[Data\]\)\s*<=\s*'[^']*'",
            f"TRY_CONVERT(date, [Data]) <= '{end_date[:10]}'",
            sql,
            flags=re.IGNORECASE,
        )

    conn = pyodbc.connect(connection_string)
    try:
        df = pd.read_sql_query(sql, conn)
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
        return pd.read_csv(args.prognoza_csv), pd.read_csv(args.wykonanie_csv)

    prognoza = query2df(Path(args.sql_prognoza), args.conn_prognoza, args.start_date, args.end_date)
    wykonanie = query2df(Path(args.sql_wykonanie), args.conn_wykonanie, args.start_date, args.end_date)
    return prognoza, wykonanie


def prepare_joined_data(prognoza_raw: pd.DataFrame, wykonanie_raw: pd.DataFrame, use_15min: bool) -> pd.DataFrame:
    floor_unit = "15min" if use_15min else "h"
    valid_minutes = [0, 15, 30, 45] if use_15min else [0]

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

    joined = pd.merge(
        prognoza,
        wykonanie,
        on=["dataGodzinaCET", "lokalizacja"],
        how="inner",
        suffixes=("", "_wykonanie_src"),
    )
    joined = joined.sort_values(["lokalizacja", "dataGodzinaCET"]).reset_index(drop=True)
    joined["rok"] = joined["dataGodzinaCET"].dt.year
    joined["miesiac"] = joined["dataGodzinaCET"].dt.month
    joined["rok_miesiac"] = joined["dataGodzinaCET"].dt.to_period("M").astype(str)
    joined["godzina"] = joined["dataGodzinaCET"].dt.hour
    return joined


def angle_diff(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    return (actual - forecast + 180) % 360 - 180


def floor_to_bin(series: pd.Series, step: int, min_value: int, max_value: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    binned = np.floor(values / step) * step
    binned = binned.clip(lower=min_value, upper=max_value)
    return binned.astype("Int64")


def add_value_bins(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.copy()
    df["poziom_wietrznosci"] = floor_to_bin(df["predkoscWiatru"], step=2, min_value=0, max_value=36)
    df["poziom_temperatury"] = floor_to_bin(df["temperatura"], step=4, min_value=-32, max_value=36)
    direction = pd.to_numeric(df["kierunekWiatru"], errors="coerce") % 360
    df["poziom_kierunku"] = floor_to_bin(direction, step=20, min_value=0, max_value=360)
    df.loc[pd.to_numeric(df["kierunekWiatru"], errors="coerce") == 360, "poziom_kierunku"] = 360
    df["idPunkt"] = df["lokalizacja"].map(ID_PUNKT_BY_LOCATION).astype("Int64")
    return df


def regression_params(forecast: pd.Series, actual: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(forecast, errors="coerce")
    y = pd.to_numeric(actual, errors="coerce")
    mask = x.notna() & y.notna()
    x = x.loc[mask].astype(float)
    y = y.loc[mask].astype(float)
    if len(x) < 2 or np.isclose(x.var(ddof=0), 0):
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}

    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    ss_res = float(np.square(y - predicted).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def summarize_group(group: pd.DataFrame, target_name: str) -> dict[str, float | int | str]:
    cfg = TARGETS[target_name]
    forecast = pd.to_numeric(group[cfg["forecast_col"]], errors="coerce")
    actual = pd.to_numeric(group[cfg["actual_col"]], errors="coerce")
    mask = forecast.notna() & actual.notna()
    forecast = forecast.loc[mask]
    actual = actual.loc[mask]

    if cfg["angular"]:
        diff = angle_diff(actual, forecast)
    else:
        diff = actual - forecast

    reg = regression_params(forecast, actual)
    forecast_mean = float(forecast.mean()) if len(forecast) else np.nan
    actual_mean = float(actual.mean()) if len(actual) else np.nan
    multiplier = actual_mean / forecast_mean if forecast_mean and not np.isclose(forecast_mean, 0) else np.nan

    return {
        "target": target_name,
        "liczba_obserwacji": int(len(diff)),
        "prognoza_srednia": forecast_mean,
        "wykonanie_srednia": actual_mean,
        "roznica_srednia": float(diff.mean()) if len(diff) else np.nan,
        "roznica_mediana": float(diff.median()) if len(diff) else np.nan,
        "roznica_std": float(diff.std(ddof=0)) if len(diff) else np.nan,
        "mae": float(diff.abs().mean()) if len(diff) else np.nan,
        "rmse": float(np.sqrt(np.square(diff).mean())) if len(diff) else np.nan,
        "roznica_p10": float(diff.quantile(0.10)) if len(diff) else np.nan,
        "roznica_p90": float(diff.quantile(0.90)) if len(diff) else np.nan,
        "wspolczynnik_mnoznik_srednich": float(multiplier) if pd.notna(multiplier) else np.nan,
        "regresja_slope": reg["slope"],
        "regresja_intercept": reg["intercept"],
        "regresja_r2": reg["r2"],
    }


def summarize_error(group: pd.DataFrame, target_name: str) -> dict[str, float | int | str]:
    summary = summarize_group(group, target_name)
    return {
        "target": target_name,
        "liczba_obserwacji": summary["liczba_obserwacji"],
        "bias": summary["roznica_srednia"],
        "mae": summary["mae"],
        "rmse": summary["rmse"],
        "prognoza_srednia": summary["prognoza_srednia"],
        "wykonanie_srednia": summary["wykonanie_srednia"],
    }


def build_value_level_coefficients(joined: pd.DataFrame, min_count: int) -> dict[str, pd.DataFrame]:
    df = add_value_bins(joined)
    specs = {
        "predkosc": {
            "bin_col": "poziom_wietrznosci",
            "bins": SPEED_BINS,
            "coefficient_col": "wsp_v",
            "coefficient_source": "wspolczynnik_mnoznik_srednich",
        },
        "kierunek": {
            "bin_col": "poziom_kierunku",
            "bins": DIRECTION_BINS,
            "coefficient_col": "wsp_k",
            "coefficient_source": "roznica_srednia",
        },
        "temperatura": {
            "bin_col": "poziom_temperatury",
            "bins": TEMPERATURE_BINS,
            "coefficient_col": "wsp_t",
            "coefficient_source": "roznica_srednia",
        },
    }

    outputs = {}
    for target_name, spec in specs.items():
        records = []
        group_cols = ["idPunkt", "lokalizacja", spec["bin_col"]]
        for keys, group in df.groupby(group_cols, dropna=False):
            base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            summary = summarize_group(group, target_name)
            coefficient = summary[spec["coefficient_source"]]
            records.append(
                {
                    **base,
                    "target": target_name,
                    spec["coefficient_col"]: coefficient,
                    **summary,
                }
            )

        long_df = pd.DataFrame(records)
        if long_df.empty:
            outputs[f"{target_name}_long"] = long_df
            outputs[f"{target_name}_wide"] = long_df
            continue

        long_df = long_df[long_df["liczba_obserwacji"] >= min_count].copy()
        long_df = long_df.sort_values(["idPunkt", spec["bin_col"]]).reset_index(drop=True)
        long_df["p"] = long_df[spec["bin_col"]]

        wide_df = long_df.pivot_table(
            index=["idPunkt", "lokalizacja"],
            columns=spec["bin_col"],
            values=spec["coefficient_col"],
            aggfunc="first",
        )
        wide_df = wide_df.reindex(columns=spec["bins"])
        wide_df = wide_df.reset_index().sort_values(["idPunkt", "lokalizacja"]).reset_index(drop=True)
        wide_df.columns = [str(c) if isinstance(c, (int, np.integer)) else c for c in wide_df.columns]

        outputs[f"{target_name}_long"] = long_df
        outputs[f"{target_name}_wide"] = wide_df

    return outputs


def build_forecast_error_report(joined: pd.DataFrame, min_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    location_records = []
    for location, group in joined.groupby("lokalizacja", dropna=False):
        for target_name in TARGETS:
            location_records.append(
                {
                    "lokalizacja": location,
                    "idPunkt": ID_PUNKT_BY_LOCATION.get(location, pd.NA),
                    **summarize_error(group, target_name),
                }
            )

    overall_records = []
    for target_name in TARGETS:
        overall_records.append(
            {
                "lokalizacja": "ALL",
                "idPunkt": pd.NA,
                **summarize_error(joined, target_name),
            }
        )

    by_location = pd.DataFrame(location_records)
    overall = pd.DataFrame(overall_records)
    by_location = by_location[by_location["liczba_obserwacji"] >= min_count].copy()
    by_location = by_location.sort_values(["target", "lokalizacja"]).reset_index(drop=True)
    return by_location, overall


def build_coefficients(joined: pd.DataFrame, group_mode: str, min_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_mode == "month_of_year":
        group_cols = ["miesiac", "lokalizacja", "godzina"]
    elif group_mode == "year_month":
        group_cols = ["rok_miesiac", "lokalizacja", "godzina"]
    else:
        raise ValueError(f"Nieznany group_mode: {group_mode}")

    records = []
    for keys, group in joined.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for target_name in TARGETS:
            record = {**base, **summarize_group(group, target_name)}
            records.append(record)

    long_df = pd.DataFrame(records)
    long_df = long_df[long_df["liczba_obserwacji"] >= min_count].copy()
    long_df = long_df.sort_values([*group_cols, "target"]).reset_index(drop=True)

    wide_df = long_df.pivot_table(
        index=group_cols,
        columns="target",
        values=[
            "liczba_obserwacji",
            "prognoza_srednia",
            "wykonanie_srednia",
            "roznica_srednia",
            "roznica_mediana",
            "roznica_std",
            "mae",
            "rmse",
            "roznica_p10",
            "roznica_p90",
            "wspolczynnik_mnoznik_srednich",
            "regresja_slope",
            "regresja_intercept",
            "regresja_r2",
        ],
        aggfunc="first",
    )
    wide_df.columns = [f"{target}_{metric}" for metric, target in wide_df.columns]
    wide_df = wide_df.reset_index().sort_values(group_cols).reset_index(drop=True)
    return long_df, wide_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wylicza parametry roznicy prognoza vs wykonanie per miesiac, lokalizacja i godzina."
    )
    parser.add_argument("--start-date", default="2025-08-01 00:00:00", help="Domyslnie od sierpnia 2025, zgodnie z mailem.")
    parser.add_argument("--end-date", default=None, help="Np. 2026-10-30 23:45:00.")
    parser.add_argument("--sql-prognoza", default=str(SQL_PROGNOZA_PATH))
    parser.add_argument("--sql-wykonanie", default=str(SQL_WYKONANIE_PATH))
    parser.add_argument("--conn-prognoza", default=CONN_PROGNOZA)
    parser.add_argument("--conn-wykonanie", default=CONN_WYKONANIE)
    parser.add_argument("--prognoza-csv", default=None, help="Opcjonalnie: CSV zamiast SQL dla prognozy.")
    parser.add_argument("--wykonanie-csv", default=None, help="Opcjonalnie: CSV zamiast SQL dla wykonania.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--group-mode", choices=["month_of_year", "year_month"], default="month_of_year")
    parser.add_argument(
        "--legacy-time-groups",
        action="store_true",
        help="Dodatkowo zapisuje stary uklad miesiac/lokalizacja/godzina.",
    )
    parser.add_argument("--min-count", type=int, default=10, help="Minimalna liczba obserwacji w grupie.")
    parser.add_argument("--use-15min", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prognoza, wykonanie = load_source_data(args)
    joined = prepare_joined_data(prognoza, wykonanie, args.use_15min)
    value_level_outputs = build_value_level_coefficients(joined, args.min_count)
    error_by_location, error_overall = build_forecast_error_report(joined, args.min_count)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_paths = {}
    for name, df in value_level_outputs.items():
        path = output_dir / f"{name}_{timestamp}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        output_paths[name] = str(path)

    error_by_location_path = output_dir / f"blad_prognozy_fw_od_2025_08_lokalizacje_{timestamp}.csv"
    error_overall_path = output_dir / f"blad_prognozy_fw_od_2025_08_ogolem_{timestamp}.csv"
    joined_sample_path = output_dir / f"polaczone_dane_sample_{timestamp}.csv"
    metadata_path = output_dir / f"metadata_{timestamp}.json"

    error_by_location.to_csv(error_by_location_path, index=False, encoding="utf-8-sig")
    error_overall.to_csv(error_overall_path, index=False, encoding="utf-8-sig")
    joined.head(1000).to_csv(joined_sample_path, index=False, encoding="utf-8-sig")

    if args.legacy_time_groups:
        long_df, wide_df = build_coefficients(joined, args.group_mode, args.min_count)
        long_path = output_dir / f"legacy_wspolczynniki_roznic_long_{args.group_mode}_{timestamp}.csv"
        wide_path = output_dir / f"legacy_wspolczynniki_roznic_wide_{args.group_mode}_{timestamp}.csv"
        long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
        wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")
        output_paths["legacy_long"] = str(long_path)
        output_paths["legacy_wide"] = str(wide_path)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "joined_rows": int(len(joined)),
        "error_by_location_rows": int(len(error_by_location)),
        "error_overall_rows": int(len(error_overall)),
        "outputs": {
            **output_paths,
            "error_by_location": str(error_by_location_path),
            "error_overall": str(error_overall_path),
            "joined_sample": str(joined_sample_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Polaczone wiersze: {len(joined)}")
    for name, path in output_paths.items():
        print(f"Zapisano {name}: {path}")
    print(f"Zapisano blad per lokalizacja: {error_by_location_path}")
    print(f"Zapisano blad ogolem: {error_overall_path}")
    print(f"Zapisano sample danych polaczonych: {joined_sample_path}")


if __name__ == "__main__":
    main()
