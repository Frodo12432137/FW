import pandas as pd
import os
import unicodedata
from datetime import datetime
import pytz

try:
    from charset_normalizer import from_path  
    _HAS_CHARSET_NORMALIZER = True
except Exception:
    _HAS_CHARSET_NORMALIZER = False

file_a = r"D:\app\weatherCorrection_fv\Kierunek\prognoza_kierunek_korekta.csv"
file_b = r"D:\app\weatherCorrection_fv\Temperatura\prognoza_temperatura_korekta.csv"
file_c = r"D:\app\weatherCorrection_fv\Predkosc\prognoza_predkosc_korekta.csv"
output_dir = r"D:\files\weatherCorrection_fv\Korekta\nowe"
output_basename = "prognoza_wiatr_korekta"

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

def remove_diacritics_str(s):
    if s is None or pd.isna(s):
        return s
    nk = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in nk if not unicodedata.combining(ch))

REV_REPLACE = {remove_diacritics_str(v).upper(): k for k, v in REPLACE_DICT.items()}

def detect_encoding(path):
    if _HAS_CHARSET_NORMALIZER:
        try:
            res = from_path(path).best()
            if res:
                return res.encoding
        except Exception:
            return None
    return None

def try_read_csv_with_detection(path):
    enc = detect_encoding(path)
    tried = []
    if enc:
        tried.append(enc)
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    for e in ("utf-8", "utf-8-sig", "cp1250", "iso-8859-2", "latin1"):
        if e in tried:
            continue
        try:
            return pd.read_csv(path, encoding=e)
        except Exception:
            continue
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise UnicodeError(f"Nie udało się wczytać plik: {path}. Błąd: {exc}")

def make_location_norm_series(s):
    ser_orig = s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    ser_orig = ser_orig.replace({"nan": pd.NA, "None": pd.NA})
    ser_norm = ser_orig.fillna(pd.NA).apply(lambda x: remove_diacritics_str(x) if pd.notna(x) else x)
    ser_norm = ser_norm.str.upper()
    return ser_orig, ser_norm

def read_and_prepare(path):
    df = try_read_csv_with_detection(path)
    df.columns = [c.strip() for c in df.columns]
    if "lokalizacja" not in df.columns and "lokalizacja.1" in df.columns:
        df = df.rename(columns={"lokalizacja.1": "lokalizacja"})
    if "dataGodzinaCET" not in df.columns:
        raise KeyError(f"Brak kolumny dataGodzinaCET w {path}")
    if "lokalizacja" not in df.columns:
        raise KeyError(f"Brak kolumny lokalizacja w {path}")
    df["dataGodzinaCET"] = pd.to_datetime(df["dataGodzinaCET"], errors="coerce")
    orig, norm = make_location_norm_series(df["lokalizacja"])
    df["lokalizacja_orig"] = orig
    df["lokalizacja_norm"] = norm
    return df

A = read_and_prepare(file_a)
B = read_and_prepare(file_b)
C = read_and_prepare(file_c)

print("DEBUG: kolumny w B:", B.columns.tolist())

def convert_cet_to_utc_series(df, cet_col="czasDanychZrodlaCET"):
    if cet_col in df.columns:
        ser = pd.to_datetime(df[cet_col], errors="coerce")
        try:
            ser = ser.dt.tz_localize("Europe/Warsaw", ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            try:
                ser = ser.dt.tz_localize("Europe/Warsaw", ambiguous="NaT", nonexistent="NaT")
            except Exception:
                ser = ser.dt.tz_localize("UTC")
        return ser.dt.tz_convert("UTC")
    if "czasDanychZrodlaUTC" in df.columns:
        ser = pd.to_datetime(df["czasDanychZrodlaUTC"], errors="coerce")
        if pd.api.types.is_datetime64tz_dtype(ser):
            return ser.dt.tz_convert("UTC")
        try:
            return ser.dt.tz_localize("UTC")
        except Exception:
            return pd.Series(pd.NaT, index=ser.index)
    return pd.Series(pd.NaT, index=df.index)

B["czasDanychZrodlaUTC_from_CET"] = convert_cet_to_utc_series(B, cet_col="czasDanychZrodlaCET")

key_cols_norm = ["dataGodzinaCET", "lokalizacja_norm"]
merged = A.merge(B, on=key_cols_norm, how="outer", suffixes=("_a", "_b"))
merged = merged.merge(C, on=key_cols_norm, how="outer", suffixes=("", "_c"))

def coalesce_datetime(df, candidates):
    out = pd.Series(pd.NaT, index=df.index)
    for c in candidates:
        if c in df.columns:
            ser = pd.to_datetime(df[c], errors="coerce")
            mask = out.isna() & ser.notna()
            out.loc[mask] = ser.loc[mask]
    return out

date_candidates = ["dataGodzinaCET", "dataGodzinaCET_b", "dataGodzinaCET_a", "dataGodzinaCET_c"]
merged["dataGodzinaCET"] = coalesce_datetime(merged, date_candidates)

def robust_ensure_datetime_make_utc(col, tz_name="Europe/Warsaw"):
    tz = pytz.timezone(tz_name)
    # jeśli już tz-aware -> convert
    if pd.api.types.is_datetime64tz_dtype(col):
        return col.dt.tz_convert("UTC")
    # zrób vektorową próbę najpierw
    ser = pd.to_datetime(col, errors="coerce")
    out = pd.Series(pd.NaT, index=ser.index, dtype="datetime64[ns, UTC]")
    try:
        loc = ser.dt.tz_localize(tz_name, ambiguous="infer", nonexistent="shift_forward")
        out.loc[loc.notna()] = loc.dt.tz_convert("UTC")
    except Exception:
        pass
    # obsłuż pozostałe wiersze per-wiersz
    missing_idx = out.isna()
    if missing_idx.any():
        for i in ser[missing_idx].index:
            val = ser.loc[i]
            if pd.isna(val):
                continue
            # spróbuj is_dst=False (zwykle standard time)
            try:
                l = tz.localize(val.to_pydatetime(), is_dst=False)
                out.loc[i] = pd.to_datetime(l.astimezone(pytz.UTC))
                continue
            except Exception:
                pass
            # spróbuj is_dst=True (letni czas)
            try:
                l = tz.localize(val.to_pydatetime(), is_dst=True)
                out.loc[i] = pd.to_datetime(l.astimezone(pytz.UTC))
                continue
            except Exception:
                pass
            # heurystyka: przesunięcie o +/-1h i dopasowanie
            try:
                l = tz.localize((val - pd.Timedelta(hours=1)).to_pydatetime(), is_dst=True)
                candidate = pd.to_datetime(l.astimezone(pytz.UTC)) + pd.Timedelta(hours=1)
                out.loc[i] = candidate
                continue
            except Exception:
                pass
            try:
                l = tz.localize((val + pd.Timedelta(hours=1)).to_pydatetime(), is_dst=False)
                candidate = pd.to_datetime(l.astimezone(pytz.UTC)) - pd.Timedelta(hours=1)
                out.loc[i] = candidate
                continue
            except Exception:
                out.loc[i] = pd.NaT
    # upewnij się że typ jest tz-aware UTC
    out = pd.to_datetime(out)
    if not pd.api.types.is_datetime64tz_dtype(out):
        out = out.dt.tz_localize("UTC")
    return out

# użyj nowej, odpornej funkcji
merged["dataGodzinaUTC"] = robust_ensure_datetime_make_utc(merged["dataGodzinaCET"])

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.NA

merged["skorygowana_predkoscWiatru"] = pick_col(merged, [
    "skorygowana_predkoscWiatru", "skorygowana_predkoscWiatru_a", "skorygowana_predkoscWiatru_b", "skorygowana_predkoscWiatru_c"
])
merged["skorygowana_temperatura"] = pick_col(merged, [
    "skorygowana_temperatura", "skorygowana_temperatura_a", "skorygowana_temperatura_b", "skorygowana_temperatura_c"
])
merged["skorygowana_kierunek"] = pick_col(merged, [
    "skorygowany_kierunekWiatru", "skorygowany_kierunekWiatru_a", "skorygowany_kierunekWiatru_b", "skorygowany_kierunekWiatru_c"
])

if "execId_b" in merged.columns:
    merged["execId"] = merged["execId_b"].where(merged["execId_b"].notna(), merged.get("execId", pd.NA))
else:
    merged["execId"] = merged.get("execId", pd.NA)

chosen_time_col = next((c for c in ("czasDanychZrodlaUTC_from_CET_b", "czasDanychZrodlaUTC_from_CET", "czasDanychZrodlaUTC_b", "czasDanychZrodlaUTC") if c in merged.columns), None)
if chosen_time_col:
    ser = pd.to_datetime(merged[chosen_time_col], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(ser):
        merged["czasDanychZrodlaUTC"] = ser.dt.tz_convert("UTC")
    else:
        # zastosuj robust konwersję zamiast prostego tz_localize
        merged["czasDanychZrodlaUTC"] = robust_ensure_datetime_make_utc(ser)
else:
    merged["czasDanychZrodlaUTC"] = pd.NaT

id_candidates = ["idPunkt_b", "idPunkt", "idPunkt_a", "id_punkt", "IDPUNKT", "idPunkt_c"]
for c in id_candidates:
    if c in merged.columns:
        merged["idPunkt"] = merged[c]
        break
else:
    merged["idPunkt"] = pd.NA

if merged["idPunkt"].isna().sum() > 0:
    if "idPunkt" in B.columns:
        b_keys = B[["dataGodzinaCET", "lokalizacja_norm", "idPunkt"]].drop_duplicates()
        merged = merged.merge(b_keys, on=["dataGodzinaCET", "lokalizacja_norm"], how="left", suffixes=("", "_fromB"))
        if "idPunkt_fromB" in merged.columns:
            merged["idPunkt"] = merged["idPunkt"].fillna(merged["idPunkt_fromB"])
            merged = merged.drop(columns=["idPunkt_fromB"])

def expand_abbrev_to_full_from_norm(row):
    for key in ("lokalizacja_norm", "lokalizacja_norm_b", "lokalizacja_norm_a", "lokalizacja_norm_c"):
        if key in row and pd.notna(row.get(key)):
            val_norm = row.get(key)
            break
    else:
        val_norm = pd.NA
    for key in ("lokalizacja", "lokalizacja_b", "lokalizacja_a", "lokalizacja_c", "lokalizacja_orig", "lokalizacja_orig_b"):
        if key in row and pd.notna(row.get(key)):
            val_orig = row.get(key)
            break
    else:
        val_orig = pd.NA
    if pd.isna(val_norm):
        return val_orig if pd.notna(val_orig) else pd.NA
    if val_norm in REV_REPLACE:
        return REV_REPLACE[val_norm]
    return val_orig if pd.notna(val_orig) else str(val_norm).title()

if "lokalizacja" not in merged.columns or merged["lokalizacja"].isna().all():
    merged["lokalizacja"] = pick_col(merged, ["lokalizacja_b", "lokalizacja_a", "lokalizacja_c", "lokalizacja_orig", "lokalizacja_orig_b"])

merged["lokalizacje"] = merged.apply(expand_abbrev_to_full_from_norm, axis=1)

exec_ts = pd.Timestamp.now(tz="UTC")
merged["data_wykonania"] = exec_ts

print("DEBUG: kolumny zawierające idPunkt w merged:", [c for c in merged.columns if "idpunkt" in c.lower()])
print("DEBUG: niepustych idPunkt:", merged["idPunkt"].notna().sum(), "/", len(merged))
if merged["idPunkt"].notna().sum() > 0:
    print(merged.loc[merged["idPunkt"].notna(), ["dataGodzinaCET", "lokalizacja_norm", "idPunkt"]].head(10))

final_cols = [
    "dataGodzinaCET",
    "dataGodzinaUTC",
    "lokalizacja",
    "lokalizacje",
    "skorygowana_predkoscWiatru",
    "skorygowana_temperatura",
    "skorygowana_kierunek",
    "data_wykonania",
    "execId",
    "czasDanychZrodlaCET",
    "czasDanychZrodlaUTC",
    "idPunkt",
]
final_cols_existing = [c for c in final_cols if c in merged.columns]
final = merged[final_cols_existing]

# diagnostyka braków dataGodzinaUTC i przykładowe wiersze dla daty zmiany czasu (26 października)
print("DEBUG: Braki dataGodzinaUTC:", merged["dataGodzinaUTC"].isna().sum())
# spróbuj wyświetlić wiersze dla 26 października roku występującego w danych (jeśli istnieje)
if merged["dataGodzinaCET"].notna().any():
    sample_year = merged["dataGodzinaCET"].dt.year.dropna().unique()
    if len(sample_year) > 0:
        y = int(sample_year.max())  # wybieramy najnowszy rok z danych
        dst_day = pd.to_datetime(f"{y}-10-26").date()
        mask_dst = merged["dataGodzinaCET"].dt.date == dst_day
        if mask_dst.any():
            print(f"DEBUG: Wiersze dla daty zmiany czasu {dst_day}:")
            display_df = merged.loc[mask_dst, ["dataGodzinaCET", "dataGodzinaUTC", "lokalizacja_norm", "idPunkt"]].sort_values("dataGodzinaCET")
            print(display_df.head(50).to_string(index=False))
        else:
            print(f"DEBUG: Brak wierszy dla {dst_day} w dataGodzinaCET.")
    else:
        print("DEBUG: Nie znaleziono roku w dataGodzinaCET do diagnostyki DST.")
else:
    print("DEBUG: Brak wartości dataGodzinaCET do diagnostyki DST.")

os.makedirs(output_dir, exist_ok=True)
file_time = exec_ts.strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"{output_basename}_{file_time}.csv")

merged["plik"] = os.path.splitext(os.path.basename(output_file))[0]
if "plik" not in final_cols:
    final_cols.append("plik")

final_cols_existing = [c for c in final_cols if c in merged.columns]
final = merged[final_cols_existing]

final.to_csv(output_file, index=False, encoding="utf-8-sig")
print("Zapisano:", output_file)
