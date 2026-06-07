# Backtest modeli korekty pogody

Ten plik opisuje uruchomienie `backtest_models.py` na mocniejszej maszynie z dostępem do SQL Servera PGE.

## Co robi skrypt

- pobiera prognozę i wykonanie z SQL albo z plików CSV,
- buduje te same cechy co obecne modele: rollingi, lagi `diff_lag_*`, miesiąc, godzinę i one-hot lokalizacji,
- robi chronologiczny walk-forward backtest zamiast losowego `train_test_split`,
- dla każdego folda trenuje model tylko na przeszłości i testuje na kolejnym oknie,
- zapisuje metryki bazowej prognozy oraz prognozy po korekcie.

## Instalacja

```bash
cd /sciezka/do/FW
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-backtest.txt
```

Na Windows musi być zainstalowany ODBC Driver 17 albo 18 for SQL Server. Jeżeli na służbowym komputerze jest tylko Driver 18, zmień `DRIVER={ODBC Driver 17 for SQL Server}` w `backtest_models.py` na `DRIVER={ODBC Driver 18 for SQL Server}`.

## Pełny backtest z SQL

```bash
python backtest_models.py ^
  --start-date "2024-01-01 00:00:00" ^
  --end-date "2026-05-31 23:45:00" ^
  --train-days 365 ^
  --test-days 30 ^
  --step-days 30 ^
  --epochs 300 ^
  --batch-size 8192
```

Domyślnie uruchamiane są trzy targety: `predkosc`, `temperatura`, `kierunek`.

## Szybki test techniczny

Na początku warto puścić krótszą wersję, żeby sprawdzić połączenie, zależności i format danych:

```bash
python backtest_models.py ^
  --targets predkosc ^
  --start-date "2025-01-01 00:00:00" ^
  --end-date "2025-04-30 23:45:00" ^
  --train-days 60 ^
  --test-days 7 ^
  --step-days 7 ^
  --epochs 5 ^
  --min-train-rows 1000
```

## Tryb bez SQL

Jeśli wcześniej wyeksportujesz dane do CSV z kolumnami zgodnymi z SQL, możesz ominąć `pyodbc`:

```bash
python backtest_models.py ^
  --prognoza-csv D:\tmp\prognoza.csv ^
  --wykonanie-csv D:\tmp\wykonanie.csv ^
  --epochs 50
```

## Wyniki

Wyniki zapisują się w `backtest_artifacts/`:

- `summary.csv` - średnie wyniki dla każdego targetu,
- `{target}/metrics_by_fold.csv` - metryki per fold,
- `{target}/predictions.csv` - predykcje i wartości rzeczywiste,
- `{target}/fold_XXX/` - artefakty modelu, scaler i lista cech dla konkretnego folda.

Najważniejsze kolumny:

- `baseline_mae`, `baseline_rmse` - błąd surowej prognozy,
- `corrected_mae`, `corrected_rmse` - błąd po korekcie modelu,
- `improvement_mae_pct`, `improvement_rmse_pct` - poprawa procentowa; wartości dodatnie oznaczają, że korekta pomaga.

## Uwaga metodologiczna

Obecne pliki treningowe w repo używają losowego `train_test_split`, co miesza okresy historyczne i przyszłe. Do oceny produkcyjnej lepszy jest backtest chronologiczny, dlatego ten skrypt uczy każdy fold na wcześniejszym oknie i testuje dopiero na późniejszym oknie.
