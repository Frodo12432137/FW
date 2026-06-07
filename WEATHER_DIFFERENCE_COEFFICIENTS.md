# Wspolczynniki roznic prognoza vs wykonanie

Skrypt `weather_difference_coefficients.py` liczy parametry roznicy pomiedzy prognoza pogody a wykonaniem.

## Uruchomienie strzalka

Jesli uruchomisz plik strzalka w IDE, skrypt uzyje dat wpisanych w SQL-ach:

- prognoza: `2024-01-01 00:00:00` do `2026-10-30 23:45:00`,
- wykonanie: `2024-11-01` do `2026-10-30`.

Po polaczeniu realny wspolny zakres zacznie sie od okolic `2024-11-01`.

Domyslnie grupowanie jest po:

```text
miesiac, lokalizacja, godzina
```

Czyli np. wszystkie stycznie z dostepnych lat sa liczone razem jako `miesiac = 1`.

## Uruchomienie z parametrami

```bash
python weather_difference_coefficients.py ^
  --start-date "2024-11-01 00:00:00" ^
  --end-date "2026-10-30 23:45:00"
```

Jesli chcesz osobno kazdy miesiac kalendarzowy, np. `2025-01`, `2025-02`, uzyj:

```bash
python weather_difference_coefficients.py --group-mode year_month
```

## Wyniki

Pliki zapisza sie do folderu `weather_difference_coefficients/`:

- `wspolczynniki_roznic_long_*.csv` - format dlugi: jeden wiersz dla `miesiac/lokalizacja/godzina/target`,
- `wspolczynniki_roznic_wide_*.csv` - format szeroki: jeden wiersz dla `miesiac/lokalizacja/godzina`, kolumny dla predkosci, temperatury i kierunku,
- `polaczone_dane_sample_*.csv` - pierwsze 1000 polaczonych wierszy do kontroli,
- `metadata_*.json` - parametry uruchomienia.

## Co oznaczaja kolumny

- `liczba_obserwacji` - ile punktow weszlo do grupy,
- `prognoza_srednia` - srednia prognoza w grupie,
- `wykonanie_srednia` - srednie wykonanie w grupie,
- `roznica_srednia` - sredni bias, czyli `wykonanie - prognoza`,
- `mae` - sredni blad bezwzgledny,
- `rmse` - pierwiastek ze sredniego bledu kwadratowego,
- `wspolczynnik_mnoznik_srednich` - `wykonanie_srednia / prognoza_srednia`,
- `regresja_slope` i `regresja_intercept` - parametry modelu `wykonanie = intercept + slope * prognoza`,
- `regresja_r2` - dopasowanie regresji.

Dla kierunku wiatru roznica liczona jest katowo, czyli w zakresie od `-180` do `180` stopni.
