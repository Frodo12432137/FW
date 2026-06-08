# Wspolczynniki roznic prognoza vs wykonanie

Skrypt `weather_difference_coefficients.py` liczy wspolczynniki zgodnie z uwaga z maila:

> Dla wiatru korekta nie powinna zalezec od godziny doby tylko od poziomu wietrznosci.

Domyslnie skrypt liczy dane od:

```text
2025-08-01 00:00:00
```

czyli od sierpnia zeszlego roku.

## Uruchomienie strzalka

Jesli uruchomisz plik strzalka w IDE, skrypt:

- pobierze prognoze i wykonanie z SQL,
- polaczy dane po `dataGodzinaCET` i `lokalizacja`,
- policzy wspolczynniki po poziomie prognozowanej wartosci,
- zapisze raport bledow prognozy predkosci, kierunku i temperatury od `2025-08-01`.

## Glowne wyniki

Pliki zapisza sie do folderu `weather_difference_coefficients/`.

Dla predkosci wiatru:

```text
predkosc_wide_*.csv
predkosc_long_*.csv
```

Grupowanie:

```text
lokalizacja x poziom_wietrznosci
```

Poziomy wietrznosci sa co 2 m/s:

```text
0, 2, 4, 6, ..., 36
```

Glowny wspolczynnik:

```text
wsp_v = srednie_wykonanie / srednia_prognoza
```

Dla kierunku wiatru:

```text
kierunek_wide_*.csv
kierunek_long_*.csv
```

Grupowanie:

```text
lokalizacja x poziom_kierunku
```

Poziomy kierunku sa co 20 stopni:

```text
0, 20, 40, ..., 360
```

Glowny wspolczynnik:

```text
wsp_k = srednia roznica katowa wykonanie - prognoza
```

Dla temperatury:

```text
temperatura_wide_*.csv
temperatura_long_*.csv
```

Grupowanie:

```text
lokalizacja x poziom_temperatury
```

Poziomy temperatury sa co 4 stopnie:

```text
-32, -28, -24, ..., 36
```

Glowny wspolczynnik:

```text
wsp_t = srednia roznica wykonanie - prognoza
```

## Raport bledow od sierpnia 2025

Skrypt zapisuje tez:

```text
blad_prognozy_fw_od_2025_08_lokalizacje_*.csv
blad_prognozy_fw_od_2025_08_ogolem_*.csv
```

Te pliki odpowiadaja na pytanie z maila: jaki blad prognozy predkosci, kierunku i temperatury jest na FW od sierpnia zeszlego roku.

Kolumny:

- `bias` - sredni blad, czyli `wykonanie - prognoza`,
- `mae` - sredni blad bezwzgledny,
- `rmse` - pierwiastek ze sredniego bledu kwadratowego,
- `prognoza_srednia` - srednia prognoza,
- `wykonanie_srednia` - srednie wykonanie.

Dla kierunku wiatru roznica liczona jest katowo, czyli w zakresie od `-180` do `180` stopni.

## Stary uklad godzinowy

Stary uklad `miesiac x lokalizacja x godzina` jest nadal dostepny, ale tylko opcjonalnie:

```bash
python weather_difference_coefficients.py --legacy-time-groups
```

Wtedy dodatkowo zapisza sie pliki:

```text
legacy_wspolczynniki_roznic_long_*.csv
legacy_wspolczynniki_roznic_wide_*.csv
```
