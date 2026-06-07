 -- zdefiniuj parametry datowe
DECLARE @start_date DATETIME = '2024-01-01 00:00:00';
DECLARE @end_date   DATETIME = '2026-10-30 23:45:00';

WITH NajblizszaPrognoza AS (
    SELECT
        punkt,
        temperatura,
        predkoscWiatru,
        kierunekWiatru,
        dataGodzinaCET,
        dataGodzinaUTC,
        ROW_NUMBER() OVER (
            PARTITION BY punkt, dataGodzinaCET
            ORDER BY ABS(DATEDIFF(
                MINUTE,
                czasDanychZrodlaCET,
                dataGodzinaCET
            ))
        ) AS rn
    FROM [PGESA_MarketAnalytics].[wa].[vPogodaPrognoza]
    WHERE punkt IN (
        'Galicja (Hnatkowice – Orzechowce)',
        'Kamieńsk',
        'Karnice',
        'Karnice II',
        'Kisielice',
        'Kisielice II',
        'Karwice',
        'Jagniątkowo (Lake Ostrowo)',
        'Lotnisko',
        'Malbork (Koniecwałd)',
        'Pelplin',
        'Resko I',
        'Resko II',
        'Rybice',
        'Skoczykłody',
        'Starza',
        'Wojciechowo',
        'Zalesie',
        'Żuromin'
    )
      AND czasDanychZrodlaCET <= dataGodzinaCET
      AND dataGodzinaCET BETWEEN @start_date AND @end_date
)

SELECT
    punkt,
    temperatura,
    predkoscWiatru,
    kierunekWiatru,
    dataGodzinaCET,
    dataGodzinaUTC
FROM NajblizszaPrognoza
WHERE rn = 1
ORDER BY punkt, dataGodzinaCET;
