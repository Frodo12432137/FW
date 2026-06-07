WITH Windowed AS (
  SELECT *
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
    AND dataGodzinaCET BETWEEN DATEADD(hour, -24, GETDATE()) AND DATEADD(hour, 48, GETDATE())
),
Ranked AS (
  SELECT
    w.*,
    ROW_NUMBER() OVER (
      PARTITION BY w.punkt, w.dataGodzinaCET
      ORDER BY ABS(DATEDIFF(second, w.dataGodzinaCET, w.czasDanychZrodlaCET)) ASC,
               w.czasDanychZrodlaCET DESC
    ) AS rn
  FROM Windowed w
)
SELECT *
FROM Ranked
WHERE rn = 1
ORDER BY punkt, dataGodzinaCET, czasDanychZrodlaCET;
