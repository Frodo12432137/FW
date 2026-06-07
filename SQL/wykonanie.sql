SELECT
    [NazwaFarmy] AS lokalizacja,
    TRY_CONVERT(datetime2,
        CONCAT(CONVERT(varchar(10), [Data], 120), ' ', COALESCE(CONVERT(varchar(8), [Czas], 108), '00:00:00'))
    ) AS DataiCzasOdczytu,
    AVG([PredkoscWiatru]) AS predkoscWiatruLokalizacja_wykonanie,
    AVG([KierunekWiatru]) AS kierunekWiatruLokalizacja_wykonanie,
    AVG([Temperatura]) AS temperaturaLokalizacja_wykonanie
FROM
    [PGEEO_DDS].[spss_odczyt].[vFarmyWiatroweDaneMeteoEnergiaBrutto10min]
WHERE
    [NazwaFarmy] IN (
        'Galicja','Kamieńsk','Karnice','Karnice II','Kisielice','Kisielice II',
        'Karwice','Jagniątkowo','Lotnisko','Malbork','Pelplin','Resko I',
        'Resko II','Rybice','Skoczykłody','Starza','Wojciechowo','Zalesie','Żuromin'
    )
    AND TRY_CONVERT(date, [Data]) >= '2024-11-01'
    AND TRY_CONVERT(date, [Data]) <= '2026-10-30'
GROUP BY
    [NazwaFarmy],
    TRY_CONVERT(datetime2,
        CONCAT(CONVERT(varchar(10), [Data], 120), ' ', COALESCE(CONVERT(varchar(8), [Czas], 108), '00:00:00'))
    )
ORDER BY
    [NazwaFarmy],
    TRY_CONVERT(datetime2,
        CONCAT(CONVERT(varchar(10), [Data], 120), ' ', COALESCE(CONVERT(varchar(8), [Czas], 108), '00:00:00'))
    );
