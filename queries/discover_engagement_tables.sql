SELECT TOP 100
  s.name AS schema_name,
  t.name AS table_name
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
WHERE s.name = 'education'
  AND (
    t.name LIKE '%engage%' OR
    t.name LIKE '%comm%' OR
    t.name LIKE '%message%' OR
    t.name LIKE '%sms%' OR
    t.name LIKE '%email%' OR
    t.name LIKE '%call%' OR
    t.name LIKE '%interaction%' OR
    t.name LIKE '%log%'
  )
ORDER BY t.name;