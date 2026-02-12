SELECT TOP 50
  s.name AS schema_name,
  t.name AS table_name
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
WHERE t.name LIKE '%outcome%' OR t.name LIKE '%conversion%' OR t.name LIKE '%booking%'
ORDER BY s.name, t.name;