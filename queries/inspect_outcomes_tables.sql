-- Column inventory for candidate outcomes tables
SELECT 
  TABLE_SCHEMA,
  TABLE_NAME,
  COLUMN_NAME,
  DATA_TYPE,
  IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE (TABLE_SCHEMA = 'education' AND TABLE_NAME IN ('booking_action_log','booking_appointments'))
ORDER BY TABLE_NAME, ORDINAL_POSITION;