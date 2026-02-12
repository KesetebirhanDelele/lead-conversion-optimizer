SELECT
appointment_id,
org_id,
action_type,
action_detail,
CONVERT(varchar(33), created_at, 127) AS created_at
FROM education.booking_action_log
WHERE created_at >= ?
AND created_at < ?
ORDER BY created_at ASC;