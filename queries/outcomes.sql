-- Outcomes extraction: booking appointments within a time window.
-- NOTE: for now, replace the placeholders manually before each run.
-- Next step we will parameterize.

SELECT
  appointment_id,
  org_id,
  contact_id,
  ghl_contact_id,
  status,
  CONVERT(varchar(33), requested_at, 127)  AS requested_at,
  CONVERT(varchar(33), scheduled_at, 127)  AS scheduled_at,
  CONVERT(varchar(33), created_at, 127)    AS created_at,
  CONVERT(varchar(33), updated_at, 127)    AS updated_at
FROM education.booking_appointments
WHERE created_at >= ?
  AND created_at < ?
ORDER BY created_at ASC;