SELECT
  ghl_contact_id,
  channel,
  direction,
  sender,
  recipient,
  subject,
  thread_id,
  message_id,
  type_id,
  CONVERT(varchar(33), [timestamp], 127) AS event_ts,
  CONVERT(varchar(33), created_at, 127)  AS created_at,
  conversation_tags
FROM dbo.Education_Cory_Communication_Log
WHERE created_at >= ?
  AND created_at < ?
  AND ghl_contact_id IS NOT NULL;
