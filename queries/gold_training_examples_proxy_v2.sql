SELECT
  org_id,
  enrollment_id,
  ghl_contact_id,
  CAST(decision_ts_utc AS datetime2(0)) AS decision_ts_utc,
  CAST(feature_cutoff_ts_utc AS datetime2(0)) AS feature_cutoff_ts_utc,
  contact_timezone,
  attempts_24h,
  attempts_sms_24h,
  attempts_email_24h,
  attempts_voice_no_voicemail_24h,
  label_responded_within_7d
FROM dbo.vw_gold_training_examples_enrollment_proxy_v2
WHERE decision_ts_utc >= ?
  AND decision_ts_utc < ?
ORDER BY org_id, enrollment_id;