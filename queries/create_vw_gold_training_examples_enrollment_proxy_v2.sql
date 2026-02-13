CREATE OR ALTER VIEW dbo.vw_gold_training_examples_enrollment_proxy_v2 AS
SELECT
    v1.*,
    ISNULL(vm.voicemail_drops_24h, 0) AS voicemail_drops_24h,
    CASE
        WHEN v1.attempts_voice_24h - ISNULL(vm.voicemail_drops_24h, 0) < 0 THEN 0
        ELSE v1.attempts_voice_24h - ISNULL(vm.voicemail_drops_24h, 0)
    END AS attempts_voice_no_voicemail_24h
FROM dbo.vw_gold_training_examples_enrollment_proxy_v1 v1
OUTER APPLY (
    SELECT
        COUNT(*) AS voicemail_drops_24h
    FROM education.campaign_activities ca
    WHERE ca.org_id = v1.org_id
      AND ca.enrollment_id = v1.enrollment_id
      AND ca.channel = 'voice'
      AND ca.outcome = 'voicemail'
      AND COALESCE(ca.completed_at, ca.call_started_at, ca.created_at) >= v1.decision_ts_utc
      AND COALESCE(ca.completed_at, ca.call_started_at, ca.created_at) <  v1.feature_cutoff_ts_utc
) vm;