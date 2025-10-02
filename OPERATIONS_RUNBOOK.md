Operations Runbook (Firebase App Hosting)

Monitoring (lightweight)

- Enable request/error logs in Firebase Console → App Hosting → Backends → default.
- Add an external uptime check (UptimeRobot/Pingdom) for the hosted URL root. Target: HTTP 200 in <2s median, sample every 5 minutes.
- Configure alerts to team email/SMS in your monitor provider.

Smoke Test

- Open hosted URL, expand Diagnostics sidebar:
  - Confirm masked OPENAI prefix, model, backend, collection, fingerprint, model home `/app/models`.
- Run the slate-roof underlayment query twice on two devices; expect IBC citations and collection `docs_ibc_v2`.

Triage (on issues)

- Check last rollout build logs for baked-model and health pass.
- Verify Diagnostics reflect App Hosting env; ensure OPENAI secret latest version is Enabled.
- Ensure no plain `OPENAI_API_KEY` env var exists.
- If retrieval empty/off-topic: confirm `RAG_COLLECTION_NAME` and check mixed-embedding warnings; switch collection or re-embed.

Rollback

- Firebase Console → App Hosting → Rollouts → select last known good rollout → Promote/Restore to live.
- After rollback, re-verify Diagnostics and the slate-roof query from two devices.
