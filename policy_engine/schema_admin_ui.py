"""Schema evolution approval UI for adaptive ingestion."""

from __future__ import annotations

import json

import streamlit as st

from policy_engine.db import execute, run_query

st.set_page_config(page_title="Schema Admin", layout="wide")
st.title("Adaptive Schema Admin")

rows = run_query(
    """
    SELECT request_id, batch_id, migration_class, decision_status, proposal_json, created_at
    FROM schema_change_requests
    ORDER BY created_at DESC
    LIMIT 200
    """
)

if not rows:
    st.info("No schema change requests yet.")
    st.stop()

selected = st.selectbox(
    "Select request",
    options=[r["request_id"] for r in rows],
)
row = next(r for r in rows if r["request_id"] == selected)

st.write(f"Batch: `{row['batch_id']}`")
st.write(f"Class: `{row['migration_class']}`")
st.write(f"Status: `{row['decision_status']}`")
st.json(row["proposal_json"])

decision_note = st.text_input("Decision note")
col1, col2 = st.columns(2)
with col1:
    if st.button("Approve"):
        execute(
            """
            UPDATE schema_change_requests
            SET decision_status = 'approved', reviewer = %s, decision_note = %s, decided_at = NOW(), updated_at = NOW()
            WHERE request_id = %s
            """,
            ("streamlit_admin", decision_note, selected),
        )
        st.success("Approved.")
with col2:
    if st.button("Reject"):
        execute(
            """
            UPDATE schema_change_requests
            SET decision_status = 'rejected', reviewer = %s, decision_note = %s, decided_at = NOW(), updated_at = NOW()
            WHERE request_id = %s
            """,
            ("streamlit_admin", decision_note, selected),
        )
        st.warning("Rejected.")

st.subheader("SQL Preview")
items = (row.get("proposal_json") or {}).get("items", [])
if isinstance(items, str):
    try:
        items = json.loads(items).get("items", [])
    except Exception:
        items = []
for item in items:
    concept = item.get("concept")
    gap_type = item.get("gap_type")
    if gap_type == "new_scalar":
        st.code(f'ALTER TABLE policies_v2 ADD COLUMN IF NOT EXISTS "{concept}" TEXT;', language="sql")
    elif gap_type == "new_relationship":
        st.code(
            f"CREATE TABLE IF NOT EXISTS policy_{concept} (id TEXT PRIMARY KEY, policy_id TEXT NOT NULL, value_text TEXT NOT NULL);",
            language="sql",
        )

