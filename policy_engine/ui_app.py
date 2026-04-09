"""
Minimal Streamlit tester for policy_engine.

Run from repository root:
    streamlit run policy_engine/ui_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=True)

import streamlit as st

from adaptive_ingestion.pipeline import AdaptiveIngestionPipeline, IngestionDocumentInput
from policy_engine.db import execute, run_query
from policy_engine.service import answer_policy_question

st.set_page_config(page_title="Policy engine", layout="centered")
st.title("Policy engine")

tab1, tab2, tab3 = st.tabs(["Q&A", "Ingestion", "Schema Requests"])

with tab1:
    question = st.text_area(
        "Your question",
        placeholder="e.g. What do I bill internal marketing work to?",
        height=120,
    )

    if st.button("Get answer"):
        q = (question or "").strip()
        if not q:
            st.warning("Enter a question.")
        else:
            with st.spinner("Running planner, database, and formatter..."):
                try:
                    out = answer_policy_question(q)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

            st.subheader("Answer")
            st.write(out["answer"])

            with st.expander("Details (search spec, SQL, rows)"):
                st.markdown("**Search spec**")
                st.json(out["search_spec"])
                st.markdown("**Primary SQL**")
                st.code(out["sql"], language="sql")
                st.write("Params:", out["params"])
                st.markdown("**Relaxed query used**")
                st.write(out["used_relaxed_query"])
                if out.get("relaxed_sql"):
                    st.code(out["relaxed_sql"], language="sql")
                    st.write("Relaxed params:", out.get("relaxed_params", []))
                st.markdown(f"**Row count:** {out['row_count']}")
                st.markdown("**Rows**")
                st.json(out.get("rows", []))

with tab2:
    st.subheader("Run Adaptive Ingestion Batch")
    source_uri = st.text_input("Source URI", value="file://example-policy.txt")
    content = st.text_area("Document Content", height=220)
    triggered_by = st.text_input("Triggered by", value="streamlit_user")
    if st.button("Run ingestion"):
        pipe = AdaptiveIngestionPipeline()
        batch_id = pipe.create_batch(triggered_by=triggered_by, trigger_source="streamlit")
        result = pipe.run_batch(
            batch_id=batch_id,
            documents=[IngestionDocumentInput(source_uri=source_uri, content=content, title="Uploaded Policy")],
        )
        proposals = pipe.plan_schema_changes(batch_id=batch_id)
        st.success(f"Batch complete: {batch_id}")
        st.json({"result": result, "proposals": proposals})

with tab3:
    st.subheader("Schema Change Requests")
    try:
        rows = run_query(
            """
            SELECT request_id, batch_id, migration_class, decision_status, proposal_json, created_at
            FROM schema_change_requests
            ORDER BY created_at DESC
            LIMIT 100
            """
        )
    except Exception as e:
        st.warning(f"Schema request table not available yet: {e}")
        rows = []
    if not rows:
        st.info("No requests yet.")
    else:
        choice = st.selectbox("Request ID", [r["request_id"] for r in rows])
        selected = next(r for r in rows if r["request_id"] == choice)
        st.json(selected)
        note = st.text_input("Reviewer note")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Approve request"):
                execute(
                    """
                    UPDATE schema_change_requests
                    SET decision_status='approved', reviewer='streamlit_user', decision_note=%s, decided_at=NOW(), updated_at=NOW()
                    WHERE request_id=%s
                    """,
                    (note, choice),
                )
                st.success("Approved.")
        with c2:
            if st.button("Reject request"):
                execute(
                    """
                    UPDATE schema_change_requests
                    SET decision_status='rejected', reviewer='streamlit_user', decision_note=%s, decided_at=NOW(), updated_at=NOW()
                    WHERE request_id=%s
                    """,
                    (note, choice),
                )
                st.warning("Rejected.")
