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

from policy_engine.service import answer_policy_question

st.set_page_config(page_title="Policy engine", layout="centered")
st.title("Policy engine")

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
        with st.spinner("Running planner, database, and formatter…"):
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
