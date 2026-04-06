"""
policy_engine — isolated pipeline:
  question → LLM search planner (JSON) → validated SQL → Postgres → LLM answer formatter.

Main entrypoints: policy_engine.service.answer_policy_question, policy_engine.runner, policy_engine.ui_app.
"""
