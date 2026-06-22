"""
policy_engine — isolated pipeline:
  question → LLM search planner (JSON) → validated SQL → Postgres → LLM answer formatter.

Main entrypoint: policy_engine.service.answer_policy_question.
"""
