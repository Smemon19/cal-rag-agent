"""
Durable chat history.

The sidebar rail rendered threads from an in-memory array, so every reload
discarded the user's conversations. These helpers back that rail with the
conversations / conversation_messages tables.

Every function takes the acting user id and scopes its SQL to that user, so a
conversation belonging to somebody else is indistinguishable from one that does
not exist.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException

from policy_badger.engine.db import execute, run_query, transaction

MAX_TITLE_LEN = 80
MAX_CONVERSATIONS = 200


def _derive_title(question: str) -> str:
    text = " ".join((question or "").split())
    if not text:
        return "New conversation"
    if len(text) <= MAX_TITLE_LEN:
        return text
    return text[: MAX_TITLE_LEN - 1].rstrip() + "…"


def list_conversations(user_id: str, limit: int = 50) -> list[dict]:
    """Most recently active conversations first, for the rail."""
    return run_query(
        """
        SELECT c.id,
               c.title,
               c.updated_at,
               COUNT(m.id) AS message_count
        FROM conversations c
        LEFT JOIN conversation_messages m ON m.conversation_id = c.id
        WHERE c.user_id = %s AND c.is_archived = false
        GROUP BY c.id, c.title, c.updated_at
        ORDER BY c.updated_at DESC
        LIMIT %s
        """,
        (user_id, min(limit, MAX_CONVERSATIONS)),
    )


def get_conversation(user_id: str, conversation_id: str) -> dict[str, Any]:
    """Full transcript for one conversation the user owns."""
    rows = run_query(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = %s AND user_id = %s",
        (conversation_id, user_id),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = run_query(
        """
        SELECT id, role, content, rows_found, created_at
        FROM conversation_messages
        WHERE conversation_id = %s
        ORDER BY created_at, id
        """,
        (conversation_id,),
    )
    conversation = dict(rows[0])
    conversation["messages"] = messages
    return conversation


def create_conversation(user_id: str, title: Optional[str] = None) -> dict[str, Any]:
    with transaction() as cur:
        cur.execute(
            "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id, title, created_at, updated_at",
            (user_id, (title or "New conversation")[:MAX_TITLE_LEN]),
        )
        row = cur.fetchone()
    return dict(row)


def _assert_owned(user_id: str, conversation_id: str) -> None:
    rows = run_query(
        "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
        (conversation_id, user_id),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Conversation not found.")


def rename_conversation(user_id: str, conversation_id: str, title: str) -> None:
    clean = " ".join((title or "").split())[:MAX_TITLE_LEN]
    if not clean:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    _assert_owned(user_id, conversation_id)
    execute("UPDATE conversations SET title = %s WHERE id = %s", (clean, conversation_id))


def delete_conversation(user_id: str, conversation_id: str) -> None:
    _assert_owned(user_id, conversation_id)
    # conversation_messages cascades on delete.
    execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))


def append_exchange(
    user_id: str,
    conversation_id: Optional[str],
    question: str,
    answer: str,
    rows_found: int = 0,
) -> str:
    """
    Record one question/answer pair, creating the conversation when needed.

    Returns the conversation id so the caller can hand it back to the client.
    """
    if conversation_id:
        _assert_owned(user_id, conversation_id)
    else:
        conversation_id = str(create_conversation(user_id, _derive_title(question))["id"])

    with transaction() as cur:
        cur.execute(
            """
            INSERT INTO conversation_messages (conversation_id, role, content)
            VALUES (%s, 'user', %s)
            """,
            (conversation_id, question),
        )
        cur.execute(
            """
            INSERT INTO conversation_messages (conversation_id, role, content, rows_found)
            VALUES (%s, 'assistant', %s, %s)
            """,
            (conversation_id, answer, rows_found),
        )
        # Touch the parent so the rail orders by real activity, not creation.
        cur.execute("UPDATE conversations SET updated_at = now() WHERE id = %s", (conversation_id,))

    return conversation_id
