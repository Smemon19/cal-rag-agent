#!/usr/bin/env python3
"""Manage Streamlit login users in Google Cloud Secret Manager."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import bcrypt
from google.cloud import secretmanager


def _project_id(arg_value: str | None) -> str:
    return arg_value or os.getenv("AUTH_PROJECT_ID") or os.getenv("BQ_PROJECT") or "badgers-487618"


def _secret_name(arg_value: str | None) -> str:
    return arg_value or os.getenv("AUTH_USERS_SECRET") or "cal-rag-users"


def _secret_path(project_id: str, secret_name: str) -> str:
    return f"projects/{project_id}/secrets/{secret_name}"


def _load_users(client: secretmanager.SecretManagerServiceClient, project_id: str, secret_name: str) -> Dict[str, str]:
    version_path = f"{_secret_path(project_id, secret_name)}/versions/latest"
    response = client.access_secret_version(request={"name": version_path})
    payload = response.payload.data.decode("utf-8")
    data = json.loads(payload) if payload.strip() else {}
    if not isinstance(data, dict):
        raise ValueError("Secret payload must be a JSON object mapping username to bcrypt hash.")
    return {str(k): str(v) for k, v in data.items()}


def _save_users(client: secretmanager.SecretManagerServiceClient, project_id: str, secret_name: str, users: Dict[str, str]) -> str:
    parent = _secret_path(project_id, secret_name)
    payload = json.dumps(users, sort_keys=True, ensure_ascii=False).encode("utf-8")
    version = client.add_secret_version(request={"parent": parent, "payload": {"data": payload}})
    return version.name


def _cmd_add(client: secretmanager.SecretManagerServiceClient, project_id: str, secret_name: str, username: str, password: str) -> None:
    users = _load_users(client, project_id, secret_name)
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    users[username] = pw_hash
    version_name = _save_users(client, project_id, secret_name, users)
    print(f"Added/updated user: {username}")
    print(f"Secret version created: {version_name}")


def _cmd_remove(client: secretmanager.SecretManagerServiceClient, project_id: str, secret_name: str, username: str) -> None:
    users = _load_users(client, project_id, secret_name)
    if username not in users:
        print(f"User not found: {username}")
        return
    del users[username]
    version_name = _save_users(client, project_id, secret_name, users)
    print(f"Removed user: {username}")
    print(f"Secret version created: {version_name}")


def _cmd_list(client: secretmanager.SecretManagerServiceClient, project_id: str, secret_name: str) -> None:
    users = _load_users(client, project_id, secret_name)
    if not users:
        print("No users configured.")
        return
    for username in sorted(users.keys()):
        print(username)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage users in Secret Manager for Streamlit login.")
    parser.add_argument("--project", default=None, help="GCP project ID (default: badgers-487618 or env).")
    parser.add_argument("--secret", default=None, help="Secret name (default: cal-rag-users or env).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add or update a user.")
    add_parser.add_argument("username")
    add_parser.add_argument("password")

    remove_parser = subparsers.add_parser("remove", help="Remove a user.")
    remove_parser.add_argument("username")

    subparsers.add_parser("list", help="List usernames.")

    args = parser.parse_args()
    project_id = _project_id(args.project)
    secret_name = _secret_name(args.secret)

    client = secretmanager.SecretManagerServiceClient()
    if args.command == "add":
        _cmd_add(client, project_id, secret_name, args.username, args.password)
    elif args.command == "remove":
        _cmd_remove(client, project_id, secret_name, args.username)
    elif args.command == "list":
        _cmd_list(client, project_id, secret_name)


if __name__ == "__main__":
    main()
