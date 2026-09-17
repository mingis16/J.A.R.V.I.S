"""Single place to construct the Anthropic client.

Some orgs issue API keys that aren't scoped to a single workspace, in which
case every request needs an explicit `anthropic-workspace-id` header or the
API rejects it with a 400 (invalid_request_error: "not scoped to a
workspace"). Set ANTHROPIC_WORKSPACE_ID in .env to fix that without touching
any calling code.
"""
from __future__ import annotations

import os

import anthropic


def build_client() -> anthropic.Anthropic:
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace_id:
        return anthropic.Anthropic(default_headers={"anthropic-workspace-id": workspace_id})
    return anthropic.Anthropic()
