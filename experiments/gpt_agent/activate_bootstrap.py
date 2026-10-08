"""Activate a tracked native gpt_agent bootstrap into .state/state.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import agent
except ImportError:
    import agent

SUPPORTED = {"gpt-agent-native-bootstrap-v1", "gpt-agent-native-bootstrap-v2"}


def activate(path: Path, force: bool = False):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") not in SUPPORTED:
        raise ValueError("Unsupported bootstrap schema")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("Bootstrap has no state")
    persona = agent.persona_load(state)
    agent.validate_memory(persona)
    if agent.STATE.exists() and not force:
        raise ValueError("State already exists; pass --force only for an explicit test reset")
    with agent.locked():
        if agent.STATE.exists() and not force:
            raise ValueError("State already exists")
        agent.atomic_save(state)
    return {
        "status": "activated",
        "schema": payload["schema"],
        "memory_count": len(state["memory"]["nodes"]),
        "bootstrap": str(path),
        "state_sha256": agent.receipt(state)["state_sha256"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bootstrap", type=Path)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    print(json.dumps(activate(args.bootstrap, args.force), ensure_ascii=False))


if __name__ == "__main__":
    main()
