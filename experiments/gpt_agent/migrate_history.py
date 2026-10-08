"""Convert GPTina legacy memory records into native gpt_agent AssociativeMemory state.

No model call is used during migration. The source repository is treated as read-only.
The result is a tracked bootstrap file that can initialize the experimental agent
after the legacy memory architecture is removed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import tempfile

try:
    from . import agent
    from . import upstream as core
except ImportError:
    import agent
    import upstream as core


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_frontmatter(text: str):
    if not text.startswith("---\n"):
        return "", text
    end = text.find("\n---\n", 4)
    if end < 0:
        return "", text
    return text[4:end], text[end + 5:]


def scalar(front: str, key: str):
    m = re.search(rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$", front)
    if not m:
        return None
    value = m.group(1).strip()
    if value in ("[]", "null", "None", ""):
        return [] if value == "[]" else None
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if value.isdigit():
        return int(value)
    return value


def list_field(front: str, key: str):
    lines = front.splitlines()
    out = []
    active = False
    indent = None
    for line in lines:
        if re.match(rf"^{re.escape(key)}:\s*\[\]\s*$", line):
            return []
        if re.match(rf"^{re.escape(key)}:\s*$", line):
            active = True
            continue
        if active:
            m = re.match(r"^(\s+)-\s+(.*)$", line)
            if m:
                if indent is None:
                    indent = len(m.group(1))
                if len(m.group(1)) == indent:
                    value = m.group(2).strip().strip('"')
                    out.append(value)
                    continue
            if line and not line.startswith(" "):
                break
    return out


def parse_time(meta: dict, path: Path) -> dt.datetime:
    raw = meta.get("recorded_at") or meta.get("event_at")
    if raw:
        try:
            parsed = dt.datetime.fromisoformat(str(raw))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
            return parsed.replace(microsecond=0)
        except ValueError:
            pass
    m = re.search(r"(20\d{2})[-/]?(\d{2})[-/]?(\d{2})", path.as_posix())
    if m:
        return dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return dt.datetime(1970, 1, 1)


def importance_to_poignancy(value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 3
    return max(1, min(10, value * 2))


def title_from_body(body: str, path: Path) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def read_record(path: Path, source_root: Path):
    rel = path.relative_to(source_root).as_posix()
    if path.suffix.lower() == ".json":
        raw = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        meta = payload if isinstance(payload, dict) else {}
        front = ""
    else:
        raw = path.read_text(encoding="utf-8")
        front, body = split_frontmatter(raw)
        meta = {k: scalar(front, k) for k in (
            "schema_version", "memory_id", "owner", "kind", "event_at",
            "recorded_at", "status", "event_id", "importance", "confidence",
            "append_only")}
        for k in ("supersedes", "thread_ids", "entity_refs", "source_refs", "media_refs", "tags"):
            meta[k] = list_field(front, k)
    title = title_from_body(body, path)
    status = meta.get("status") or "legacy"
    memory_id = meta.get("memory_id") or rel
    prefix = f"[IMPORTED HISTORY | status={status} | source={rel}]\n"
    text = prefix + body.strip()
    tags = list(meta.get("tags") or [])
    keywords = {"gptina", "imported-history", status.lower()}
    keywords.update(str(v).lower() for v in tags if v)
    for v in meta.get("entity_refs") or []:
        if isinstance(v, str) and len(v) < 80:
            keywords.add(v.lower())
    return {
        "path": rel,
        "sha256": sha256(path),
        "title": title,
        "text": text,
        "memory_id": str(memory_id),
        "created": parse_time(meta, path),
        "poignancy": importance_to_poignancy(meta.get("importance")),
        "keywords": sorted(keywords),
        "metadata": meta,
        "frontmatter": front,
    }


def inventory(source_root: Path):
    candidates = []
    root = source_root / "rag" / "memories"
    if not root.exists():
        raise ValueError(f"Missing legacy memory root: {root}")
    for path in root.glob("*.md"):
        if path.name != "README.md":
            candidates.append(path)
    candidates.extend(root.glob("*.json"))
    gptina = root / "gptina"
    if gptina.exists():
        candidates.extend(gptina.rglob("*.md"))
        candidates.extend(gptina.rglob("*.json"))
    # Never import other owners such as rag/memories/tessa/**.
    return sorted(set(candidates), key=lambda p: p.as_posix())


def empty_memory():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "nodes.json").write_text("{}", encoding="utf-8")
        (directory / "embeddings.json").write_text("{}", encoding="utf-8")
        (directory / "kw_strength.json").write_text(
            json.dumps({"kw_strength_event": {}, "kw_strength_thought": {}}), encoding="utf-8")
        return core.AssociativeMemory(str(directory))


def build(source_root: Path, agent_name: str, generated_at: str):
    records = [read_record(path, source_root) for path in inventory(source_root)]
    records.sort(key=lambda r: (r["created"], r["path"]))

    scratch = core.Scratch(str(source_root / "__absent_bootstrap__.json"))
    scratch.name = agent_name
    generated = dt.datetime.fromisoformat(generated_at)
    if generated.tzinfo is not None or generated.microsecond:
        raise ValueError("generated_at must be naive UTC seconds precision")
    scratch.curr_time = generated

    memory = empty_memory()
    sources = {}
    for record in records:
        node = memory.add_event(
            record["created"], None,
            agent_name, "remembers", record["memory_id"],
            record["text"], set(record["keywords"]), record["poignancy"],
            (record["text"], None), None)
        sources[node.node_id] = {
            "source": f"canonical://MATRIXNEO23/scodinzolina-conntinuity/{record['path']}",
            "migration": "legacy-to-gpt-agent-native-v1",
            "original_path": record["path"],
            "original_sha256": record["sha256"],
            "original_title": record["title"],
            "original_metadata": record["metadata"],
            "original_frontmatter": record["frontmatter"],
            "imported_as": "event",
            "historical_not_lived_by_new_runtime": True,
        }

    state = {
        "scratch": agent.scratch_dump(scratch),
        "memory": agent.memory_dump(memory),
        "sources": sources,
        "pending": None,
        "last": {"answers": [], "result": {
            "initialized": agent_name,
            "history_migrated": len(records),
            "migration": "legacy-to-gpt-agent-native-v1"
        }},
    }
    agent.validate_memory(agent.persona_load(state))
    return {
        "schema": "gpt-agent-native-bootstrap-v1",
        "source_repository": "MATRIXNEO23/scodinzolina-conntinuity",
        "source_scope": ["rag/memories/*.md|json", "rag/memories/gptina/**/*.md|json"],
        "record_count": len(records),
        "generated_at": generated_at,
        "state": state,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="GPTina")
    p.add_argument("--generated-at", required=True)
    args = p.parse_args()
    result = build(args.source_root.resolve(), args.name, args.generated_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({"status": "migrated", "records": result["record_count"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
