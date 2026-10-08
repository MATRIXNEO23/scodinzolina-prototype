"""Migrate the complete GPTina continuity into native gpt_agent memory.

The canonical repository is read-only input. Durable memories, decisions/checkpoints,
reflections, identity material, shared language, transcripts/raw sessions and protected
historical sources are preserved with provenance. Legacy recovery protocols/policies are
preserved as historical NON-OPERATIVE knowledge so they cannot govern the new runtime.
Derived indexes/projections and other owners are intentionally excluded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
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

CANONICAL = "MATRIXNEO23/scodinzolina-conntinuity"
THOUGHT_KINDS = {
    "reflection", "self_portrait", "shared_language", "stable_principles",
    "legacy_message", "protected_historical_gptina", "historical_voice_capsule",
    "historical_continuity_hypothesis",
}
NON_OPERATIVE_KINDS = {
    "restore_protocol", "recovery_protocol", "save_recovery_runbook",
    "memory_protocol", "ownership_policy", "access_policy", "scale_strategy",
    "visual_schema", "live_protocol", "live_documentation", "regression_test",
}
DERIVED_KINDS = {"current_router", "fast_router", "chronology_router", "visual_router"}
EXTRA_PATTERNS = [
    ("agent-exchanges/agents/gptina.md", "historical_agent_profile"),
    ("agent-exchanges/correspondence/**/*.md", "historical_agent_correspondence"),
    ("RISPOSTA_GPTINA_POSTICINO_SEGRETO*.md", "protected_historical_gptina"),
]


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
    lines, out, active, indent = front.splitlines(), [], False, None
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
                    out.append(m.group(2).strip().strip('"'))
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
    # Only accept calendar-looking separators; never infer dates from temp-dir digits.
    for m in re.finditer(r"(20\d{2})[-/](\d{2})[-/](\d{2})", path.as_posix()):
        try:
            return dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
    return dt.datetime(1970, 1, 1)


def importance_to_poignancy(value, kind: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 4 if kind in THOUGHT_KINDS else 3
    return max(1, min(10, value * 2))


def title_from_body(body: str, path: Path) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def manifest(source_root: Path):
    path = source_root / "rag" / "memory_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def excluded(rel: str, data: dict) -> bool:
    patterns = list(data.get("exclude", [])) + list(data.get("rag_exclude", []))
    patterns += ["rag/index/**", ".git/**", "rag/memories/tessa/**", "romanzo/**"]
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def inventory(source_root: Path):
    data = manifest(source_root)
    found = {}
    declared = list(data.get("sources", [])) + list(data.get("rag_sources", []))
    for spec in declared:
        pattern, kind = spec["pattern"], spec.get("kind", "legacy_source")
        if kind in DERIVED_KINDS:
            continue
        for path in source_root.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(source_root).as_posix()
            if excluded(rel, data) or rel.startswith("rag/memories/tessa/"):
                continue
            found.setdefault(rel, {"path": path, "source_role": kind})
    for pattern, kind in EXTRA_PATTERNS:
        for path in source_root.glob(pattern):
            if path.is_file():
                rel = path.relative_to(source_root).as_posix()
                if not excluded(rel, data):
                    found.setdefault(rel, {"path": path, "source_role": kind})
    return [found[k] for k in sorted(found)]


def read_record(item: dict, source_root: Path, overrides: dict):
    path, source_role = item["path"], item["source_role"]
    rel = path.relative_to(source_root).as_posix()
    raw = path.read_text(encoding="utf-8", errors="replace")
    front, body, meta = "", raw, {}
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        meta = payload if isinstance(payload, dict) else {}
    else:
        front, body = split_frontmatter(raw)
        meta = {k: scalar(front, k) for k in (
            "schema_version", "memory_id", "owner", "kind", "event_at",
            "recorded_at", "status", "event_id", "importance", "confidence",
            "append_only")}
        for k in ("supersedes", "thread_ids", "entity_refs", "source_refs", "media_refs", "tags"):
            meta[k] = list_field(front, k)

    override = overrides.get(rel, {})
    status = override.get("status") or meta.get("status") or "historical"
    title = title_from_body(body, path)
    memory_id = meta.get("memory_id") or rel
    non_operational = source_role in NON_OPERATIVE_KINDS
    marker = "NON-OPERATIVE LEGACY SOURCE" if non_operational else "IMPORTED HISTORY"
    prefix = f"[{marker} | role={source_role} | status={status} | source={rel}]\n"
    text = prefix + body.strip()
    tags = list(meta.get("tags") or [])
    keywords = {"gptina", "imported-history", source_role.lower(), str(status).lower()}
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
        "poignancy": importance_to_poignancy(meta.get("importance"), source_role),
        "keywords": sorted(keywords),
        "metadata": meta,
        "frontmatter": front,
        "source_role": source_role,
        "status": status,
        "status_override": override or None,
        "non_operational": non_operational,
        "node_type": "thought" if source_role in THOUGHT_KINDS else "event",
    }


def empty_memory():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "nodes.json").write_text("{}", encoding="utf-8")
        (directory / "embeddings.json").write_text("{}", encoding="utf-8")
        (directory / "kw_strength.json").write_text(json.dumps({
            "kw_strength_event": {}, "kw_strength_thought": {}}), encoding="utf-8")
        return core.AssociativeMemory(str(directory))


def build(source_root: Path, agent_name: str, generated_at: str):
    mfest = manifest(source_root)
    records = [read_record(item, source_root, mfest.get("status_overrides", {}))
               for item in inventory(source_root)]
    records.sort(key=lambda r: (r["created"], r["path"]))

    scratch = core.Scratch(str(source_root / "__absent_bootstrap__.json"))
    scratch.name = agent_name
    generated = dt.datetime.fromisoformat(generated_at)
    if generated.tzinfo is not None or generated.microsecond:
        raise ValueError("generated_at must be naive UTC seconds precision")
    scratch.curr_time = generated

    memory, sources = empty_memory(), {}
    counts = {"event": 0, "thought": 0, "non_operational": 0}
    for record in records:
        add = memory.add_thought if record["node_type"] == "thought" else memory.add_event
        node = add(record["created"], None, agent_name,
                   "historically records" if record["node_type"] == "event" else "historically reflects on",
                   record["memory_id"], record["text"], set(record["keywords"]),
                   record["poignancy"], (record["text"], None), None)
        counts[record["node_type"]] += 1
        counts["non_operational"] += int(record["non_operational"])
        sources[node.node_id] = {
            "source": f"canonical://{CANONICAL}/{record['path']}",
            "migration": "continuity-to-gpt-agent-native-v2",
            "original_path": record["path"],
            "original_sha256": record["sha256"],
            "original_title": record["title"],
            "source_role": record["source_role"],
            "status": record["status"],
            "status_override": record["status_override"],
            "original_metadata": record["metadata"],
            "original_frontmatter": record["frontmatter"],
            "imported_as": record["node_type"],
            "non_operational_legacy_instruction": record["non_operational"],
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
            "migration": "continuity-to-gpt-agent-native-v2",
            "counts": counts,
        }},
    }
    agent.validate_memory(agent.persona_load(state))
    return {
        "schema": "gpt-agent-native-bootstrap-v2",
        "source_repository": CANONICAL,
        "source_commit": None,
        "source_scope": "memory_manifest declared sources + GPTina historical extras; derived projections excluded",
        "record_count": len(records),
        "counts": counts,
        "generated_at": generated_at,
        "state": state,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="GPTina")
    p.add_argument("--generated-at", required=True)
    p.add_argument("--source-commit")
    args = p.parse_args()
    result = build(args.source_root.resolve(), args.name, args.generated_at)
    if args.source_commit:
        result["source_commit"] = args.source_commit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({"status": "migrated", "records": result["record_count"],
                      "counts": result["counts"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
