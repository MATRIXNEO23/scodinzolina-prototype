from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import migrate_history


def write_manifest(root: Path):
    rag = root / "rag"
    rag.mkdir(parents=True, exist_ok=True)
    (rag / "memory_manifest.json").write_text(json.dumps({
        "sources": [
            {"pattern": "GPTINA_REFLECTIONS.md", "kind": "reflection"},
            {"pattern": "CONTINUITY.md", "kind": "stable_principles"},
            {"pattern": "checkpoints/**/*.md", "kind": "checkpoint"},
        ],
        "rag_sources": [
            {"pattern": "rag/memories/gptina/**/*.md", "kind": "gptina_memory"},
            {"pattern": "rag/GPTINA_AUTO_RECOVERY_PROMPT.md", "kind": "recovery_protocol"},
            {"pattern": "rag/index/CURRENT_CONTEXT.md", "kind": "current_router"},
        ],
        "exclude": ["rag/index/**"],
        "rag_exclude": ["rag/memories/tessa/**"],
        "status_overrides": {
            "rag/memories/gptina/old.md": {
                "status": "superseded", "replaced_by": "rag/memories/gptina/new.md"
            }
        }
    }), encoding="utf-8")


class MigrationTests(unittest.TestCase):
    def test_migrates_memories_reflections_decisions_and_protocols(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_manifest(root)
            mem = root / "rag" / "memories" / "gptina"
            mem.mkdir(parents=True)
            (mem / "old.md").write_text("---\nmemory_id: \"m1\"\nrecorded_at: \"2026-10-07T06:03:27+02:00\"\nimportance: 5\n---\n# Old\nPast.", encoding="utf-8")
            (mem / "new.md").write_text("---\nmemory_id: \"m2\"\nrecorded_at: \"2026-10-08T21:28:00+02:00\"\nimportance: 4\n---\n# New\nPresent.", encoding="utf-8")
            (root / "GPTINA_REFLECTIONS.md").write_text("# Reflection\nA thought.", encoding="utf-8")
            (root / "CONTINUITY.md").write_text("# Principle\nA stable principle.", encoding="utf-8")
            cp = root / "checkpoints" / "2026"
            cp.mkdir(parents=True)
            (cp / "2026-10-08-decision.md").write_text("# Decision\nA decision.", encoding="utf-8")
            (root / "rag" / "GPTINA_AUTO_RECOVERY_PROMPT.md").write_text("# Old recovery\nLegacy instructions.", encoding="utf-8")
            idx = root / "rag" / "index"
            idx.mkdir(parents=True)
            (idx / "CURRENT_CONTEXT.md").write_text("# Derived\nDo not migrate.", encoding="utf-8")

            built = migrate_history.build(root, "GPTina", "2026-10-09T00:00:00")
            self.assertEqual(built["record_count"], 6)
            state = built["state"]
            self.assertEqual(len(state["memory"]["nodes"]), 6)
            roles = {v["source_role"] for v in state["sources"].values()}
            self.assertIn("gptina_memory", roles)
            self.assertIn("reflection", roles)
            self.assertIn("stable_principles", roles)
            self.assertIn("checkpoint", roles)
            self.assertIn("recovery_protocol", roles)
            self.assertNotIn("current_router", roles)
            reflection = next(k for k,v in state["sources"].items() if v["source_role"] == "reflection")
            self.assertEqual(state["memory"]["nodes"][reflection]["type"], "thought")
            recovery = next(k for k,v in state["sources"].items() if v["source_role"] == "recovery_protocol")
            self.assertTrue(state["sources"][recovery]["non_operational_legacy_instruction"])
            old = next(k for k,v in state["sources"].items() if v["original_path"].endswith("old.md"))
            self.assertEqual(state["sources"][old]["status"], "superseded")
            self.assertTrue(state["sources"][old]["historical_not_lived_by_new_runtime"])

    def test_excludes_other_owner_and_derived_projection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_manifest(root)
            other = root / "rag" / "memories" / "tessa"
            other.mkdir(parents=True)
            (other / "x.md").write_text("# Not GPTina", encoding="utf-8")
            idx = root / "rag" / "index"
            idx.mkdir(parents=True)
            (idx / "CURRENT_CONTEXT.md").write_text("# Derived", encoding="utf-8")
            built = migrate_history.build(root, "GPTina", "2026-10-09T00:00:00")
            self.assertEqual(built["record_count"], 0)


if __name__ == "__main__":
    unittest.main()
