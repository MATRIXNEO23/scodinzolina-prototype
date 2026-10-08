from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import migrate_history


class MigrationTests(unittest.TestCase):
    def test_migrates_history_with_provenance_and_status(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mem = root / "rag" / "memories" / "gptina" / "2026" / "10"
            mem.mkdir(parents=True)
            (mem / "a.md").write_text(
                """---\nmemory_id: \"m1\"\nowner: gptina\nrecorded_at: \"2026-10-07T06:03:27+02:00\"\nstatus: current\nimportance: 5\nconfidence: \"verified\"\ntags:\n  - \"alpha\"\nsupersedes: []\n---\n# Primo\n\nContenuto uno.\n""",
                encoding="utf-8")
            (mem / "b.md").write_text(
                """---\nmemory_id: \"m2\"\nowner: gptina\nrecorded_at: \"2026-10-08T21:28:00+02:00\"\nstatus: superseded\nimportance: 2\nconfidence: \"verified\"\nsupersedes:\n  - \"m1\"\n---\n# Secondo\n\nContenuto due.\n""",
                encoding="utf-8")
            built = migrate_history.build(root, "GPTina", "2026-10-08T20:00:00")
            self.assertEqual(built["record_count"], 2)
            state = built["state"]
            self.assertEqual(len(state["memory"]["nodes"]), 2)
            self.assertEqual(state["memory"]["nodes"]["node_1"]["created"], "2026-10-07 04:03:27")
            self.assertEqual(state["memory"]["nodes"]["node_1"]["poignancy"], 10)
            self.assertEqual(state["sources"]["node_2"]["original_metadata"]["status"], "superseded")
            self.assertTrue(state["sources"]["node_2"]["historical_not_lived_by_new_runtime"])
            self.assertIsNone(state["memory"]["embeddings"][state["memory"]["nodes"]["node_1"]["embedding_key"]])

    def test_does_not_import_other_owner_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "rag" / "memories" / "gptina").mkdir(parents=True)
            other = root / "rag" / "memories" / "tessa"
            other.mkdir(parents=True)
            (other / "x.md").write_text("# Not GPTina", encoding="utf-8")
            built = migrate_history.build(root, "GPTina", "2026-10-08T20:00:00")
            self.assertEqual(built["record_count"], 0)


if __name__ == "__main__":
    unittest.main()
