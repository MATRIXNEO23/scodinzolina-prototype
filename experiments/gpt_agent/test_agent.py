"""Offline contract tests. Canned responses do NOT prove GPT semantic quality."""
import ast
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import agent

AT = "2026-10-08T20:00:00"
PROFILE = {"name": "Test agent", "innate": "Explicit test identity", "learned": "Test fixture only"}


def fixture_response(request):
    prompt = request["prompt"]
    if request["kind"] == "relevance":
        return {node["id"]: 0.5 for node in request["candidates"]}
    if "What 5 high-level insights" in prompt:
        return "The speaker's preference is grounded in the exchange (because of 0)"
    if "most salient high-level questions" in prompt:
        return json.dumps({"output": '["What did the speaker request?"]'})
    if "scale of 1 to 10" in prompt and request["kind"] == "json-output":
        return '{"output": "8"}'
    if "Output a json file" in prompt:
        return '{"output": 1}'
    if request["kind"] == "json-output":
        return '{"output": "A summary grounded only in the supplied test exchange."}'
    if "semantic triple" in prompt or "Subject" in prompt or prompt.rstrip().endswith("(Test agent,"):
        return "notes, supplied test exchange)"
    return "A response grounded only in the supplied test exchange."


def finish(result):
    for _ in range(100):
        if result["status"] != "needs_gpt":
            return result
        request = result["request"]
        result = agent.answer(request["id"], fixture_response(request))
    raise AssertionError("Operation did not terminate")


def concurrent_answer(path, request_id, response, queue):
    agent.STATE = Path(path)
    try:
        result = agent.answer(request_id, response)
        queue.put(result["status"])
    except Exception as exc:
        queue.put(str(exc))


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_state = agent.STATE
        agent.STATE = Path(self.tmp.name) / "state.json"
        agent.initialize(PROFILE, AT)

    def tearDown(self):
        agent.STATE = self.original_state
        self.tmp.cleanup()

    def observe(self, text="The speaker requested a direct answer."):
        return finish(agent.start("observe", AT, text=text, source="fixture://original-turn-1"))

    def test_verbatim_upstream_definitions_and_templates(self):
        source = (agent.ROOT / "upstream.py").read_text()
        definitions = {node.name: ast.get_source_segment(source, node)
                       for node in ast.parse(source).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        provenance = json.loads((agent.ROOT / "SOURCE.json").read_text())
        for item in provenance["definitions"]:
            self.assertEqual(hashlib.sha256(definitions[item["name"]].encode()).hexdigest(), item["sha256"])
        for item in provenance["templates"]:
            self.assertEqual(hashlib.sha256((agent.ROOT / item["path"]).read_bytes()).hexdigest(), item["sha256"])

    def test_no_network_client_or_existing_memory_import(self):
        for file in ("agent.py", "upstream.py"):
            tree = ast.parse((agent.ROOT / file).read_text())
            imports = [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
            imports += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            self.assertFalse(set(imports) & {"openai", "requests", "httpx", "urllib", "socket", "gptina_memory", "live_context"})

    def test_pending_keeps_memory_unmodified_and_invalid_answer_can_retry(self):
        before = agent.load_state()["memory"]
        result = agent.start("observe", AT, text="Original wording (must remain accessible)", source="fixture://turn")
        self.assertEqual(agent.load_state()["memory"], before)
        request = result["request"]
        with self.assertRaises(ValueError):
            agent.answer(request["id"], "invalid triple")
        self.assertEqual(agent.load_state()["pending"]["request"]["id"], request["id"])
        result = finish(agent.answer(request["id"], "notes, original wording)"))
        node_id = result["result"]["node_id"]
        self.assertEqual(agent.load_state()["sources"][node_id]["operation"]["text"], "Original wording (must remain accessible)")

    def test_missing_source_is_not_fabricated_and_local_receipt_is_honest(self):
        result = self.observe()
        self.assertFalse(result["remote_verified"])
        self.assertIn("no complete transcript", result["session_coverage"])
        self.assertEqual(result["state_sha256"], hashlib.sha256(agent.STATE.read_bytes()).hexdigest())
        self.assertEqual(agent.load_state()["sources"]["node_1"]["source"], "fixture://original-turn-1")

    def test_retry_and_conflicting_retry(self):
        result = agent.start("observe", AT, text="Test", source="fixture://turn")
        req = result["request"]
        response = fixture_response(req)
        next_result = agent.answer(req["id"], response)
        self.assertEqual(agent.answer(req["id"], response), next_result)
        with self.assertRaises(ValueError):
            agent.answer(req["id"], "a different answer")
        done = finish(next_result)
        last = agent.load_state()["last"]["answers"][-1]
        self.assertEqual(agent.answer(last["id"], last["response"])["state_sha256"], done["state_sha256"])
        self.assertEqual(len(agent.load_state()["memory"]["nodes"]), 1)

    def test_recall_preserves_original_scoring_and_last_access_after_reopen(self):
        self.observe("First memory")
        self.observe("Second memory")
        result = agent.start("recall", "2026-10-08T21:00:00", text="First memory", limit=1)
        result = agent.answer(result["request"]["id"], {"node_1": 1.0, "node_2": 0.0})
        self.assertEqual(result["result"]["nodes"][0]["id"], "node_1")
        restored = agent.persona_load(agent.load_state())
        self.assertEqual(restored.a_mem.id_to_node["node_1"].last_accessed.isoformat(), "2026-10-08T21:00:00")
        self.assertIsNone(restored.a_mem.embeddings["First memory"])
        self.assertEqual(agent.core.normalize_dict_floats({"a": 3, "b": 5}, 0, 1), {"a": 0, "b": 1})

    def test_relevance_cannot_omit_invent_nodes_or_supply_nan(self):
        self.observe()
        result = agent.start("recall", AT, text="Test")
        req = result["request"]
        for invalid in ({}, {"fake": 1}, {"node_1": float("nan")}, {"node_1": True}):
            with self.assertRaises(ValueError):
                agent.answer(req["id"], invalid)
        finish(result)

    def test_empty_retrieval_and_reflection_threshold(self):
        result = agent.start("recall", AT, text="Nothing yet")
        self.assertEqual(result["result"]["nodes"], [])
        self.assertEqual(agent.start("reflect", AT)["result"]["added"], 0)

    def test_reflection_has_real_evidence_and_original_counter_reset(self):
        self.observe()
        state = agent.load_state()
        state["scratch"]["importance_trigger_curr"] = 0  # Test-only threshold fixture.
        agent.atomic_save(state)
        result = finish(agent.start("reflect", AT))
        self.assertEqual(result["result"]["added"], 1)
        node = agent.load_state()["memory"]["nodes"]["node_2"]
        self.assertEqual(node["filling"], ["node_1"])
        self.assertEqual(node["depth"], 1)
        self.assertEqual(agent.load_state()["scratch"]["importance_trigger_curr"], 150)

    def test_reflection_rejects_fabricated_evidence_before_accepting_answer(self):
        self.observe()
        state = agent.load_state()
        state["scratch"]["importance_trigger_curr"] = 0
        agent.atomic_save(state)
        result = agent.start("reflect", AT)
        while "What 5 high-level insights" not in result["request"]["prompt"]:
            req = result["request"]
            result = agent.answer(req["id"], fixture_response(req))
        req = result["request"]
        with self.assertRaises(ValueError):
            agent.answer(req["id"], "Invented thought (because of 999)")
        self.assertEqual(len(agent.load_state()["memory"]["nodes"]), 1)
        finish(result)

    def test_dialogue_relationship_and_post_conversation_memory(self):
        self.observe()
        reply = finish(agent.start("chat", AT, text="What did I request?", speaker="Test speaker", source="fixture://turn-2"))
        self.assertIn("reply", reply["result"])
        relationship = finish(agent.start("relationship", AT, text="Test speaker"))
        self.assertIn("summary", relationship["result"])
        result = finish(agent.start("consolidate", AT))
        self.assertEqual(len(result["result"]["nodes"]), 2)
        for node_id in result["result"]["nodes"]:
            self.assertEqual(agent.load_state()["memory"]["nodes"][node_id]["filling"], [reply["result"]["chat_node_id"]])

    def test_atomic_crash_before_replace_then_retry(self):
        result = agent.start("observe", AT, text="Test", source="fixture://turn")
        result = agent.answer(result["request"]["id"], fixture_response(result["request"]))
        before = agent.STATE.read_bytes()
        req = result["request"]
        with patch.object(agent.os, "replace", side_effect=OSError("simulated crash")):
            with self.assertRaises(OSError):
                agent.answer(req["id"], fixture_response(req))
        self.assertEqual(agent.STATE.read_bytes(), before)
        self.assertEqual(finish(result)["memory_count"], 1)

    def test_concurrent_identical_final_answers_do_not_duplicate(self):
        result = agent.start("observe", AT, text="Test", source="fixture://turn")
        result = agent.answer(result["request"]["id"], fixture_response(result["request"]))
        req = result["request"]
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        processes = [ctx.Process(target=concurrent_answer, args=(str(agent.STATE), req["id"], fixture_response(req), queue)) for _ in range(2)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual([queue.get(timeout=2) for _ in processes], ["stored_local", "stored_local"])
        self.assertEqual(len(agent.load_state()["memory"]["nodes"]), 1)

    def test_initialization_busy_operation_and_timestamp_boundaries(self):
        with self.assertRaises(ValueError):
            agent.initialize(PROFILE, AT)
        with self.assertRaises(ValueError):
            agent.start("observe", AT + "+02:00", text="Test", source="fixture://turn")
        agent.start("observe", AT, text="Test", source="fixture://turn")
        with self.assertRaises(ValueError):
            agent.start("observe", AT, text="Other", source="fixture://turn-other")

    def test_capture_requires_explicit_source_and_preserves_identical_new_observations(self):
        with self.assertRaises(ValueError):
            agent.start("observe", AT, text="Test")
        self.observe("Same wording")
        self.observe("Same wording")
        self.assertEqual(len(agent.load_state()["memory"]["nodes"]), 2)

    def test_original_research_guard_remains_only_in_analysis_mode(self):
        result = agent.start("analysis", AT, text="Test", speaker="Test speaker", source="fixture://turn")
        self.assertIn("Output a json file", result["request"]["prompt"])
        result = agent.answer(result["request"]["id"], '{"output": 9}')
        self.assertTrue(result["result"]["upstream_analysis_guard"])
        self.assertEqual(result["memory_count"], 0)

    def test_social_initiative_uses_original_yes_no_decision(self):
        result = agent.start("decide-talk", AT, text="Test speaker", own_activity="reviewing the supplied exchange", activity="asking a clarification")
        with self.assertRaises(ValueError):
            agent.answer(result["request"]["id"], "maybe")
        result = agent.answer(result["request"]["id"], "no")
        self.assertFalse(result["result"]["should_talk"])
        self.assertEqual(result["memory_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
