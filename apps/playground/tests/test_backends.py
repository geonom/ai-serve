"""Without ai-serve (owner, 2026-10-21: the Playground should also work with standard Ollama or LM Studio): the
gateway's catalogs are rebuilt from the standard OpenAI API, everything that needs ai-serve reports itself
unavailable instead of failing."""
import unittest

from playground import backends


class TestCatalogFromStandardServer(unittest.TestCase):
    def test_models_become_chat_rows(self):
        cat = backends.llm_catalog({"object": "list", "data": [
            {"id": "llama3.2:3b", "object": "model"}, {"id": "qwen/qwen3-8b"},
            {"id": "nomic-embed-text:latest"}, {"id": "bge-reranker-v2-m3"}, {"bad": 1}, "x"]})
        rows = {r["id"]: r for r in cat["models"]}
        self.assertEqual(set(rows), {"llama3.2:3b", "qwen/qwen3-8b", "nomic-embed-text:latest", "bge-reranker-v2-m3"})
        self.assertEqual(rows["llama3.2:3b"]["kind"], "chat")
        self.assertEqual(rows["nomic-embed-text:latest"]["kind"], "embedding")
        self.assertEqual(rows["bge-reranker-v2-m3"]["kind"], "rerank")
        self.assertEqual(rows["qwen/qwen3-8b"]["where"], [{"place": "local", "loaded": False}])
        self.assertFalse(rows["llama3.2:3b"]["exclusive"])

    def test_llama_server_and_ollama_style_lists(self):
        # llama-server answers {"models": [...], "data": [...]}, some servers only {"models": [{"name": ...}]}
        cat = backends.llm_catalog({"models": [{"name": "a", "model": "a"}]})
        self.assertEqual([r["id"] for r in cat["models"]], ["a"])

    def test_unreadable(self):
        self.assertEqual(backends.llm_catalog("<html>")["models"], [])

    def test_what_needs_ai_serve_is_unavailable_with_a_reason(self):
        audio = backends.EMPTY["/v1/audio/models"]
        self.assertFalse(audio["tts"]["available"])
        self.assertFalse(audio["stt"]["available"])
        self.assertIn("ai-serve", audio["tts"]["reason"])
        self.assertEqual(backends.EMPTY["/v1/images/models"]["data"], [])


if __name__ == "__main__":
    unittest.main()


class TestWhatTheUpstreamIs(unittest.TestCase):
    """A slow answer is not a different server (2026-11-22: ai-serve's catalog took > 5 s right after a restart and
    while the backend stalled; the Playground then said "gateway not running" or switched images and speech off)."""

    def test_ai_serve_answers_its_catalog(self):
        self.assertEqual(backends.upstream_kind(None, 200, None), "ai-serve")

    def test_no_answer_from_a_known_ai_serve_keeps_it(self):
        self.assertEqual(backends.upstream_kind("ai-serve", 0, None), "ai-serve")

    def test_a_standard_server_has_only_the_model_list(self):
        self.assertEqual(backends.upstream_kind(None, 404, 200), "openai")
        self.assertEqual(backends.upstream_kind("ai-serve", 404, 200), "openai")   # it really answered: not ai-serve

    def test_nothing_answers(self):
        self.assertEqual(backends.upstream_kind(None, 0, 0), "down")
        self.assertEqual(backends.upstream_kind("openai", 0, 0), "down")

    def test_only_a_sure_answer_is_kept_long(self):
        self.assertEqual(backends.cache_seconds(200), 30)
        self.assertEqual(backends.cache_seconds(0), 5)
