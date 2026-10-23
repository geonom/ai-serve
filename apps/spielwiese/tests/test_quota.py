"""Quota rules: guests get a daily budget of GPU seconds and images and one GPU request at a time; the
owner is unlimited; what is GPU-bound is decided by path, not by the client; batch runs are the owner's;
limits come from an owner-edited file that is checked strictly."""
import json
import os
import tempfile
import threading
import unittest

from spielwiese.quota import (DEFAULT_LIMITS, Ledger, Limits, QuotaError, QuotaExceeded, classify, parse_limits_file,
                              load_limits)


class TestClassify(unittest.TestCase):
    def test_gpu_bound_paths(self):
        for m, p, kind in [("POST", "/v1/chat/completions", "chat"), ("POST", "/api/chat/completions", "chat"),
                           ("POST", "/v1/images/generations", "image"), ("POST", "/api/comfyui/generate", "image"),
                           ("POST", "/api/comfyui/edit", "image"), ("POST", "/api/comfyui/style-transfer", "image"),
                           ("POST", "/api/tts/generate", "speech"), ("POST", "/api/stt/transcribe", "speech"),
                           ("POST", "/api/ocr", "ocr"), ("POST", "/api/comfyui/batch", "owner_only"),
                           ("POST", "/api/plugins/sd-serve/install", "owner_only"),
                           ("POST", "/api/plugins/comfyui/start", "owner_only"),
                           ("POST", "/v1/embeddings", "text"), ("POST", "/v1/rerank", "text"),   # load models too
                           ("POST", "/api/comfyui/batch/pause", "owner_only"),
                           ("GET", "/api/models", None), ("GET", "/v1/images/models", None),
                           ("POST", "/api/tokenize", None), ("GET", "/api/comfyui/task/abc", None),
                           ("POST", "/v1/chat/completions?x=1", "chat")]:
            self.assertEqual(classify(m, p), kind, (m, p))


class TestLimits(unittest.TestCase):
    def test_owner_is_unlimited_guests_get_defaults_or_their_own(self):
        cfg = parse_limits_file({"defaults": {"gpu_seconds_per_day": 600, "images_per_day": 20, "concurrent": 1},
                                 "people": {"anna": {"images_per_day": 50}, "ben": None}})
        self.assertEqual(load_limits(cfg, "alex", True), Limits(None, None, None))
        self.assertEqual(load_limits(cfg, "carl", False), Limits(600, 20, 1))
        self.assertEqual(load_limits(cfg, "anna", False), Limits(600, 50, 1))
        self.assertEqual(load_limits(cfg, "ben", False), Limits(600, 20, 1))
        self.assertEqual(load_limits(parse_limits_file(None), "carl", False), DEFAULT_LIMITS)

    def test_unlimited_per_person(self):
        cfg = parse_limits_file({"people": {"anna": {"gpu_seconds_per_day": None, "images_per_day": None}}})
        self.assertEqual(load_limits(cfg, "anna", False).images_per_day, None)

    def test_file_is_checked_strictly(self):
        for bad in ({"evil": 1}, {"defaults": {"gpu_seconds_per_day": -1}}, {"defaults": {"images_per_day": "5"}},
                    {"defaults": {"concurrent": 0}}, {"defaults": {"x": 1}}, {"people": {"a": {"images_per_day": 1.5}}},
                    {"people": []}, {"defaults": {"images_per_day": True}}, []):
            with self.assertRaises(QuotaError, msg=bad):
                parse_limits_file(bad)


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.day = ["2026-10-19"]
        self.ledger = Ledger(self.tmp.name, today=lambda: self.day[0])
        self.lim = Limits(gpu_seconds_per_day=100, images_per_day=3, concurrent=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_charges_and_refuses_when_used_up(self):
        t = self.ledger.begin("anna", self.lim, "image", images=2)
        self.ledger.end(t, seconds=30, images=2)
        self.assertEqual(self.ledger.usage("anna"), {"day": "2026-10-19", "gpu_seconds": 30, "images": 2})
        with self.assertRaises(QuotaExceeded) as e:
            self.ledger.begin("anna", self.lim, "image", images=2)          # only 1 image left
        self.assertIn("1", str(e.exception))
        self.ledger.end(self.ledger.begin("anna", self.lim, "chat"), seconds=80, images=0)   # 110 s now
        with self.assertRaises(QuotaExceeded):
            self.ledger.begin("anna", self.lim, "chat")                     # seconds used up

    def test_one_request_at_a_time_and_release_on_end(self):
        t = self.ledger.begin("anna", self.lim, "chat")
        with self.assertRaises(QuotaExceeded) as e:
            self.ledger.begin("anna", self.lim, "chat")
        self.assertIn("still running", str(e.exception)); self.assertTrue(e.exception.busy)
        self.ledger.begin("ben", self.lim, "chat")                          # others aren't blocked
        self.ledger.end(t, seconds=1, images=0)
        self.ledger.begin("anna", self.lim, "chat")

    def test_new_day_starts_fresh(self):
        self.ledger.end(self.ledger.begin("anna", self.lim, "image", images=3), seconds=99, images=3)
        self.day[0] = "2026-10-20"
        self.assertEqual(self.ledger.usage("anna")["images"], 0)
        self.ledger.begin("anna", self.lim, "image", images=3)

    def test_unlimited_owner_is_counted_but_never_refused(self):
        free = Limits(None, None, None)
        a = self.ledger.begin("alex", free, "chat")
        b = self.ledger.begin("alex", free, "image", images=100)
        self.ledger.end(a, seconds=5000, images=0)
        self.ledger.end(b, seconds=1, images=100)
        self.assertEqual(self.ledger.usage("alex")["images"], 100)

    def test_survives_restart(self):
        self.ledger.end(self.ledger.begin("anna", self.lim, "image", images=1), seconds=12.5, images=1)
        again = Ledger(self.tmp.name, today=lambda: self.day[0])
        self.assertEqual(again.usage("anna")["gpu_seconds"], 12.5)

    def test_concurrent_begins_cannot_both_pass(self):
        ok = []

        def go():
            try:
                ok.append(self.ledger.begin("anna", self.lim, "chat"))
            except QuotaExceeded:
                pass
        threads = [threading.Thread(target=go) for _ in range(20)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(ok), 1)
