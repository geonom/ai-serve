"""The comparison runner: one image after the other, as the person who asked (their address goes to the
gateway), each into their own gallery; failures stay per cell; one running comparison per person."""
import base64
import tempfile
import threading
import unittest

from spielwiese.compare import CompareError, Runner
from spielwiese.gallery import Gallery

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16


def same_lane(plan):
    for c in plan["cells"]:
        c["lane"] = "one"
    return plan


def plan_of(*cells, lane=None):
    """lane=None: one lane per model (they run in parallel); a name: all cells one after the other in that lane."""
    return {"prompt": "a fox", "aspect_ratio": "16:9",
            "cells": [{"model": m, "label": m.upper(), "steps": s, "seed": 7, "send_steps": send, "aspect_ratio": "16:9",
                       "lane": lane or m} for m, s, send in cells]}


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sent = []
        self.answers = {}

        def post(body, peer, request_id=None):
            self.sent.append((body, peer))
            return self.answers.get(body["model"], (200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}))
        self.gallery = Gallery(self.tmp.name)
        self.runner = Runner(self.tmp.name, self.gallery, post)

    def tearDown(self):
        self.tmp.cleanup()

    def test_runs_every_cell_as_the_person(self):
        job = self.runner.start("anna", "100.64.0.9", plan_of(("z", 8, True), ("iris", 4, False)), wait=True)
        sent = {b["model"]: b for b, _ in self.sent}               # lanes run in parallel: order varies
        self.assertEqual(sent["z"], {"model": "z", "prompt": "a fox", "aspect_ratio": "16:9", "seed": 7,
                                     "response_format": "b64_json", "steps": 8})
        self.assertNotIn("steps", sent["iris"])                   # Iris: its fixed schedule
        self.assertEqual({p for _, p in self.sent}, {"100.64.0.9"})
        done = self.runner.get("anna", job["id"])
        self.assertEqual(done["status"], "done")
        self.assertEqual([c["status"] for c in done["cells"]], ["done", "done"])
        images = self.gallery.list("anna", comparison=job["id"])
        self.assertEqual({e["model"] for e in images}, {"z", "iris"})
        self.assertEqual(self.gallery.list("ben"), [])
        self.assertIsNone(self.runner.get("ben", job["id"]))   # comparisons are per person too

    def test_a_failing_cell_doesnt_stop_the_rest(self):
        self.answers["moss"] = (500, {"detail": "MOSS-Image failed: CUDA error"})   # not retried (507 is, see TestBusyGpu)
        job = self.runner.start("anna", "127.0.0.1", plan_of(("moss", 4, True), ("z", 8, True)), wait=True)
        cells = self.runner.get("anna", job["id"])["cells"]
        self.assertEqual([c["status"] for c in cells], ["failed", "done"])
        self.assertIn("CUDA error", cells[0]["error"])

    def test_one_running_comparison_per_person_and_cancel(self):
        gate = threading.Event()

        started = threading.Event()

        def slow(body, peer, request_id=None):
            started.set()
            gate.wait(5)
            return 200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}
        self.runner.post = slow
        # one lane: the cells run one after the other, so the order is fixed (parallel lanes made this flaky)
        job = self.runner.start("anna", "127.0.0.1", plan_of(("a", 4, True), ("b", 4, True), ("c", 4, True), lane="gpu"))
        self.assertTrue(started.wait(5))
        with self.assertRaises(CompareError):
            self.runner.start("anna", "127.0.0.1", plan_of(("a", 4, True)))
        self.assertFalse(self.runner.cancel("ben", job["id"]))   # someone else's
        self.assertTrue(self.runner.cancel("anna", job["id"]))
        gate.set()
        for _ in range(100):
            if self.runner.get("anna", job["id"])["status"] != "running":
                break
            threading.Event().wait(0.05)
        done = self.runner.get("anna", job["id"])
        self.assertEqual(done["status"], "cancelled")
        self.assertEqual([c["status"] for c in done["cells"]], ["done", "cancelled", "cancelled"])
        self.runner.start("anna", "127.0.0.1", plan_of(("a", 4, True)), wait=True)   # free again

    def test_list_and_bad_ids(self):
        job = self.runner.start("anna", "127.0.0.1", plan_of(("z", 8, True)), wait=True)
        self.assertEqual([j["id"] for j in self.runner.list("anna")], [job["id"]])
        self.assertEqual(self.runner.list("anna")[0]["models"], ["Z"])
        for bad in ("../x", "", "ABC"):
            self.assertIsNone(self.runner.get("anna", bad))


class TestBusyGpu(unittest.TestCase):
    """The gateway answers 'no room' (507) fast and expects callers to retry -- a comparison is such a caller.
    It used to fail every remaining cell while an earlier, slow cell still held the GPU (2026-10-19)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gallery = Gallery(self.tmp.name)
        self.slept = []

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, answers):
        it = iter(answers)

        def post(body, peer, request_id=None):
            return next(it)
        r = Runner(self.tmp.name, self.gallery, post)
        r.sleep = self.slept.append
        return r

    def test_507_is_retried_until_the_gpu_is_free(self):
        busy = (507, {"detail": "MOSS-Image: not enough GPU memory -- held by ComfyUI (busy, 12724 MB)"})
        ok = (200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}]})
        r = self.runner([busy, busy, ok])
        job = r.start("anna", "127.0.0.1", plan_of(("moss", 4, True)), wait=True)
        cell = r.get("anna", job["id"])["cells"][0]
        self.assertEqual(cell["status"], "done")
        self.assertEqual(sum(self.slept), 20)          # two waits of 10 s, slept in 1 s slices

    def test_gives_up_after_the_wait_limit_with_the_reason(self):
        busy = (507, {"detail": "Iris: not enough GPU memory -- held by ComfyUI (busy)"})
        r = self.runner([busy] * 1000)
        r.busy_wait_s = 30          # 3 tries of 10 s
        job = r.start("anna", "127.0.0.1", plan_of(("iris", 4, False)), wait=True)
        cell = r.get("anna", job["id"])["cells"][0]
        self.assertEqual(cell["status"], "failed")
        self.assertIn("GPU", cell["error"])

    def test_other_errors_are_not_retried(self):
        r = self.runner([(400, {"detail": "steps must be 1..60"}), (200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}]})])
        job = r.start("anna", "127.0.0.1", plan_of(("a", 99, True), ("b", 4, True), lane="gpu"), wait=True)
        self.assertEqual([c["status"] for c in r.get("anna", job["id"])["cells"]], ["failed", "done"])
        self.assertEqual(self.slept, [])

    def test_cancel_while_waiting_stops_at_once(self):
        """Users may rather cancel than wait for the GPU (user, 2026-10-19)."""
        busy = (507, {"detail": "held by ComfyUI (busy)"})
        calls = []

        def post(body, peer, request_id=None):
            calls.append(body["model"])
            return busy
        r = Runner(self.tmp.name, self.gallery, post)
        job_box = {}

        def sleep(s):
            self.slept.append(s)
            if len(self.slept) == 3:          # three seconds into the first wait
                r.cancel("anna", job_box["id"])
        r.sleep = sleep
        orig_start = r.start

        def start(*a, **k):
            k["wait"] = False
            job = orig_start(*a, **k)
            job_box["id"] = job["id"]
            return job
        job = start("anna", "127.0.0.1", same_lane(plan_of(("moss", 4, True), ("iris", 4, False))))
        for _ in range(200):
            if r.get("anna", job["id"])["status"] != "running":
                break
            threading.Event().wait(0.01)
        done = r.get("anna", job["id"])
        self.assertEqual(done["status"], "cancelled")
        self.assertEqual([c["status"] for c in done["cells"]], ["cancelled", "cancelled"])
        self.assertEqual(calls, ["moss"])                # no further tries, the second cell never asked
        self.assertLess(sum(self.slept), 10)


class TestStopRunningImage(unittest.TestCase):
    """Cancel stops the image that is being generated right now, not only the cells after it
    (user, 2026-10-19: "it works, but only after the next image is generated fully")."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_cancel_reaches_the_gateway_with_the_cells_request_id(self):
        started, stopped = threading.Event(), threading.Event()
        seen = {}

        def post(body, peer, request_id=None):
            seen["rid"] = request_id
            started.set()
            stopped.wait(5)                      # the gateway answers once the job was interrupted
            return 409, {"detail": "Image generation cancelled"}

        def cancel_remote(request_id, peer):
            seen["cancelled"] = (request_id, peer)
            stopped.set()
        r = Runner(self.tmp.name, Gallery(self.tmp.name), post, cancel_remote=cancel_remote)
        job = r.start("anna", "100.64.0.9", same_lane(plan_of(("moss", 30, True), ("iris", 4, False))))
        started.wait(5)
        self.assertTrue(r.cancel("anna", job["id"]))
        for _ in range(200):
            if r.get("anna", job["id"])["status"] != "running":
                break
            threading.Event().wait(0.01)
        done = r.get("anna", job["id"])
        self.assertEqual(done["status"], "cancelled")
        self.assertEqual([c["status"] for c in done["cells"]], ["cancelled", "cancelled"])
        self.assertEqual(seen["cancelled"], (seen["rid"], "100.64.0.9"))
        self.assertGreaterEqual(len(seen["rid"]), 32)


class TestGatewayAway(unittest.TestCase):
    """A gateway that is restarting refuses connections for a few seconds; every remaining cell used to fail
    at once with "[Errno 111] Connection refused" (2026-10-19). It's waited out like a busy GPU."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_connection_refused_is_retried(self):
        answers = [ConnectionRefusedError(111, "Connection refused")] * 2 + [
            (200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}]})]

        def post(body, peer, request_id=None):
            a = answers.pop(0)
            if isinstance(a, Exception):
                raise a
            return a
        r = Runner(self.tmp.name, Gallery(self.tmp.name), post)
        slept = []
        r.sleep = slept.append
        job = r.start("anna", "127.0.0.1", plan_of(("moss", 4, True)), wait=True)
        self.assertEqual(r.get("anna", job["id"])["cells"][0]["status"], "done")
        self.assertEqual(sum(slept), 20)


class TestParallelLanes(unittest.TestCase):
    """Models of different services (ComfyUI, MOSS, Iris) can run at the same time -- on different GPUs they
    don't wait for each other (owner, 2026-10-20). Within one service cells stay one after the other."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_lanes_run_at_the_same_time(self):
        both = threading.Barrier(2, timeout=5)
        order = []

        def post(body, peer, request_id=None):
            order.append(body["model"])
            both.wait()                                   # only passes if the other lane asks meanwhile
            return 200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}
        r = Runner(self.tmp.name, Gallery(self.tmp.name), post)
        p = plan_of(("moss", 4, True), ("iris", 4, False))
        job = r.start("anna", "127.0.0.1", p, wait=True)
        done = r.get("anna", job["id"])
        self.assertEqual([c["status"] for c in done["cells"]], ["done", "done"])
        self.assertEqual(done["status"], "done")

    def test_one_lane_is_sequential(self):
        running, peak = [0], [0]
        lock = threading.Lock()

        def post(body, peer, request_id=None):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            threading.Event().wait(0.05)
            with lock:
                running[0] -= 1
            return 200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}
        r = Runner(self.tmp.name, Gallery(self.tmp.name), post)
        p = plan_of(("z", 4, True), ("z", 8, True), ("z", 16, True))
        r.start("anna", "127.0.0.1", p, wait=True)
        self.assertEqual(peak[0], 1)

    def test_cancel_stops_every_running_image(self):
        started = threading.Barrier(3, timeout=5)
        release = threading.Event()
        cancelled = []

        def post(body, peer, request_id=None):
            started.wait()
            release.wait(5)
            return 409, {"detail": "Image generation cancelled"}

        def cancel_remote(rid, peer):
            cancelled.append(rid)
            if len(cancelled) == 2:
                release.set()
        r = Runner(self.tmp.name, Gallery(self.tmp.name), post, cancel_remote=cancel_remote)
        job = r.start("anna", "127.0.0.1", plan_of(("moss", 4, True), ("iris", 4, False)))
        started.wait()
        r.cancel("anna", job["id"])
        for _ in range(300):
            if r.get("anna", job["id"])["status"] != "running":
                break
            threading.Event().wait(0.01)
        self.assertEqual(len(set(cancelled)), 2)
        self.assertEqual(r.get("anna", job["id"])["status"], "cancelled")
