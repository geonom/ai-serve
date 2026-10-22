"""Routing a model request to a node (backend/cluster/routing.py) -- pure.

When this machine can't serve a model (not here, or no VRAM, or Strata owns the
GPUs), the gateway uses an enrolled node (user, 2026-10-16). A chat request must not
hang for a multi-GB copy, so:

1. reuse   -- a node already runs the model (instance ready) -> use it.
2. start   -- a node has the file and an accelerator with room -> start it there.
3. copy    -- otherwise the coordinator copies the model to the best node with room
              in the background and the caller gets 503 + Retry-After; a node that
              refuses the download (quota/disk) is skipped for the next (user:
              "wenn ein Download nicht möglich ist, wird auf ein anderes ausgewichen").

Rules: offline nodes, nodes without a runtime and accelerators the runtime can't use
are never chosen; memory need = model size + KV estimate; room comes from the cluster
ledger (budget, hardware free, estimated running instances); among equals, higher
node priority wins (domain.place)."""
from backend.cluster.routing import NodeSnapshot, plan_route

MODEL = "gemma-12b.gguf"


def snap(node="mac", online=True, runtime_kinds=("metal",), accels=None, models=(), instances=(), priority=0):
    return NodeSnapshot(
        id=node, online=online, priority=priority, runtime_kinds=frozenset(runtime_kinds),
        accelerators=accels if accels is not None else [
            {"index": 0, "kind": "metal", "name": "M3", "total_mb": 24000, "budget_mb": 16000, "free_mb": None}],
        models={m: 7000 for m in models}, instances=list(instances))


def test_reuse_running_instance():
    s = snap(instances=[{"id": "i1", "model": "llama-serve-downloads/gemma-12b.gguf", "state": "ready",
                         "gpu": 0, "ctx_size": 8192}], models=["llama-serve-downloads/gemma-12b.gguf"])
    plan = plan_route(MODEL, 7000, 8192, [s])
    assert plan.action == "reuse" and plan.node == "mac" and plan.instance == "i1"


def test_start_where_file_and_room():
    plan = plan_route(MODEL, 7000, 8192, [snap(models=["gemma-12b.gguf"])])
    assert plan.action == "start" and plan.model_path == "gemma-12b.gguf" and plan.gpu == 0


def test_copy_when_no_node_has_it():
    plan = plan_route(MODEL, 7000, 8192, [snap(models=[])])
    assert plan.action == "copy" and plan.candidates == ["mac"]


def test_copy_candidates_in_order_for_fallback():
    small = snap("pi", runtime_kinds=("cpu",), accels=[
        {"index": 0, "kind": "cpu", "name": "arm", "total_mb": 16000, "budget_mb": 12000, "free_mb": 12000}])
    big = snap("mac", priority=5)
    plan = plan_route(MODEL, 7000, 8192, [small, big])
    assert plan.action == "copy" and plan.candidates == ["mac", "pi"]


def test_no_room_anywhere():
    s = snap(accels=[{"index": 0, "kind": "metal", "name": "M1", "total_mb": 8000, "budget_mb": 4000,
                      "free_mb": None}], models=["gemma-12b.gguf"])
    plan = plan_route(MODEL, 7000, 8192, [s])
    assert plan.action == "none" and "room" in plan.reason


def test_offline_and_no_runtime_skipped():
    plan = plan_route(MODEL, 7000, 8192, [snap(online=False, models=["gemma-12b.gguf"]),
                                         snap("x", runtime_kinds=(), models=["gemma-12b.gguf"])])
    assert plan.action == "none"


def test_runtime_kind_must_match_accelerator():
    # a CPU-only build can't use the node's CUDA GPU
    s = snap(runtime_kinds=("cpu",), accels=[{"index": 0, "kind": "cuda", "name": "4060", "total_mb": 8000,
                                              "budget_mb": 8000, "free_mb": 8000}], models=["gemma-12b.gguf"])
    assert plan_route(MODEL, 7000, 8192, [s]).action == "none"


def test_running_instances_take_room():
    s = snap(accels=[{"index": 0, "kind": "metal", "name": "M3", "total_mb": 24000, "budget_mb": 16000,
                      "free_mb": None}],
             models=["gemma-12b.gguf", "other.gguf"],
             instances=[{"id": "o", "model": "other.gguf", "state": "ready", "gpu": 0, "ctx_size": 8192}])
    # other.gguf (7000 + KV) already uses most of the 16 GB budget
    assert plan_route(MODEL, 9000, 8192, [s]).action == "none"


def test_model_in_subdir_matches_by_basename():
    plan = plan_route(MODEL, 7000, 8192, [snap(models=["llama-serve-downloads/gemma-12b.gguf"])])
    assert plan.action == "start" and plan.model_path == "llama-serve-downloads/gemma-12b.gguf"


class TestModes:
    """A chat server can't answer /v1/embeddings and vice versa: reuse only an
    instance started in the same mode (and with the same pooling when one is asked
    for); otherwise start a new one in that mode. Instances of older agents carry no
    mode and count as chat."""

    def running(self, mode=None, pooling=None):
        inst = {"id": "i1", "model": "bge-m3.gguf", "state": "ready", "gpu": 0, "ctx_size": 8192}
        if mode:
            inst["mode"] = mode
        if pooling:
            inst["pooling"] = pooling
        return snap(models=["bge-m3.gguf"], instances=[inst])

    def test_reuse_same_mode(self):
        plan = plan_route("bge-m3.gguf", 600, 8192, [self.running("embedding")], mode="embedding")
        assert plan.action == "reuse"

    def test_chat_instance_not_reused_for_embeddings(self):
        plan = plan_route("bge-m3.gguf", 600, 8192, [self.running()], mode="embedding")
        assert plan.action == "start" and plan.mode == "embedding"

    def test_pooling_must_match_when_asked(self):
        s = self.running("embedding", "mean")
        assert plan_route("bge-m3.gguf", 600, 8192, [s], mode="embedding", pooling="cls").action == "start"
        assert plan_route("bge-m3.gguf", 600, 8192, [s], mode="embedding").action == "reuse"

    def test_reranking(self):
        plan = plan_route("bge-reranker.gguf", 600, 8192, [snap(models=["bge-reranker.gguf"])], mode="reranking")
        assert plan.action == "start" and plan.mode == "reranking"
