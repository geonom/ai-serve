"""Hugging Face GGUF downloads through the dashboard (backend/routers/hf_download.py + backend/hf_download.py):
the listing comes from Hugging Face, every file is checked against its sha256 while streaming, a bad file is never
kept, the token never travels to the CDN. Rules: test_hf_models.py."""
import hashlib
import io
import time
import urllib.request

import pytest
from fastapi.testclient import TestClient

from backend import hf_download

DATA = b"GGUF fake model bytes"
PROJ = b"GGUF fake projector"


def tree(model_sha=None):
    return [{"type": "file", "path": "M-Q4_K_M.gguf", "size": len(DATA),
             "lfs": {"oid": model_sha or hashlib.sha256(DATA).hexdigest(), "size": len(DATA)}},
            {"type": "file", "path": "mmproj-F16.gguf", "size": len(PROJ),
             "lfs": {"oid": hashlib.sha256(PROJ).hexdigest(), "size": len(PROJ)}}]


@pytest.fixture
def client(tmp_path, monkeypatch):
    from backend.main import app
    monkeypatch.setattr(hf_download, "model_dir", lambda: str(tmp_path))
    monkeypatch.setattr(hf_download, "_jobs", {})
    fetched = []

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    def opener(req, timeout=None):
        fetched.append((req.full_url, dict(req.header_items())))
        return R(PROJ if "mmproj" in req.full_url else DATA)
    monkeypatch.setattr(hf_download._OPENER, "open", opener)
    c = TestClient(app, client=("127.0.0.1", 50000))
    c.fetched, c.dir = fetched, tmp_path
    return c


def wait(c, model):
    for _ in range(100):
        j = {d["model"]: d for d in c.get("/api/models/hf/downloads").json()["downloads"]}.get(model)
        if j and j["state"] != "downloading":
            return j
        time.sleep(0.02)
    raise AssertionError("download didn't finish")


def test_inspect_lists_the_repo(client, monkeypatch):
    monkeypatch.setattr(hf_download, "get_tree", lambda repo, rev: tree())
    r = client.post("/api/models/hf/inspect", json={"url": "https://huggingface.co/someone/M-GGUF"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["repo"] == "someone/M-GGUF" and [m["path"] for m in body["models"]] == ["M-Q4_K_M.gguf"]
    assert [p["path"] for p in body["mmproj"]] == ["mmproj-F16.gguf"] and body["selected"] is None


def test_download_checks_and_pairs(client, monkeypatch):
    monkeypatch.setattr(hf_download, "get_tree", lambda repo, rev: tree())
    r = client.post("/api/models/hf/download", json={"url": "someone/M-GGUF", "model": "M-Q4_K_M.gguf",
                                                     "mmproj": "mmproj-F16.gguf"})
    assert r.status_code == 202, r.text
    assert wait(client, "M-Q4_K_M.gguf")["state"] == "installed"
    assert (client.dir / "M-Q4_K_M.gguf").read_bytes() == DATA
    assert (client.dir / "M-Q4_K_M-mmproj-F16.gguf").read_bytes() == PROJ
    assert all(u.startswith("https://huggingface.co/someone/M-GGUF/resolve/main/") for u, _ in client.fetched)


def test_a_wrong_checksum_keeps_nothing(client, monkeypatch):
    monkeypatch.setattr(hf_download, "get_tree", lambda repo, rev: tree(model_sha="0" * 64))
    client.post("/api/models/hf/download", json={"url": "someone/M-GGUF", "model": "M-Q4_K_M.gguf"})
    j = wait(client, "M-Q4_K_M.gguf")
    assert j["state"] == "failed" and "checksum" in j["error"]
    assert list(client.dir.iterdir()) == []


def test_bad_input_is_refused(client, monkeypatch):
    monkeypatch.setattr(hf_download, "get_tree", lambda repo, rev: tree())
    for body in ({"url": "https://evil.example/a/b"}, {"url": "someone/M-GGUF", "model": "../x.gguf"},
                 {"url": "someone/M-GGUF", "model": "M-Q4_K_M.gguf", "path": "/etc"}, {"url": 3}):
        assert client.post("/api/models/hf/download", json=body).status_code == 400, body


def test_the_token_stays_on_huggingface(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    url = "https://huggingface.co/a/b/resolve/main/x.gguf"
    req = urllib.request.Request(url, headers=hf_download._headers(url))
    assert req.get_header("Authorization") == "Bearer hf_secret"
    new = hf_download._NoAuthOnRedirect().redirect_request(req, None, 302, "Found", {}, "https://cdn-lfs.hf.co/x")
    assert new.get_header("Authorization") is None and not new.unredirected_hdrs.get("Authorization")
    assert "Authorization" not in hf_download._headers("https://cdn-lfs.hf.co/x")
