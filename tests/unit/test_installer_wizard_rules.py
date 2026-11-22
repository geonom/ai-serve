"""installer/wizard_rules.py -- who may talk to the setup page, and what a choice may contain."""
import json

import pytest

from installer import offers as O
from installer import wizard_rules as W

PORT = 49152
TOKEN = "t" * 43
LINUX_NV = O.Machine("Linux", "x86_64", nvidia=True, gpus_mb=(24576,), ram_mb=65536, disk_free_gb=500)
WINDOWS = O.Machine("Windows", "AMD64", nvidia=True, gpus_mb=(24576,), ram_mb=65536, disk_free_gb=500)


def check(method="GET", path="/", host=f"127.0.0.1:{PORT}", origin=None, token=None, ctype=None):
    headers = {"Host": host}
    if origin:
        headers["Origin"] = origin
    if token:
        headers["X-Setup-Token"] = token
    if ctype:
        headers["Content-Type"] = ctype
    return W.check_request(method, path, headers, TOKEN, PORT)


def test_the_token_is_long_and_random():
    a, b = W.make_token(), W.make_token()
    assert a != b and len(a) >= 40


def test_the_page_itself_needs_no_token():
    assert check(path="/") is None
    assert check(path="/wizard.js") is None
    assert check(host=f"localhost:{PORT}") is None


@pytest.mark.parametrize("host", ["evil.example:49152", f"127.0.0.1:{PORT + 1}", "127.0.0.1", "", f"localhost.evil:{PORT}"])
def test_another_host_name_is_refused(host):
    # a page that points its own name at 127.0.0.1 (DNS rebinding) sends its name in Host
    assert check(host=host) is not None
    assert check("GET", "/api/state", host=host, token=TOKEN) is not None


def test_the_api_needs_the_token():
    assert check("GET", "/api/state") is not None
    assert check("GET", "/api/state", token="wrong") is not None
    assert check("GET", "/api/state", token=TOKEN) is None


def test_a_post_needs_json_and_the_token():
    ok = dict(token=TOKEN, ctype="application/json")
    assert check("POST", "/api/install", **ok) is None
    assert check("POST", "/api/install", token=TOKEN, ctype="text/plain") is not None   # a plain form post
    assert check("POST", "/api/install", ctype="application/json") is not None


def test_a_foreign_origin_is_refused_even_with_the_token():
    ok = dict(token=TOKEN, ctype="application/json")
    assert check("POST", "/api/install", origin="https://evil.example", **ok) is not None
    assert check("POST", "/api/install", origin=f"http://127.0.0.1:{PORT}", **ok) is None
    assert check("POST", "/api/install", origin=f"http://localhost:{PORT}", **ok) is None


def test_unknown_paths_and_methods():
    assert check(path="/../etc/passwd") is not None
    assert check(path="/backend/main.py") is not None
    assert check("DELETE", "/api/state", token=TOKEN) is not None
    assert check("GET", "/api/install", token=TOKEN) is not None


def test_a_choice_is_checked_against_this_machine():
    assert W.parse_choice(json.dumps({"offers": ["stt-whisper", "llm-gemma-12b"]}).encode(), LINUX_NV) == \
        ["llm-gemma-12b", "stt-whisper"]
    assert W.parse_choice(b'{"offers": []}', LINUX_NV) == []


@pytest.mark.parametrize("body", [b"", b"[]", b'{"offers": "llm-gemma-12b"}', b'{"offers": [1]}',
                                  b'{"offers": ["nope"]}', b'{"offers": ["llm-gemma-12b", "llm-gemma-e2b"]}',
                                  b'{"offers": [], "extra": 1}', b"{" + b" " * 5000 + b"}"])
def test_a_bad_choice_is_refused(body):
    with pytest.raises(ValueError):
        W.parse_choice(body, LINUX_NV)


def test_offers_that_need_bash_are_not_offered_on_windows_yet():
    usable = {o.id for o in O.offers(WINDOWS) if W.available(o, WINDOWS)[0]}
    assert "llm-gemma-12b" in usable and "tts-chatterbox" not in usable
    with pytest.raises(ValueError):
        W.parse_choice(b'{"offers": ["tts-chatterbox"]}', WINDOWS)


def test_the_job_runs_once_at_a_time():
    job = W.Job()
    assert job.begin() and not job.begin()
    job.end(ok=False, error="x")
    assert job.state == "failed" and job.begin()     # a failed run may be tried again
    job.end(ok=True)
    assert job.state == "done" and not job.begin()
