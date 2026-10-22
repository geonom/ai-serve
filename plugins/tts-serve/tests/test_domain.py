"""tts-serve domain (tts_serve/domain.py): provider manifests, registry, the common
request, VRAM needs and eviction order -- pure rules, no processes, no GPU.

Threat model: tts-serve listens on 127.0.0.1 and starts processes described by
provider manifests in its own providers/ directory (trust anchor: this repo). A
request must never make it run something else, read an arbitrary file, or pass
unchecked data on as process arguments."""
import base64
import json
import stat

import pytest

from tts_serve.domain import (
    ProviderError,
    ProviderManifest,
    ProviderRegistry,
    RequestError,
    TtsRequest,
    UnknownEngine,
    eviction_order,
    overall_status,
    vram_need_mb,
)

WAV = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEfmt ").decode()


def _provider(root, pid="cosyvoice", **over):
    d = root / pid
    (d / "venv" / "bin").mkdir(parents=True, exist_ok=True)
    py = d / "venv" / "bin" / "python"
    py.write_text("#!/bin/sh\n")
    py.chmod(py.stat().st_mode | stat.S_IXUSR)
    (d / "worker.py").write_text("")
    (d / "adapter.py").write_text("def build_payload(req, ref_audio_path):\n    return {'text': req.text}\n")
    data = {
        "id": pid, "name": pid.title(), "capability": "tts",
        "command": ["./venv/bin/python", "worker.py", "--port", "{port}"],
        "env": {"REPO": "{dir}/repo"}, "vram_mb": 3700,
    }
    data.update(over)
    (d / "llama-serve.provider.json").write_text(json.dumps(data))
    return d


def _manifest(tmp_path, **over):
    d = _provider(tmp_path, **over)
    return ProviderManifest.from_dict(json.loads((d / "llama-serve.provider.json").read_text()), d)


# --------------------------------------------------------------------------- manifest

class TestManifest:
    def test_valid_with_defaults(self, tmp_path):
        m = _manifest(tmp_path)
        assert (m.id, m.gpu, m.keep_alive_s, m.startup_timeout_s, m.concurrent) == ("cosyvoice", 0, 60, 60, False)
        assert m.argv(4242) == [str(m.dir / "venv/bin/python"), "worker.py", "--port", "4242"]
        assert m.environment(4242) == {"REPO": f"{m.dir}/repo"}

    def test_interpreter_symlink_outside_is_fine(self, tmp_path):
        # uv venvs symlink bin/python to a shared interpreter -- normal, allowed.
        d = _provider(tmp_path)
        py = d / "venv" / "bin" / "python"
        py.unlink()
        real = tmp_path / "python3.12"
        real.write_text("#!/bin/sh\n")
        real.chmod(0o755)
        py.symlink_to(real)
        ProviderManifest.from_dict(json.loads((d / "llama-serve.provider.json").read_text()), d)

    @pytest.mark.parametrize("command", [
        [], "./venv/bin/python", ["python3", "worker.py"], ["/usr/bin/python3", "worker.py"],
        ["./../x/python", "w.py"], ["./venv/bin/missing"], ["./venv/bin/python", 5],
    ])
    def test_command_must_start_with_an_executable_inside_the_provider(self, tmp_path, command):
        with pytest.raises(ProviderError):
            _manifest(tmp_path, command=command)

    @pytest.mark.parametrize("env", [{"LD_PRELOAD": "x"}, {"PATH": "x"}, {"PYTHONSTARTUP": "x"},
                                     {"lower": "x"}, {"X": 1}])
    def test_env_rules(self, tmp_path, env):
        with pytest.raises(ProviderError):
            _manifest(tmp_path, env=env)

    @pytest.mark.parametrize("over", [
        {"id": "Cosy Voice"}, {"capability": "stt"}, {"gpu": -1}, {"vram_mb": 0},
        {"vram_mb": 100, "vram_fraction": 0.5}, {"vram_mb": None},
        {"vram_mb": None, "vram_fraction": 1.5}, {"keep_alive_s": -5}, {"concurrent": "yes"},
        {"aliases": ["Bad Alias"]}, {"variants": "zero_shot"}, {"comand": []},
    ])
    def test_invalid_fields(self, tmp_path, over):
        with pytest.raises(ProviderError):
            _manifest(tmp_path, **over)

    def test_vram_fraction(self, tmp_path):
        m = _manifest(tmp_path, vram_mb=None, vram_fraction=0.5, env={"MEM": "{vram_fraction}"})
        assert m.environment(1) == {"MEM": "0.5"}
        assert vram_need_mb(m, gpu_total_mb=24576) == int(24576 * 0.5) + 500 + 512

    def test_vram_mb_need_includes_margin(self, tmp_path):
        assert vram_need_mb(_manifest(tmp_path), gpu_total_mb=24576) == 3700 + 512

    def test_comment_keys_allowed(self, tmp_path):
        _manifest(tmp_path, _comment="x")


# --------------------------------------------------------------------------- registry

class TestRegistry:
    def test_loads_providers_and_reports_broken_ones(self, tmp_path):
        _provider(tmp_path, "cosyvoice", aliases=["cosy", "cosyvoice3"])
        _provider(tmp_path, "chatterbox", aliases=["chatter"])
        _provider(tmp_path, "broken", command=["bash", "-c", "x"])
        (tmp_path / "no-manifest").mkdir()
        reg = ProviderRegistry.load(tmp_path)
        assert [p.id for p in reg.providers] == ["chatterbox", "cosyvoice"]
        assert len(reg.errors) == 1 and "broken" in reg.errors[0]["dir"]

    def test_resolve_by_id_or_alias_case_insensitive(self, tmp_path):
        _provider(tmp_path, "cosyvoice", aliases=["cosy"])
        reg = ProviderRegistry.load(tmp_path)
        assert reg.resolve("COSY").id == "cosyvoice" and reg.resolve("cosyvoice").id == "cosyvoice"

    def test_unknown_engine(self, tmp_path):
        _provider(tmp_path, "cosyvoice")
        with pytest.raises(UnknownEngine) as e:
            ProviderRegistry.load(tmp_path).resolve("bogus")
        assert "cosyvoice" in e.value.message

    def test_alias_clash_rejected(self, tmp_path):
        _provider(tmp_path, "a", aliases=["x"])
        _provider(tmp_path, "b", aliases=["x"])
        reg = ProviderRegistry.load(tmp_path)
        assert len(reg.providers) == 1 and len(reg.errors) == 1

    def test_symlinked_provider_dir_ignored(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        _provider(outside, "evil")
        root = tmp_path / "providers"
        root.mkdir()
        (root / "evil").symlink_to(outside / "evil", target_is_directory=True)
        assert ProviderRegistry.load(root).providers == []


# --------------------------------------------------------------------------- request

class TestRequest:
    def test_minimal(self):
        r = TtsRequest.from_dict({"text": "Hallo"})
        assert (r.text, r.language, r.engine, r.response_format, r.ref_audio) == ("Hallo", "en", None, "json", None)

    def test_full(self):
        r = TtsRequest.from_dict({
            "text": "Hallo", "engine": "cosyvoice", "language": "de", "variant": "instruct",
            "ref_audio_base64": WAV, "ref_text": "x", "instruct": "langsam", "speaker": "Vivian",
            "params": {"temperature": 0.7, "exaggeration": 0.5, "cfg_weight": 2}, "keep_alive": 120,
            "response_format": "wav",
        })
        assert r.ref_audio.startswith(b"RIFF") and r.params == {"temperature": 0.7, "exaggeration": 0.5, "cfg_weight": 2.0}
        assert r.keep_alive == 120 and r.response_format == "wav"

    @pytest.mark.parametrize("body", [
        {}, {"text": ""}, {"text": "   "}, {"text": 5}, {"text": "x" * 20001},
        {"text": "x", "language": "de; rm -rf /"}, {"text": "x", "ref_audio_base64": "!!!notbase64"},
        {"text": "x", "params": {"temperature": "hot"}}, {"text": "x", "params": {"ex ec": 1}},
        {"text": "x", "keep_alive": -1}, {"text": "x", "keep_alive": 10 ** 9}, {"text": "x", "response_format": "exe"},
        {"text": "x", "ref_audio_path": "/etc/passwd"},  # paths are never accepted
        {"text": "x", "variant": "../../x"}, {"text": "x", "engine": "a b"},
    ])
    def test_rejected(self, body):
        with pytest.raises(RequestError):
            TtsRequest.from_dict(body)

    def test_reference_size_limit(self):
        big = base64.b64encode(b"\0" * (TtsRequest.MAX_REF_BYTES + 1)).decode()
        with pytest.raises(RequestError):
            TtsRequest.from_dict({"text": "x", "ref_audio_base64": big})


# --------------------------------------------------------------------------- eviction / status

class TestEviction:
    def test_least_recently_used_idle_on_same_gpu_first(self):
        engines = [
            {"id": "a", "gpu": 0, "idle": True, "last_used": 30},
            {"id": "b", "gpu": 0, "idle": True, "last_used": 10},
            {"id": "c", "gpu": 0, "idle": False, "last_used": 1},   # busy: never
            {"id": "d", "gpu": 1, "idle": True, "last_used": 0},    # other GPU
            {"id": "target", "gpu": 0, "idle": True, "last_used": 0},
        ]
        assert eviction_order("target", 0, engines) == ["b", "a"]


class TestOverallStatus:
    def test_most_severe_wins(self):
        assert overall_status(["idle", "generating", "unloaded"]) == "generating"
        assert overall_status(["idle", "error"]) == "error"
        assert overall_status([]) == "unloaded"


def test_bundled_providers_are_valid():
    """Every provider shipped in providers/ loads, or (fresh checkout, setup.sh not run) says it isn't set up yet --
    never a broken manifest."""
    from tts_serve.providers import PROVIDERS_DIR
    reg = ProviderRegistry.load(PROVIDERS_DIR)
    ids = {p.id for p in reg.providers}
    for e in reg.errors:
        assert "not set up yet" in e["error"] or "not an executable" in e["error"], e
    shipped = {d.name for d in PROVIDERS_DIR.iterdir() if (d / "llama-serve.provider.json").is_file()}
    assert {"cosyvoice", "chatterbox", "chatterbox-mlx", "voxcpm", "qwen3", "omnivoice"} <= shipped
    assert len(ids) + len(reg.errors) == len(shipped)


class TestStatusFacts:
    """Short plain facts for ai-serve-ui's overview (GET /v1/status -> "facts")."""

    def _e(self, id, name, status, last_used=None, inflight=0):
        return {"id": id, "name": name, "status": status, "last_used": last_used, "inflight": inflight}

    def test_nothing_loaded(self):
        from tts_serve.domain import status_facts
        facts = status_facts([self._e("cosyvoice", "CosyVoice 3", "unloaded")], "cosyvoice", now=1000.0)
        assert facts == [{"label": "Geladen", "value": "keine Engine"},
                         {"label": "Standard", "value": "CosyVoice 3"}]

    def test_loaded_working_and_last_use(self):
        from tts_serve.domain import status_facts
        engines = [self._e("cosyvoice", "CosyVoice 3", "idle", last_used=1000.0 - 300),
                   self._e("omnivoice", "OmniVoice", "generating", last_used=1000.0 - 5, inflight=1),
                   self._e("chatterbox", "Chatterbox", "unloaded", last_used=10.0)]
        facts = {f["label"]: f["value"] for f in status_facts(engines, "cosyvoice", now=1000.0)}
        assert facts["Geladen"] == "CosyVoice 3, OmniVoice"
        assert facts["Arbeitet"] == "OmniVoice"
        assert facts["Zuletzt genutzt"] == "OmniVoice, vor 5 s"

    def test_error_reported(self):
        from tts_serve.domain import status_facts
        facts = {f["label"]: f["value"] for f in status_facts([self._e("voxcpm", "VoxCPM2", "error")], "cosyvoice", now=1.0)}
        assert facts["Fehler"] == "VoxCPM2"


# --------------------------------------------------------------------------- modes (self-description)
# Each engine says what it can do and which inputs each mode needs, so a client (the Spielwiese,
# StoryTime) can build its form from /v1/engines instead of knowing every engine (owner, 2026-10-20:
# "möglichst offen für zukünftige Modelle").

MODES = [
    {"variant": "a", "label": "Klonen", "description": "Stimme aus einer Aufnahme",
     "inputs": {"reference": "required", "ref_text": "optional"},
     "params": [{"name": "temperature", "label": "Temperatur", "min": 0.1, "max": 1.5, "step": 0.05, "default": 0.8}]},
    {"variant": "b", "label": "Sprecher", "inputs": {"speaker": "required", "instruct": "optional"},
     "speakers": ["Vivian", "Ryan"], "instruct": {"kind": "text", "hint": "z. B. fröhlich"}},
]


class TestModes:
    def test_parsed_and_described(self, tmp_path):
        m = _manifest(tmp_path, variants=["a", "b"], modes=MODES, languages=["de", "en"])
        assert [x["variant"] for x in m.modes] == ["a", "b"]
        assert m.modes[1]["speakers"] == ["Vivian", "Ryan"] and m.languages == ("de", "en")
        assert m.param_names == {"temperature"}

    def test_engine_without_variants_has_one_mode_with_variant_null(self, tmp_path):
        m = _manifest(tmp_path, modes=[{"variant": None, "label": "Stil", "inputs": {"instruct": "optional"},
                                        "instruct": {"kind": "tags", "groups": [["male", "female"], ["child", "elderly"]]}}])
        assert m.modes[0]["variant"] is None and m.modes[0]["instruct"]["groups"][0] == ["male", "female"]

    def test_no_modes_is_fine_and_keeps_the_old_params(self, tmp_path):
        m = _manifest(tmp_path)
        assert m.modes == () and m.languages == ()
        assert m.param_names == {"temperature", "exaggeration", "cfg_weight"}

    @pytest.mark.parametrize("modes", [
        "x", [{"variant": "zzz", "label": "x"}],                       # variant must be one of the manifest's
        [{"variant": "a"}],                                             # label required
        [{"variant": "a", "label": "x", "inputs": {"exec": "required"}}],   # only known input kinds
        [{"variant": "a", "label": "x", "inputs": {"instruct": "maybe"}}],  # required | optional
        [{"variant": "a", "label": "x", "inputs": {"speaker": "required"}}],  # speaker needs a speakers list
        [{"variant": "a", "label": "x", "speakers": ["<script>"], "inputs": {"speaker": "required"}}],
        [{"variant": "a", "label": "x", "params": [{"name": "t", "min": 2, "max": 1}]}],
        [{"variant": "a", "label": "x", "params": [{"name": "t", "min": 0, "max": 1, "default": 5}]}],
        [{"variant": "a", "label": "x", "params": [{"name": "Bad Name", "min": 0, "max": 1}]}],
        [{"variant": "a", "label": "x", "instruct": {"kind": "tags", "groups": "male"}}],
        [{"variant": "a", "label": "x", "instruct": {"kind": "magic"}}],
        [{"variant": "a", "label": "x", "unknown": 1}],
        [{"variant": "a", "label": "x"}, {"variant": "a", "label": "y"}],   # one mode per variant
    ])
    def test_rejected(self, tmp_path, modes):
        with pytest.raises(ProviderError):
            _manifest(tmp_path, variants=["a", "b"], modes=modes)

    @pytest.mark.parametrize("languages", ["de", ["de; x"], [1]])
    def test_bad_languages(self, tmp_path, languages):
        with pytest.raises(ProviderError):
            _manifest(tmp_path, languages=languages)


class TestOpenParams:
    def test_a_future_param_name_passes_the_request_shape(self):
        # which params an engine takes is the manifest's business (checked per engine at generate)
        assert TtsRequest.from_dict({"text": "x", "params": {"top_p": 0.9}}).params == {"top_p": 0.9}

    @pytest.mark.parametrize("params", [{"Bad Name": 1}, {"x" * 40: 1}, {f"p{i}": 1 for i in range(17)}])
    def test_shape_still_checked(self, params):
        with pytest.raises(RequestError):
            TtsRequest.from_dict({"text": "x", "params": params})


def test_installed_means_its_program_is_there(tmp_path):
    """A provider folder alone is only an offer (a fresh ai-serve install sets up just Chatterbox)."""
    m = _manifest(tmp_path)
    assert m.installed is True
    (m.dir / "venv" / "bin" / "python").unlink()
    assert m.installed is False
