"""Autostart per service (backend/services/autostart.py rules; endpoint + startup run
in backend/routers/services.py). Added 2026-10-16 when the Nextcloud Funnel moved
into its own service (nextcloud-bridge): it must come back after a reboot on its
own, and the user decides per service in the dashboard -- never the manifest.

Rules: off unless switched on in the dashboard; the setting file is tolerant (a
broken entry is reported, the rest stays usable); at startup only enabled,
registered and not-already-running services are started -- through the same
validated start path as the Start button; ids of services that no longer exist
are ignored."""
from backend.services.autostart import AutostartSettings, plan_autostart


class TestSettings:
    def test_default_off(self):
        assert not AutostartSettings.from_dict(None).enabled("x")

    def test_roundtrip(self):
        s = AutostartSettings.from_dict({}).with_enabled("bridge", True)
        assert s.enabled("bridge") and AutostartSettings.from_dict(s.to_dict()).enabled("bridge")
        assert not s.with_enabled("bridge", False).enabled("bridge")

    def test_tolerant(self):
        s = AutostartSettings.from_dict({"services": {"ok": True, "bad": "yes", "off": False}})
        assert s.enabled("ok") and not s.enabled("bad") and not s.enabled("off")
        assert any("bad" in e for e in s.errors)

    def test_garbage(self):
        assert AutostartSettings.from_dict(["nope"]).errors


def test_plan():
    s = AutostartSettings.from_dict({"services": {"a": True, "b": True, "gone": True, "c": False}})
    assert plan_autostart(s, registered=["a", "b", "c"], running={"b"}) == ["a"]
