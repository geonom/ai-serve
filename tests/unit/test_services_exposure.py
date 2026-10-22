"""Exposure of services via Tailscale (backend/services/exposure.py) -- pure rules.

Threat model: a service exposed beyond this machine must reach exactly the chosen
audience -- never the internet (Funnel), and with level "owner" only the Tailscale
user who owns this machine (other tailnet users exist: a shared-in device). Changing
the level is itself privileged: someone on the LAN must not be able to put a
local-only app into the tailnet."""
import stat

import pytest

from backend.services.domain import ServiceManifest
from backend.services.exposure import (
    FUNNEL_PORTS,
    Exposure,
    ExposureError,
    ExposureSettings,
    gate_allows,
    may_change_exposure,
    foreign_routes,
    parse_serve_status,
    plan_exposure,
    reconcile_serve,
)

OWNER = "owner@github"
HOST = "my-pc.example-tailnet.ts.net"


def _manifest(tmp_path, **over):
    d = tmp_path / "proj"
    d.mkdir(exist_ok=True)
    s = d / "run.sh"
    s.write_text("#!/bin/sh\n")
    s.chmod(s.stat().st_mode | stat.S_IXUSR)
    data = {"id": "storytime", "name": "StoryTime", "command": ["./run.sh"], "port": 8008, "match": ["main.py"]}
    data.update(over)
    return ServiceManifest.from_dict(data, d)


# --------------------------------------------------------------------------- planning

class TestPlanExposure:
    def test_local_needs_nothing(self, tmp_path):
        assert plan_exposure(_manifest(tmp_path), Exposure.LOCAL, OWNER).route is None

    def test_tailnet_proxies_straight_to_the_service(self, tmp_path):
        plan = plan_exposure(_manifest(tmp_path, identity="multi"), Exposure.TAILNET, OWNER)
        assert plan.route.https_port == 18008 and plan.route.target == "http://127.0.0.1:8008"
        assert plan.gate is None and plan.warning is None

    def test_owner_goes_through_the_gate(self, tmp_path):
        plan = plan_exposure(_manifest(tmp_path), Exposure.OWNER, OWNER)
        assert plan.route.https_port == 18008
        assert plan.gate.listen_port == 38008 and plan.gate.upstream_port == 8008 and plan.gate.owner == OWNER
        assert plan.route.target == "http://127.0.0.1:38008"

    def test_custom_tailnet_port(self, tmp_path):
        plan = plan_exposure(_manifest(tmp_path, tailnet_port=9008, identity="multi"), Exposure.TAILNET, OWNER)
        assert plan.route.https_port == 9008

    @pytest.mark.parametrize("tport", sorted(FUNNEL_PORTS))
    def test_funnel_capable_ports_never_used(self, tmp_path, tport):
        with pytest.raises(ExposureError):
            plan_exposure(_manifest(tmp_path, tailnet_port=tport), Exposure.TAILNET, OWNER)

    def test_tailnet_port_out_of_range(self, tmp_path):
        with pytest.raises(ExposureError):
            plan_exposure(_manifest(tmp_path, port=60000), Exposure.TAILNET, OWNER)

    def test_owner_requires_known_owner(self, tmp_path):
        with pytest.raises(ExposureError):
            plan_exposure(_manifest(tmp_path), Exposure.OWNER, None)

    def test_url(self, tmp_path):
        plan = plan_exposure(_manifest(tmp_path, open_path="/app", identity="multi"), Exposure.TAILNET, OWNER)
        assert plan.url(HOST) == f"https://{HOST}:18008/app"
        assert plan_exposure(_manifest(tmp_path), Exposure.LOCAL, OWNER).url(HOST) is None


# --------------------------------------------------------------------------- settings

class TestSettings:
    def test_roundtrip(self):
        s = ExposureSettings.from_dict({"services": {"storytime": "owner", "munin": "tailnet"}})
        assert s.level("storytime") == Exposure.OWNER and s.level("unknown") == Exposure.LOCAL
        s2 = s.with_level("storytime", Exposure.LOCAL)
        assert s2.to_dict() == {"services": {"munin": "tailnet"}}
        assert s.level("storytime") == Exposure.OWNER  # immutable

    def test_invalid_entries_dropped_and_reported(self):
        s = ExposureSettings.from_dict({"services": {"a": "public", "b": "funnel", "c": 5, "d": "tailnet"}})
        assert s.to_dict() == {"services": {"d": "tailnet"}} and len(s.errors) == 3

    @pytest.mark.parametrize("data", [None, {}, {"services": None}])
    def test_empty(self, data):
        assert ExposureSettings.from_dict(data).to_dict() == {"services": {}}

    def test_public_is_not_a_level(self):
        assert {e.value for e in Exposure} == {"local", "tailnet", "owner"}


# --------------------------------------------------------------------------- serve config

SERVE_STATUS = {
    "TCP": {"443": {"HTTPS": True}, "8443": {"HTTPS": True}, "18008": {"HTTPS": True}, "19999": {"HTTPS": True}},
    "Web": {
        f"{HOST}:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8000"}}},
        f"{HOST}:8443": {"Handlers": {"/public": {"Proxy": "http://127.0.0.1:8030/public"}}},
        f"{HOST}:18008": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8008"}}},
        f"{HOST}:19999": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:1234"}}},
    },
    "AllowFunnel": {f"{HOST}:8443": True},
}


class TestServeConfig:
    def test_parse(self):
        routes = parse_serve_status(SERVE_STATUS)
        assert routes[18008] == "http://127.0.0.1:8008" and routes[443] == "http://127.0.0.1:8000"
        assert 8443 not in routes  # not a root mount -- not ours, never touched

    def test_parse_empty(self):
        assert parse_serve_status({}) == {} and parse_serve_status(None) == {}

    def test_reconcile_adds_changes_and_removes_only_managed_ports(self):
        actual = parse_serve_status(SERVE_STATUS)
        desired = {18040: "http://127.0.0.1:8040", 18008: "http://127.0.0.1:38008"}
        managed = {18008, 18040, 15000, 19999}
        actions = reconcile_serve(desired, actual, managed)
        assert sorted(actions) == sorted([
            ("set", 18040, "http://127.0.0.1:8040"),
            ("set", 18008, "http://127.0.0.1:38008"),
            ("off", 19999, None),
        ])

    def test_reconcile_noop(self):
        actual = parse_serve_status(SERVE_STATUS)
        assert reconcile_serve({18008: "http://127.0.0.1:8008"}, actual, {18008}) == []

    def test_reconcile_never_touches_funnel_ports_or_unmanaged(self):
        actual = parse_serve_status(SERVE_STATUS)
        actions = reconcile_serve({}, actual, {443, 8443, 10000})
        assert actions == []

    def test_reconcile_removes_hand_made_routes_to_service_ports(self):
        # The dashboard is the only authority over a service's reachability: a root
        # route to a service's port set up by hand (Munin: 443 -> :8000) goes away,
        # even on a Funnel port. Path mounts (/public on 8443) and routes to ports
        # that belong to no service (19999 -> :1234) are not touched.
        actual = parse_serve_status(SERVE_STATUS)
        actions = reconcile_serve({18008: "http://127.0.0.1:8008"}, actual, {18008}, service_ports={8000, 8008})
        assert actions == [("off", 443, None)]

    def test_reconcile_keeps_the_dashboards_own_route_to_a_service(self):
        actual = {18008: "http://127.0.0.1:8008"}
        assert reconcile_serve({18008: "http://127.0.0.1:8008"}, actual, {18008}, service_ports={8008}) == []


# --------------------------------------------------------------------------- gate / who may change

class TestGate:
    def test_owner_from_tailscale_allowed(self):
        assert gate_allows("127.0.0.1", {"tailscale-user-login": OWNER}, OWNER)[0]
        assert gate_allows("::1", {"Tailscale-User-Login": OWNER.upper()}, OWNER)[0]

    @pytest.mark.parametrize("peer,headers", [
        ("127.0.0.1", {"tailscale-user-login": "someone@else"}),
        ("127.0.0.1", {}),  # local process, tagged node, or Funnel -- no identity
        ("127.0.0.1", {"tailscale-user-login": OWNER, "tailscale-funnel-request": "?1"}),
        ("192.168.1.20", {"tailscale-user-login": OWNER}),  # forged header from the LAN
        ("100.64.0.5", {"tailscale-user-login": OWNER}),  # direct, bypassing serve
    ])
    def test_denied(self, peer, headers):
        ok, reason = gate_allows(peer, headers, OWNER)
        assert not ok and reason

    def test_unknown_owner_denies_everything(self):
        assert not gate_allows("127.0.0.1", {"tailscale-user-login": OWNER}, None)[0]


class TestMayChangeExposure:
    def test_local_machine(self):
        assert may_change_exposure("127.0.0.1", None, OWNER)
        assert may_change_exposure("::1", None, OWNER)

    def test_owners_tailnet_device(self):
        assert may_change_exposure("100.101.1.2", OWNER, OWNER)

    @pytest.mark.parametrize("peer,login", [
        ("100.101.1.2", "someone@else"), ("100.101.1.2", None), ("192.168.1.20", None),
        ("192.168.1.20", OWNER), ("10.0.0.3", OWNER),
    ])
    def test_others_not(self, peer, login):
        assert not may_change_exposure(peer, login, OWNER)

    def test_no_owner_known_only_local(self):
        assert may_change_exposure("127.0.0.1", None, None)
        assert not may_change_exposure("100.101.1.2", OWNER, None)


class TestForeignRoutes:
    """A route set up outside the dashboard (e.g. `tailscale serve` by hand) already
    exposes a service -- the card must say so instead of showing 'local'."""

    def test_route_to_service_port_on_unmanaged_port_reported(self, tmp_path):
        m = _manifest(tmp_path, port=8000, id="munin", match=["munin.app"])
        actual = {443: "http://127.0.0.1:8000", 18000: "http://127.0.0.1:8000", 19999: "http://127.0.0.1:1"}
        assert foreign_routes(m, actual) == [443]

    def test_localhost_spelling_and_trailing_path(self, tmp_path):
        m = _manifest(tmp_path, port=8000)
        assert foreign_routes(m, {444: "http://localhost:8000/", 445: "http://127.0.0.1:8000/sub"}) == [444, 445]

    def test_other_ports_ignored(self, tmp_path):
        m = _manifest(tmp_path, port=8000)
        assert foreign_routes(m, {443: "http://127.0.0.1:80001", 446: "http://127.0.0.1:18000"}) == []


class TestNoPage:
    """`"ui": false` (APIs such as tts-serve): nothing to open, so nothing to expose --
    an API must not land in the tailnet by a click meant for an app page."""

    def test_api_service_stays_local(self, tmp_path):
        m = _manifest(tmp_path, ui=False)
        assert plan_exposure(m, Exposure.LOCAL, "owner@github").route is None
        for level in (Exposure.TAILNET, Exposure.OWNER):
            with pytest.raises(ExposureError, match="no page"):
                plan_exposure(m, level, "owner@github")

    def test_default_is_a_page(self, tmp_path):
        assert _manifest(tmp_path).ui is True
        assert plan_exposure(_manifest(tmp_path), Exposure.OWNER, "owner@github").route is not None


class TestIdentityGuard:
    """A `single` app keeps one person's data: only the owner may reach it. „Ganzes Tailnet"
    would let other users' shared-in devices in, so it is refused (docs/identity.md 4)."""

    def test_single_app_not_for_the_whole_tailnet(self, tmp_path):
        m = _manifest(tmp_path)   # no identity field -> single
        with pytest.raises(ExposureError, match="Nur ich"):
            plan_exposure(m, Exposure.TAILNET, "owner@github")
        assert plan_exposure(m, Exposure.OWNER, "owner@github").route is not None

    @pytest.mark.parametrize("identity", ["none", "multi"])
    def test_shareable_apps_may_be_on_the_tailnet(self, tmp_path, identity):
        assert plan_exposure(_manifest(tmp_path, identity=identity), Exposure.TAILNET, "owner@github").route is not None
