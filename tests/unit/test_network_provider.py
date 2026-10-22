"""Network providers (backend/network/): the private overlay network the dashboard,
the services and the cluster rely on, behind one small interface -- so Tailscale
is one option, not a hard dependency (user decision 2026-10-16: keep Tailscale for
now, but make open-source alternatives possible, even if the Funnel then goes).

Interface: is_private(ip) / owner() / identify(ip) / capabilities.
- TailscaleProvider: 100.64.0.0/10 + fd7a:115c:a1e0::/48, owner from `tailscale
  status`, identity from `tailscale whois`; capabilities serve + funnel.
- StaticProvider: for WireGuard, Nebula, NetBird, ZeroTier, Headscale without
  whois -- private networks and an address -> identity map from a config file.
  On WireGuard an overlay address is bound to one device key (cryptokey routing),
  which is what makes the map trustworthy. No serve/funnel.
- NoneProvider: no private network -- only loopback and the API key."""
import json

import pytest

from backend.network import NoneProvider, StaticProvider, TailscaleProvider, load_provider, ProviderError


class TestTailscale:
    def test_ranges(self):
        p = TailscaleProvider(info=lambda: None, whois=lambda ip: None)
        assert p.is_private("100.64.7.11") and p.is_private("fd7a:115c:a1e0::1")
        assert not p.is_private("192.168.1.40") and not p.is_private("bogus")

    def test_owner_and_identity(self):
        class Info:
            owner = "owner@github"
        p = TailscaleProvider(info=lambda: Info(), whois=lambda ip: "owner@github")
        assert p.owner() == "owner@github" and p.identify("100.64.7.11") == "owner@github"
        assert {"serve", "funnel"} <= p.capabilities

    def test_identify_only_private(self):
        calls = []
        p = TailscaleProvider(info=lambda: None, whois=lambda ip: calls.append(ip) or "x")
        assert p.identify("192.168.1.40") is None and calls == []


class TestStatic:
    def make(self, **over):
        cfg = {"networks": ["10.8.0.0/24"], "owner": "alex", "peers": {"10.8.0.2": "alex", "10.8.0.3": "guest"}}
        cfg.update(over)
        return StaticProvider.from_dict(cfg)

    def test_basics(self):
        p = self.make()
        assert p.is_private("10.8.0.2") and not p.is_private("10.9.0.1")
        assert p.owner() == "alex" and p.identify("10.8.0.2") == "alex" and p.identify("10.8.0.3") == "guest"
        assert p.identify("10.8.0.9") is None and p.capabilities == frozenset()

    def test_peer_outside_networks_refused(self):
        with pytest.raises(ProviderError):
            self.make(peers={"192.168.1.40": "alex"})

    @pytest.mark.parametrize("nets", [["0.0.0.0/0"], ["::/0"], ["8.8.8.0/24"], [], ["nonsense"]])
    def test_networks_must_be_private_ranges(self, nets):
        # a "private" network that covers the internet would trust everybody
        with pytest.raises(ProviderError):
            self.make(networks=nets, peers={})

    def test_owner_required(self):
        with pytest.raises(ProviderError):
            self.make(owner="")

    def test_unknown_fields(self):
        with pytest.raises(ProviderError):
            self.make(trust_everyone=True)


class TestNone:
    def test_nothing_private(self):
        p = NoneProvider()
        assert not p.is_private("100.64.7.11") and p.owner() is None and p.identify("10.0.0.1") is None


class TestLoad:
    def test_default_tailscale(self, tmp_path):
        assert load_provider("", str(tmp_path / "missing.json")).name == "tailscale"

    def test_static_from_file(self, tmp_path):
        f = tmp_path / "network.json"
        f.write_text(json.dumps({"networks": ["10.8.0.0/24"], "owner": "alex", "peers": {}}))
        assert load_provider("static", str(f)).name == "static"

    def test_static_without_file_fails_closed(self, tmp_path):
        with pytest.raises(ProviderError):
            load_provider("static", str(tmp_path / "missing.json"))

    def test_none_and_unknown(self, tmp_path):
        assert load_provider("none", "").name == "none"
        with pytest.raises(ProviderError):
            load_provider("zerotrust-magic", "")
