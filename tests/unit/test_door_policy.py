"""backend/door/policy.py -- who may use which app from outside, through the door (edge-door,
Pomerium). Security rules: an app that doesn't separate people's data (identity `single`)
can only be shared with the owner; page-less APIs (`ui: false`) never; names are plain labels,
unique, not the door's own; every share names someone. Rules are re-checked when the route
list is built, so a manifest that later becomes `single` narrows its share automatically."""
from pathlib import Path

import pytest

from backend.door.policy import DoorError, Share, check_share, identity_env, parse_shares, routes_doc
from backend.services.domain import ServiceManifest

D = "apps.example.org"


def manifest(id="pv-calc", identity="multi", ui=True, port=5180, **over):
    d = {"id": id, "name": "Sonnenrechner", "description": "Häuser durchrechnen", "command": ["./serve.sh"],
         "port": port, "bind": "127.0.0.1", "match": ["vite"], "identity": identity, "ui": ui}
    d.update(over)
    return ServiceManifest(id=d["id"], name=d["name"], description=d["description"], project_dir=Path("/x"),
                           command=tuple(d["command"]), env={}, port=d["port"], bind=d["bind"],
                           match=tuple(d["match"]), identity=d["identity"], ui=d["ui"])


def share(**over):
    s = {"app": "pv-calc", "name": "sonne", "groups": ("apps",), "users": ()}
    s.update(over)
    return Share(**s)


def test_multi_app_any_group_or_person():
    check_share(manifest(), share(), owner="alex", taken={})
    check_share(manifest(), share(groups=(), users=("anna", "bob")), owner="alex", taken={})


def test_single_app_only_for_the_owner():
    m = manifest(identity="single")
    check_share(m, share(groups=(), users=("alex",)), owner="alex", taken={})
    for bad in [share(), share(groups=(), users=("anna",)), share(groups=(), users=("alex", "anna")),
                share(groups=("apps",), users=("alex",))]:
        with pytest.raises(DoorError, match="nur für dich"):
            check_share(m, bad, owner="alex", taken={})
    with pytest.raises(DoorError):  # owner unknown -> a single app can't be shared at all
        check_share(m, share(groups=(), users=("owner",)), owner=None, taken={})


def test_no_api_without_page():
    with pytest.raises(DoorError):
        check_share(manifest(id="tts-serve", ui=False, identity="none"), share(app="tts-serve"), owner="alex", taken={})


def test_names_and_principals():
    for bad in ["", "Sonne", "login", "www", "a.b", "-x", "x" * 32]:
        with pytest.raises(DoorError):
            check_share(manifest(), share(name=bad), owner="alex", taken={})
    with pytest.raises(DoorError, match="schon"):
        check_share(manifest(), share(), owner="alex", taken={"sonne": "storytime"})
    check_share(manifest(), share(), owner="alex", taken={"sonne": "pv-calc"})  # its own name
    for bad in [share(groups=(), users=()), share(groups=("a b",)), share(users=("x;y",))]:
        with pytest.raises(DoorError):
            check_share(manifest(), bad, owner="alex", taken={})


def test_never_the_dashboard():
    with pytest.raises(DoorError):
        check_share(manifest(port=8030), share(), owner="alex", taken={})


def test_parse_shares():
    shares, errors = parse_shares({"pv-calc": {"name": "sonne", "groups": ["apps"]},
                                   "x": {"name": 3}, "y": "nope"})
    assert list(shares) == ["pv-calc"] and shares["pv-calc"].users == ()
    assert len(errors) == 2


def test_routes_doc_rechecks():
    ms = {"pv-calc": manifest(), "story": manifest(id="story", identity="single", port=8008)}
    shares = {"pv-calc": share(), "story": share(app="story", name="story", groups=("apps",)),
              "gone": share(app="gone", name="gone")}
    doc, errors = routes_doc(shares, ms, owner="alex")
    assert doc == {"routes": [{"name": "sonne", "upstream": "http://127.0.0.1:5180", "groups": ["apps"], "users": [],
                               "app": "pv-calc", "title": "Sonnenrechner", "description": "Häuser durchrechnen"}]}
    assert set(errors) == {"story", "gone"}


def test_identity_env():
    assert identity_env(share(), D) == {
        "LLAMA_SERVE_IDENTITY_AUDIENCE": "sonne.apps.example.org",
        "LLAMA_SERVE_IDENTITY_JWKS": "https://sonne.apps.example.org/.well-known/pomerium/jwks.json"}


def test_routes_carry_maturity():
    m = ServiceManifest(id="pv-calc", name="S", description="d", project_dir=Path("/x"), command=("./s",), env={},
                        port=5180, bind="127.0.0.1", match=("v",), identity="multi", maturity="beta")
    doc, _ = routes_doc({"pv-calc": share()}, {"pv-calc": m}, owner="alex")
    assert doc["routes"][0]["maturity"] == "beta"
