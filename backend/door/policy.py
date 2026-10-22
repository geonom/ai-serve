"""Who may use which app from outside, through the door (edge-door: Pomerium + edge server).

Vocabulary: a *share* opens one app at `https://<name>.<domain>/` for Nextcloud *groups*
and/or *people* (Nextcloud account ids = OIDC `sub`). The dashboard holds the shares
(door_shares.json) and sends the door the resulting *routes*; the door checks the login and
the policy, the app learns who it is from the signed identity (docs/identity.md).

Threat model / rules (tests/unit/test_door_policy.py):
- An app with identity `single` keeps one person's data and treats whoever comes in as that
  person -- sharing it with anyone but the owner would hand them the owner's data. So: only
  the owner, by account id, no groups (a group can gain members later). Unknown owner -> no.
- `ui: false` services are APIs for this machine, never published.
- The dashboard (8030) trusts loopback, and the door connects from loopback -> never published.
- Names are plain DNS labels, unique, not the door's own (`login`, `www`).
- Every share names at least one group or person -- there is no "everyone".
- Rules are re-applied whenever routes are built, so a manifest that later becomes `single`
  (or `ui: false`, or disappears) drops out of the door instead of staying open.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from backend.services.domain import ServiceManifest

_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]{0,29}[a-z0-9])?$")
_PRINCIPAL = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")
RESERVED = frozenset({"login", "www"})
FORBIDDEN_PORTS = frozenset({8030, 8444, 8081, 8046, 8047, int(__import__("os").getenv("AI_SERVE_PORT", "8030"))})


class DoorError(ValueError):
    pass


@dataclass(frozen=True)
class Share:
    app: str
    name: str
    groups: Tuple[str, ...] = ()
    users: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "groups": list(self.groups), "users": list(self.users)}


def _principals(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) and _PRINCIPAL.match(v) for v in value):
        raise DoorError("Gruppen und Personen: nur Buchstaben, Ziffern und _ . @ -")
    return tuple(dict.fromkeys(value))


def make_share(app: str, name: Any, groups: Any, users: Any) -> Share:
    if not isinstance(name, str) or not _NAME.match(name) or name in RESERVED:
        raise DoorError("Adresse: nur Kleinbuchstaben, Ziffern und Bindestrich (nicht login/www)")
    return Share(app=app, name=name, groups=_principals(groups), users=_principals(users))


def check_share(m: ServiceManifest, s: Share, *, owner: Optional[str], taken: Mapping[str, str]) -> None:
    """Raises DoorError if the share must not exist. `taken`: name -> app id of the other shares."""
    make_share(s.app, s.name, list(s.groups), list(s.users))  # shape (also for Share built elsewhere)
    if not m.ui:
        raise DoorError(f"{m.name} hat keine eigene Seite und wird nicht nach außen freigegeben.")
    if m.port in FORBIDDEN_PORTS:
        raise DoorError(f"Port {m.port} wird nie nach außen freigegeben.")
    if taken.get(s.name, s.app) != s.app:
        raise DoorError(f"Die Adresse {s.name} gehört schon zu {taken[s.name]}.")
    if not s.groups and not s.users:
        raise DoorError("Wähle mindestens eine Gruppe oder Person.")
    if m.identity == "single":
        if not owner or s.groups or s.users != (owner,):
            raise DoorError(f"{m.name} trennt Daten nicht pro Person – nur für dich freigebbar.")


def parse_shares(data: Any) -> Tuple[Dict[str, Share], List[str]]:
    shares: Dict[str, Share] = {}
    errors: List[str] = []
    if not isinstance(data, dict):
        return shares, ["door_shares.json: object expected"]
    for app, v in data.items():
        try:
            if not isinstance(v, dict):
                raise DoorError("object expected")
            shares[app] = make_share(app, v.get("name"), v.get("groups"), v.get("users"))
        except DoorError as e:
            errors.append(f"{app}: {e}")
    return shares, errors


def routes_doc(shares: Mapping[str, Share], manifests: Mapping[str, ServiceManifest], *,
               owner: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """The door's routes.json from the shares; shares that no longer pass are left out (with why)."""
    routes: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    taken = {s.name: s.app for s in shares.values()}
    for app, s in sorted(shares.items()):
        m = manifests.get(app)
        if m is None:
            errors[app] = "App nicht mehr registriert"
            continue
        try:
            check_share(m, s, owner=owner, taken=taken)
        except DoorError as e:
            errors[app] = str(e)
            continue
        route = {"name": s.name, "upstream": f"http://127.0.0.1:{m.port}", "groups": list(s.groups),
                 "users": list(s.users), "app": m.id, "title": m.name[:80], "description": m.description[:400]}
        if m.maturity:
            route["maturity"] = m.maturity   # shown on the overview page for the people it is shared with
        routes.append(route)
    return {"routes": routes}, errors


def identity_env(s: Share, domain: str) -> Dict[str, str]:
    """What the app needs to verify the door's identity for its public host (docs/identity.md)."""
    host = f"{s.name}.{domain}"
    return {"LLAMA_SERVE_IDENTITY_AUDIENCE": host,
            "LLAMA_SERVE_IDENTITY_JWKS": f"https://{host}/.well-known/pomerium/jwks.json"}
