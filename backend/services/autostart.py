"""Autostart per service -- pure rules. Reasons and rules: tests/unit/test_services_autostart.py.
Chosen in the dashboard (stored in SERVICE_AUTOSTART_FILE), never by a manifest."""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple


@dataclass(frozen=True)
class AutostartSettings:
    flags: Mapping[str, bool] = field(default_factory=dict)
    errors: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Any) -> "AutostartSettings":
        if data is None:
            return cls()
        raw = data.get("services") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return cls(errors=("autostart settings: expected {\"services\": {id: true|false}}",))
        flags: Dict[str, bool] = {}
        errors: List[str] = []
        for sid, value in raw.items():
            if isinstance(value, bool):
                if value:
                    flags[str(sid)] = True
            else:
                errors.append(f"{sid}: autostart must be true or false, not {value!r}")
        return cls(flags=flags, errors=tuple(errors))

    def enabled(self, service_id: str) -> bool:
        return bool(self.flags.get(service_id))

    def with_enabled(self, service_id: str, on: bool) -> "AutostartSettings":
        flags = dict(self.flags)
        if on:
            flags[service_id] = True
        else:
            flags.pop(service_id, None)
        return AutostartSettings(flags=flags)

    def to_dict(self) -> Dict[str, Any]:
        return {"services": {k: True for k in sorted(self.flags)}}


def plan_autostart(settings: AutostartSettings, registered: Iterable[str], running: Set[str]) -> List[str]:
    return [sid for sid in registered if settings.enabled(sid) and sid not in running]
