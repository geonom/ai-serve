"""Process listing that is safe to call from several threads at once.

psutil >= 6 keeps Process objects in a shared cache, and process_iter(attrs) stores the
requested fields on that shared object as `p.info`. Two threads iterating at the same
time with different attrs overwrite each other's `p.info` -- found live (2026-10-17):
the Services snapshot crashed with KeyError: 'create_time' while the GPU manager was
listing processes. `processes()` reads the fields per call into a fresh dict instead."""
from typing import Dict, Iterator, List, Tuple

import psutil


def processes(attrs: List[str]) -> Iterator[Tuple[psutil.Process, Dict]]:
    for p in psutil.process_iter():
        try:
            yield p, p.as_dict(attrs=attrs, ad_value=None)
        except psutil.Error:  # gone meanwhile
            continue
