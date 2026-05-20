from __future__ import annotations

import os
import sys
from pathlib import Path


def harden_scientific_runtime() -> None:
    """Avoid optional dependency import failures in mixed NumPy environments."""
    cache_root = Path(os.environ.get("MAOF_RUNTIME_CACHE_DIR", "/private/tmp/maof_runtime_cache"))
    for env_name, child in {
        "MPLCONFIGDIR": "matplotlib",
        "XDG_CACHE_HOME": "xdg",
    }.items():
        if not os.environ.get(env_name):
            path = cache_root / child
            path.mkdir(parents=True, exist_ok=True)
            os.environ[env_name] = str(path)

    for module_name in ("numexpr", "bottleneck", "xarray"):
        sys.modules.setdefault(module_name, None)

