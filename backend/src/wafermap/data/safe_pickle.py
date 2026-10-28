"""Restricted unpickler for the WM-811K ``LSWMD.pkl`` file.

``pickle.load`` on a file downloaded from the internet can execute arbitrary code, because a
pickle may reference any importable callable. ``LSWMD.pkl`` was written by a 2015-era pandas,
so we cannot avoid pickle entirely; instead we only allow the handful of numpy / pandas
reconstruction helpers a DataFrame pickle legitimately needs, and remap the legacy module paths
(``pandas.indexes.*`` -> ``pandas.core.indexes.*``) that modern pandas renamed.

Anything outside the allow-list raises ``pickle.UnpicklingError`` before it is imported.
"""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path
from typing import Any, BinaryIO

# Legacy module paths used by old pandas pickles -> current location.
_MODULE_REMAP: dict[str, str] = {
    "__builtin__": "builtins",  # file was written by Python 2
    "pandas.indexes.base": "pandas.core.indexes.base",
    "pandas.indexes.range": "pandas.core.indexes.range",
    "pandas.indexes.numeric": "pandas.core.indexes.base",
    "pandas.core.indexes.numeric": "pandas.core.indexes.base",
    "pandas.core.internals": "pandas.core.internals.managers",
    "numpy.core.multiarray": "numpy._core.multiarray",
    "numpy.core.numeric": "numpy._core.numeric",
}

# Classes old numeric index types map to after pandas 2.0 removed them.
_NAME_REMAP: dict[tuple[str, str], tuple[str, str]] = {
    ("pandas.core.indexes.base", "Int64Index"): ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "Float64Index"): ("pandas.core.indexes.base", "Index"),
}

# (module, name) pairs that may be resolved. Everything else is rejected.
ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("builtins", "slice"),
    }
)


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that resolves only allow-listed globals."""

    def __init__(self, file: BinaryIO, *, encoding: str = "latin1") -> None:
        super().__init__(file, encoding=encoding)
        self.requested: set[tuple[str, str]] = set()

    def find_class(self, module: str, name: str) -> Any:
        module = _MODULE_REMAP.get(module, module)
        module, name = _NAME_REMAP.get((module, name), (module, name))
        self.requested.add((module, name))
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"global '{module}.{name}' is not allow-listed")
        return getattr(importlib.import_module(module), name)


def load_restricted(path: Path) -> Any:
    """Load a pickle using :class:`RestrictedUnpickler`."""
    with path.open("rb") as fh:
        return RestrictedUnpickler(fh).load()
