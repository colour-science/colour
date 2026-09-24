"""
Skip the backend interpolator modules whose array backend is not installed.

The ``torch`` and ``jax`` backends ``import`` their namespace at module scope,
which is safe at runtime (they are imported lazily, only for arrays of their
type) but errors during ``pytest --doctest-modules`` collection when the backend
is absent. Ignoring the corresponding file keeps the backends optional.
"""

from __future__ import annotations

from importlib.util import find_spec

__all__: list = []

collect_ignore: list = []

for _backend, _module in (("torch", "torch.py"), ("jax", "jax.py")):
    try:
        _available = find_spec(_backend) is not None
    except ImportError:
        _available = False

    if not _available:
        collect_ignore.append(_module)
