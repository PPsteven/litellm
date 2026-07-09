"""Loader for the packaged LiteLLM Rust extension."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType

_BRIDGE_SENTINEL = object()
_cached_bridge: ModuleType | None | object = _BRIDGE_SENTINEL


def _probe_native_bridge() -> bool:
    """Return True only if the native extension can be imported without SIGILL.

    Runs a subprocess so that a SIGILL (Illegal Instruction, exit code 132)
    from a CPU that doesn't support the compiled instruction set crashes the
    child process instead of the main server process.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from litellm.rust_bridge import _native"],
            timeout=10,
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_native_bridge() -> ModuleType | None:
    """Return the packaged Rust extension, or ``None`` when unavailable."""
    global _cached_bridge
    if _cached_bridge is not _BRIDGE_SENTINEL:
        return _cached_bridge if isinstance(_cached_bridge, ModuleType) else None

    if not _probe_native_bridge():
        _cached_bridge = None
        return None

    try:
        from litellm.rust_bridge import _native
    except (ImportError, Exception):
        _cached_bridge = None
        return None
    _cached_bridge = _native
    return _native


def native_bridge_available() -> bool:
    """Whether the packaged Rust extension is importable."""
    return get_native_bridge() is not None
