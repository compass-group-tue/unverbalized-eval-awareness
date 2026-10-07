# /fast on the MPI cluster has no file locking, which HuggingFace's cache needs.
# Swapping in filelock's SoftFileLock is the documented workaround; it must
# happen before huggingface_hub is imported.
from __future__ import annotations

import os


def install_if_requested() -> bool:
    """Replace filelock.FileLock with SoftFileLock when SOFTFILELOCK is set."""
    if not os.environ.get("SOFTFILELOCK"):
        return False
    try:
        import filelock
        from filelock import SoftFileLock
    except ImportError:
        return False
    filelock.FileLock = SoftFileLock
    return True
