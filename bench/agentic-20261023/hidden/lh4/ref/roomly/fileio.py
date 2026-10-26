# roomly - internal use only
"""Atomic file writes."""
import os
import tempfile


def safe_write(path: str, text: str) -> None:
    """Writes text to path through a temporary file in the same directory and os.replace."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
