# roomly - internal use only
"""Duration and start-time helpers."""
import re
from datetime import datetime, timezone

from roomly.errors import ValidationError

_DUR = re.compile(r"(?:(\d+)h(?:(\d{1,2})m)?|(\d+)m?)")


def parse_duration(text: str) -> int:
    """Parses '45', '45m', '2h', '1h30m' into minutes (> 0)."""
    m = _DUR.fullmatch(text.strip())
    if not m:
        raise ValidationError(f"bad duration {text!r}")
    if m.group(1) is not None:
        mins = int(m.group(2) or 0)
        if mins > 59:
            raise ValidationError(f"bad duration {text!r}")
        total = int(m.group(1)) * 60 + mins
    else:
        total = int(m.group(3))
    if total <= 0:
        raise ValidationError(f"bad duration {text!r}")
    return total


def format_duration(minutes: int) -> str:
    """Formats minutes as '45m', '1h', '1h30m', '2h05m'."""
    if minutes < 0:
        raise ValidationError("negative duration")
    h, m = divmod(minutes, 60)
    if h == 0:
        return f"{m}m"
    return f"{h}h" if m == 0 else f"{h}h{m:02d}m"


def parse_start(text: str) -> datetime:
    """Parses 'YYYY-MM-DDTHH:MM' into an aware UTC datetime."""
    s = text.strip()
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d", s):
        raise ValidationError(f"bad start {text!r}")
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValidationError(f"bad start {text!r}") from None


def format_start(dt: datetime) -> str:
    """Formats a datetime as 'YYYY-MM-DDTHH:MM' (UTC)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M")
