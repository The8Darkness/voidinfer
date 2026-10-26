# roomly - internal use only
"""Error classes of roomly."""


class RoomlyError(Exception):
    """Base class of every roomly error."""
    slug = "roomly-error"


class ValidationError(RoomlyError):
    """Invalid input."""
    slug = "validation"


class UnknownRoomError(RoomlyError):
    """No such room."""
    slug = "unknown-room"


class UnknownBookingError(RoomlyError):
    """No such active booking."""
    slug = "unknown-booking"


class ConflictError(RoomlyError):
    """The slot collides with another booking."""
    slug = "conflict"


class CapacityError(RoomlyError):
    """Too many attendees for the room."""
    slug = "capacity"


class LoadError(RoomlyError):
    """The schedule file cannot be loaded."""
    slug = "load-failed"
