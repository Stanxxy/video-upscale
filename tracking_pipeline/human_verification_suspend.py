"""Raised when human verification has been persisted and tracking must stop immediately."""

class HumanVerificationSuspend(Exception):
    """Service-mode detection_callback signals suspend; tracking exits and returns None."""
