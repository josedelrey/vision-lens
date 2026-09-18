"""Exceptions raised by the supported Vision Lens API."""

__all__ = ["ConfigurationError", "PipelineError", "VisionLensError"]


class VisionLensError(Exception):
    """Base class for expected Vision Lens failures."""


class ConfigurationError(VisionLensError, ValueError):
    """Raised when a configuration cannot be loaded or validated."""


class PipelineError(VisionLensError, RuntimeError):
    """Raised when a configured pipeline cannot complete."""
