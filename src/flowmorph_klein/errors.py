"""Project-specific exception hierarchy.

The package raises typed exceptions at user-facing boundaries so notebook and
CLI callers can report actionable failures without treating a partially valid
object as a successful result.
"""

from __future__ import annotations


class FlowMorphKleinError(Exception):
    """Base class for expected FlowMorph Klein failures."""


class ConfigurationError(FlowMorphKleinError, ValueError):
    """A configuration is invalid or has not been fully resolved."""


class ManifestError(FlowMorphKleinError, ValueError):
    """An input manifest is malformed or references invalid inputs."""


class InputStagingError(FlowMorphKleinError, OSError):
    """Inputs could not be staged safely into the local working directory."""


class ImagePreprocessingError(FlowMorphKleinError, ValueError):
    """An endpoint image cannot be decoded or deterministically processed."""


class UnsupportedHardwareError(FlowMorphKleinError, RuntimeError):
    """The requested production profile is unsupported by the active runtime."""


class ChecksumMismatchError(InputStagingError):
    """A staged or copied file does not match its expected checksum."""
