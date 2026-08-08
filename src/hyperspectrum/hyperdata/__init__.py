"""Safe, machine-readable access to the external HyperData CLI."""

from .gateway import (
    HydAuthenticationError,
    HydClientNotFoundError,
    HydCommandError,
    HydGateway,
    HydGatewayError,
    HydTransportError,
    HydUnsupportedClientError,
    HydUnsupportedJsonError,
)
from .models import DatasetCandidate, HydCommandResult

__all__ = [
    "DatasetCandidate",
    "HydAuthenticationError",
    "HydClientNotFoundError",
    "HydCommandError",
    "HydCommandResult",
    "HydGateway",
    "HydGatewayError",
    "HydTransportError",
    "HydUnsupportedClientError",
    "HydUnsupportedJsonError",
]
