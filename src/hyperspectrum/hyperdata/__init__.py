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
from .models import HydCommandResult

__all__ = [
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
