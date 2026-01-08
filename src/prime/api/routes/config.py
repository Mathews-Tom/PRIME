"""Configuration endpoint.

PUT /config - Update runtime configuration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends

from prime.api.dependencies import get_prime
from prime.api.models import ConfigUpdateRequest, ConfigUpdateResponse

if TYPE_CHECKING:
    from prime import PRIME

router = APIRouter(tags=["config"])


@router.put("/config", response_model=ConfigUpdateResponse)
def update_config(
    request: ConfigUpdateRequest,
    prime: PRIME = Depends(get_prime),  # noqa: ARG001
) -> ConfigUpdateResponse:
    """Update runtime configuration.

    Allows updating certain configuration parameters at runtime
    without restarting the service.

    Note: Not all configuration parameters can be updated at runtime.
    Only supported parameters will be applied.

    Args:
        request: ConfigUpdateRequest with parameters to update.
        prime: PRIME instance (injected, ensures PRIME is initialized).

    Returns:
        ConfigUpdateResponse indicating what was updated.
    """
    # prime is injected to ensure PRIME is initialized
    # Runtime config updates will be implemented in a future version
    changes: dict[str, Any] = {}

    # Update SSM variance threshold
    if request.variance_threshold is not None:
        # Note: This would require SSM to support runtime config updates
        # For now, we track the request but don't apply it
        changes["variance_threshold"] = {
            "requested": request.variance_threshold,
            "status": "not_supported",
        }

    # Update MCS similarity threshold
    if request.similarity_threshold is not None:
        changes["similarity_threshold"] = {
            "requested": request.similarity_threshold,
            "status": "not_supported",
        }

    # Update MCS consolidation threshold
    if request.consolidation_threshold is not None:
        changes["consolidation_threshold"] = {
            "requested": request.consolidation_threshold,
            "status": "not_supported",
        }

    return ConfigUpdateResponse(
        updated=len(changes) > 0,
        changes=changes,
    )
