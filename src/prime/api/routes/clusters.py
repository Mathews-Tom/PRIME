"""Clusters endpoint.

GET /clusters - List all memory clusters
GET /clusters/{cluster_id} - Get cluster details
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from prime.api.dependencies import get_prime
from prime.api.models import ClusterInfoResponse, ClusterListResponse

if TYPE_CHECKING:
    from prime import PRIME

router = APIRouter(prefix="/clusters", tags=["clusters"])


@router.get("", response_model=ClusterListResponse)
def list_clusters(
    prime: PRIME = Depends(get_prime),
) -> ClusterListResponse:
    """List all memory clusters.

    Returns summary information about all clusters in the MCS.

    Args:
        prime: PRIME instance (injected).

    Returns:
        ClusterListResponse with list of cluster info.
    """
    stats = prime.mcs.get_stats()
    clusters_info = prime.mcs.list_clusters()

    return ClusterListResponse(
        clusters=[
            ClusterInfoResponse(
                cluster_id=info.cluster_id,
                size=info.size,
                is_consolidated=info.is_consolidated,
                prototype_norm=info.prototype_norm,
                creation_timestamp=info.creation_timestamp,
                last_access_timestamp=info.last_access_timestamp,
                access_count=info.access_count,
                representative_content=info.representative_content,
            )
            for info in clusters_info
        ],
        total_count=stats.get("cluster_count", 0),
    )


@router.get("/{cluster_id}", response_model=ClusterInfoResponse)
def get_cluster(
    cluster_id: int,
    prime: PRIME = Depends(get_prime),
) -> ClusterInfoResponse:
    """Get details for a specific cluster.

    Args:
        cluster_id: ID of the cluster to retrieve.
        prime: PRIME instance (injected).

    Returns:
        ClusterInfoResponse with cluster details.

    Raises:
        HTTPException: 404 if cluster not found.
    """
    info = prime.mcs.get_cluster_info(cluster_id)

    if info is None:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    return ClusterInfoResponse(
        cluster_id=info.cluster_id,
        size=info.size,
        is_consolidated=info.is_consolidated,
        prototype_norm=info.prototype_norm,
        creation_timestamp=info.creation_timestamp,
        last_access_timestamp=info.last_access_timestamp,
        access_count=info.access_count,
        representative_content=info.representative_content,
    )
