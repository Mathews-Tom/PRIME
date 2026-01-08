"""Exception hierarchy for Memory Cluster Store.

Defines specific exception types for MCS operations following
the fail-fast principle with explicit error handling.
"""

from __future__ import annotations


class MCSError(Exception):
    """Base exception for all MCS-related errors.

    All MCS exceptions inherit from this class, enabling
    catch-all handling when needed while preserving specificity.
    """


class ClusterNotFoundError(MCSError):
    """Raised when requested cluster does not exist.

    Causes include:
    - Invalid cluster_id provided to get_cluster
    - Cluster was deleted during consolidation
    - Race condition with concurrent operations
    """

    def __init__(self, cluster_id: int) -> None:
        """Initialize with cluster ID that was not found.

        Args:
            cluster_id: The cluster ID that could not be located.
        """
        self.cluster_id = cluster_id
        super().__init__(f"Cluster {cluster_id} not found")


class ConsolidationError(MCSError):
    """Raised when cluster consolidation fails.

    Causes include:
    - Cluster size below minimum threshold
    - Failed to compute centroid
    - Index rebuild failure after consolidation
    - Concurrent modification during consolidation
    """


class IndexError(MCSError):
    """Raised when vector index operations fail.

    Causes include:
    - Qdrant connection failure
    - Index not initialized
    - Search operation timeout
    - Incompatible vector dimensions
    """


class WriteError(MCSError):
    """Raised when memory write operation fails.

    Causes include:
    - Encoding failure for content
    - Index insertion failure
    - Cluster assignment failure
    - Max clusters limit reached
    """


class SearchError(MCSError):
    """Raised when memory search operation fails.

    Causes include:
    - Empty or invalid query embedding
    - Index search failure
    - Hybrid search fusion error
    - Timeout during search
    """


class ConfigurationError(MCSError):
    """Raised when MCS configuration is invalid.

    Causes include:
    - Invalid threshold values
    - Incompatible index configuration
    - Missing required dependencies
    """
