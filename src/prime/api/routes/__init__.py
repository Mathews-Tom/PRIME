"""API route modules.

Exports all route routers for inclusion in the main app.
"""

from __future__ import annotations

from prime.api.routes.clusters import router as clusters_router
from prime.api.routes.config import router as config_router
from prime.api.routes.diagnostics import router as diagnostics_router
from prime.api.routes.memory import router as memory_router
from prime.api.routes.process import router as process_router

__all__ = [
    "clusters_router",
    "config_router",
    "diagnostics_router",
    "memory_router",
    "process_router",
]
