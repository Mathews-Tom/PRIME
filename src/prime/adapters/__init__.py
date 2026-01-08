"""Framework adapters for PRIME.

Provides integration with popular ML frameworks:
- LangChain: PRIMERetriever for LangChain chains and agents
- LlamaIndex: PRIMELlamaIndexRetriever for LlamaIndex query engines

Example (LangChain):
    >>> from prime import PRIME, PRIMEConfig
    >>> from prime.adapters import PRIMERetriever
    >>> prime = PRIME(PRIMEConfig.for_testing())
    >>> retriever = PRIMERetriever(prime=prime, mode="process_turn")

Example (LlamaIndex):
    >>> from prime import PRIME, PRIMEConfig
    >>> from prime.adapters import PRIMELlamaIndexRetriever
    >>> prime = PRIME(PRIMEConfig.for_testing())
    >>> retriever = PRIMELlamaIndexRetriever(prime=prime)
"""

from __future__ import annotations

from prime.adapters.langchain import PRIMERetriever
from prime.adapters.llamaindex import PRIMELlamaIndexRetriever

__all__ = [
    "PRIMELlamaIndexRetriever",
    "PRIMERetriever",
]
