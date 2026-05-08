"""rag-hybrid-search — production-grade hybrid retrieval + grounded generation."""

from importlib import metadata

try:
    __version__ = metadata.version("rag-hybrid-search")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
