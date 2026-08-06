"""
RAG utilities: TF-IDF index + query building + prompt formatting.
"""

from src.rag.index import build_rag_index, load_rag_index, retrieve, retrieve_multi
from src.rag.context import (
    build_rag_query,
    format_rag_sources_for_prompt,
    format_rag_references_section,
)

__all__ = [
    "build_rag_index",
    "load_rag_index",
    "retrieve",
    "retrieve_multi",
    "build_rag_query",
    "format_rag_sources_for_prompt",
    "format_rag_references_section",
]
