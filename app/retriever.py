"""Tenant-scoped retrieval.

Two passes, merged and de-duplicated:
1. semantic search over everything the tenant owns (document chunks + facts);
2. optional category pass over extracted facts ("Brand Brain" memory), so a
   prompt that always needs e.g. `target_audience` gets it even when the user's
   question does not mention it.
"""
from typing import Iterable

from langchain_core.documents import Document
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue

from app.config import get_settings
from app.vector_store import get_vector_store

FACT = "fact"  # metadata.data_type of extracted knowledge facts
CHUNK = "chunk"  # metadata.data_type of raw document chunks


def tenant_filter(
    tenant_id: str,
    data_type: str | None = None,
    categories: Iterable[str] | None = None,
) -> Filter:
    """Every query goes through here, so no search can forget the tenant."""
    if not tenant_id:
        raise ValueError("tenant_id is required")
    must = [FieldCondition(key="metadata.tenant_id", match=MatchValue(value=tenant_id))]
    if data_type:
        must.append(FieldCondition(key="metadata.data_type", match=MatchValue(value=data_type)))
    categories = list(categories or [])
    if categories:
        must.append(FieldCondition(key="metadata.category", match=MatchAny(any=categories)))
    return Filter(must=must)


def merge_unique(*batches: list[Document]) -> list[Document]:
    """Concatenate result lists, dropping repeated page_content, keeping order."""
    seen: set[str] = set()
    out: list[Document] = []
    for batch in batches:
        for doc in batch:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                out.append(doc)
    return out


def retrieve(
    tenant_id: str,
    query: str,
    required_categories: list[str] | None = None,
    k: int | None = None,
) -> list[Document]:
    k = k or get_settings().retrieval_k
    store = get_vector_store()
    semantic = store.similarity_search(query, k=k, filter=tenant_filter(tenant_id))
    by_category: list[Document] = []
    if required_categories:
        by_category = store.similarity_search(
            query,
            k=k,
            filter=tenant_filter(tenant_id, data_type=FACT, categories=required_categories),
        )
    return merge_unique(semantic, by_category)[:k]
