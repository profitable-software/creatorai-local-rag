"""Ingestion: raw text -> chunks -> Qdrant, and facts -> Qdrant."""
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import FieldCondition, FilterSelector, MatchValue

from app.config import get_settings
from app.retriever import CHUNK, FACT, tenant_filter
from app.vector_store import get_client, get_vector_store


def split_text(text: str, tenant_id: str, document_id: str, source: str) -> list[Document]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    chunks = splitter.create_documents([text])
    for chunk in chunks:
        chunk.metadata.update(
            tenant_id=tenant_id,
            document_id=document_id,
            source=source,
            data_type=CHUNK,
        )
    return chunks


def fact_documents(
    facts: list[dict], tenant_id: str, document_id: str, source: str
) -> list[Document]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        Document(
            page_content=f["fact"],
            metadata={
                "tenant_id": tenant_id,
                "document_id": document_id,
                "source": source,
                "data_type": FACT,
                "category": f.get("category") or "other",
                "confidence": f.get("confidence"),
                "created_at": now,
            },
        )
        for f in facts
        if f.get("fact", "").strip()
    ]


def store(docs: list[Document]) -> int:
    if docs:
        get_vector_store().add_documents(docs)
    return len(docs)


def delete_document(tenant_id: str, document_id: str) -> None:
    """Remove a document's chunks and the facts extracted from it."""
    flt = tenant_filter(tenant_id)
    flt.must.append(FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id)))
    get_client().delete(
        collection_name=get_settings().qdrant_collection,
        points_selector=FilterSelector(filter=flt),
    )
