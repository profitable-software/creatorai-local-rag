"""Qdrant vector store with Infinity (BAAI/bge-m3) embeddings.

One collection is shared by all tenants; isolation is done with a payload
filter on `metadata.tenant_id` that every read and delete must carry
(see retriever.py and ingest.py). A keyword payload index keeps those
filters cheap.
"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams

from app.config import get_settings

INDEXED_FIELDS = (
    "metadata.tenant_id",
    "metadata.data_type",
    "metadata.category",
    "metadata.document_id",
)


@lru_cache
def get_client() -> QdrantClient:
    s = get_settings()
    client = QdrantClient(url=s.qdrant_url)
    if not client.collection_exists(s.qdrant_collection):
        client.create_collection(
            collection_name=s.qdrant_collection,
            vectors_config=VectorParams(size=s.embedding_dim, distance=Distance.COSINE),
        )
        for field in INDEXED_FIELDS:
            client.create_payload_index(
                collection_name=s.qdrant_collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
    return client


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        base_url=s.infinity_url,
        api_key="not-needed",  # Infinity has no auth inside the private network
        # Send raw strings: LangChain would otherwise pre-tokenize with
        # tiktoken and send token ids, which Infinity does not accept.
        check_embedding_ctx_length=False,
        request_timeout=600,  # bge-m3 on CPU is slow for big batches
    )


def get_vector_store() -> QdrantVectorStore:
    s = get_settings()
    return QdrantVectorStore(
        client=get_client(),
        collection_name=s.qdrant_collection,
        embedding=get_embeddings(),
    )
