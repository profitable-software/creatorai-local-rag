"""HTTP API for the local Brand Brain RAG demo."""
import asyncio
import logging
import uuid
from typing import Annotated

from fastapi import FastAPI, Path
from pydantic import BaseModel, Field

from app import ingest
from app.answer import answer
from app.extractor import extract_facts
from app.retriever import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="CreatorAI local RAG (Brand Brain)", version="1.0.0")

TenantId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class IngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500_000)
    source: str = Field(default="manual", max_length=200)
    extract_facts: bool = True
    language: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunks: int
    facts: list[dict]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    required_categories: list[str] = []
    language: str | None = None


class Source(BaseModel):
    text: str
    data_type: str | None
    category: str | None
    source: str | None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/tenants/{tenant_id}/documents", response_model=IngestResponse)
async def add_document(tenant_id: TenantId, req: IngestRequest) -> IngestResponse:
    document_id = str(uuid.uuid4())
    chunks = ingest.split_text(req.text, tenant_id, document_id, req.source)
    n_chunks = await asyncio.to_thread(ingest.store, chunks)
    facts: list[dict] = []
    if req.extract_facts:
        facts = await extract_facts(req.text, req.language)
        docs = ingest.fact_documents(facts, tenant_id, document_id, req.source)
        await asyncio.to_thread(ingest.store, docs)
    return IngestResponse(document_id=document_id, chunks=n_chunks, facts=facts)


@app.post("/tenants/{tenant_id}/ask", response_model=AskResponse)
async def ask(tenant_id: TenantId, req: AskRequest) -> AskResponse:
    docs = await asyncio.to_thread(retrieve, tenant_id, req.question, req.required_categories)
    text = await answer(req.question, docs, req.language)
    return AskResponse(
        answer=text,
        sources=[
            Source(
                text=d.page_content,
                data_type=d.metadata.get("data_type"),
                category=d.metadata.get("category"),
                source=d.metadata.get("source"),
            )
            for d in docs
        ],
    )


@app.delete("/tenants/{tenant_id}/documents/{document_id}", status_code=204)
async def delete_document(tenant_id: TenantId, document_id: uuid.UUID) -> None:
    await asyncio.to_thread(ingest.delete_document, tenant_id, str(document_id))
