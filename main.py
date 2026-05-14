# NEWSROOMAGENT API. RETURNS CHUNKS

from fastapi import FastAPI
from pydantic import BaseModel
from newsroomagent.retrieval import retrieve


class ResearchRequest(BaseModel):
    topic: str
    k: int = 5

class Chunk(BaseModel):
    source: str
    content: str

class ResearchResponse(BaseModel):
    chunks: list[Chunk]

app = FastAPI(title="NewsroomAgent", version="0.1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

# PRIMARY ENDPOINT. TAKES A TOPIC AND RETURNS TOP K MATCHING CHUNKS.
@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest) -> ResearchResponse:
    docs = retrieve(req.topic, k=req.k)
    chunks = [
        Chunk(source=d.metadata.get("source", "unknown"), content=d.page_content)
            for d in docs
    ]
    return ResearchResponse(chunks=chunks)


