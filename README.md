# Build the chroma vector store. 
# Loads, chunks, embeds with ollama, and persists to data/chroma. 
# Rerun whenever source articles or chunking config changes.
uv run python -c "from newsroomagent.ingest import ingest; ingest()"


# Smoke Test for retrieval with sample question
uv run python
chunks = retrieve("What election occured recently?", k=3)
for c in chunks: print(c.metadata["source"])

# Generate Swagger UI
uv run uvicorn main:app --reload
# UI PATH
Click /research. Click "Try it out". Insert prompt like "What elections happened recently?"
# CMD PATH
curl -X POST http://127.0.0.1:8000/research -H "Content-Type: application/json" -d "{\"topic\":\"What recent elections happened?\",\"k\":3}"