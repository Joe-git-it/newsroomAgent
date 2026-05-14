# Build the chroma vector store. 
# Loads, chunks, embeds with ollama, and persists to data/chroma. 
# Rerun whenever source articles or chunking config changes.
uv run python -c "from newsroomagent.ingest import ingest; ingest()"


# Smoke Test for retrieval with sample question
uv run python
chunks = retrieve("What election occured recently?", k=3)
for c in chunks: print(c.metadata["source"])