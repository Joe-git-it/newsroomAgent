# INGEST PIPELINE WILL LOAD TXT FILES, CHUNK, EMBED WITH OLLAMA, PERSIST TO CHROMA

import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from newsroomagent.config import (
      CHUNK_OVERLAP,
      CHUNK_SIZE,
      DATA_DIR,
      EMBED_MODEL,
      PERSIST_DIR,
  )

def load_documents() -> list:
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        doc = TextLoader(str(path), encoding="utf-8").load()[0] 
        #LANGCHAIN CITATION KEY. COPIES METADATA FOR EVERY CHUNK PRODUCED                                                                            
        doc.metadata["source"] = path.name
        docs.append(doc)                                                                                                                
    return docs

def chunk_documents(docs: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    #PREFER PARAGRAPH BREAKS, THEN LINE BREAKS, THEN WORD BREAKS, THEN CHARACTER BREAKS
    separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

def get_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL)

# EMBEDS EACH CHUNK AND SAVES THE VECTOR STORE TO DISK. 
def build_vectorstore(chunks: list, persist_dir: Path) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embedder(),
        persist_directory=str(persist_dir),
    )

# TO BE RAN ONCE PER CHUNK SIZE FROM CMD LINE. RERUN WHEN SOURCE DATA OR OVERLAP CHANGES
def ingest(chunk_size: int, chunk_overlap: int = 100):
    # persist = persist_dir_for(chunk_size)
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)  # WIPE BEFORE REBUILD. AVOID ADDITIVE DUPLICATES
    docs = load_documents()
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"EMBEDDING {len(chunks)} CHUNKS INTO {PERSIST_DIR.name}")
    build_vectorstore(chunks, PERSIST_DIR)
    print("DONE")





















