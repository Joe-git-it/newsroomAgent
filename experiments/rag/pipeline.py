from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
import shutil

DATA_DIR = Path(__file__).parent / "data"


def load_documents() -> list:
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        doc = TextLoader(str(path), encoding="utf-8").load()[0] 
        #LANGCHAIN CITATION KEY. COPIES METADATA FOR EVERY CHUNK PRODUCED                                                                            
        doc.metadata["source"] = path.name
        docs.append(doc)                                                                                                                
    return docs

def chunk_documents(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    splitter = RecursiveCharacterTextSplitter(
    #SMALLER CHUNKS, FINER-GRAINED RETRIEVAL, TIGHTER CONTEXT, CHEAPER PROMPTS AND LESS OFF-TOPIC NOISE
    chunk_size=chunk_size,
    #NBR OF REPEATED CHARS BETWEEN CHUNKS
    chunk_overlap=chunk_overlap,
    #PREFER PARAGRAPH BREAKS, THEN LINE BREAKS, THEN WORD BREAKS, THEN CHARACTER BREAKS
    separators=["\n\n", "\n", ". ", " ", ""],
    )
    #RETURNS DOCUMENT OBJECTS. KEEPS METADATA
    return splitter.split_documents(docs)

def get_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")

# EMBEDS EACH CHUNK AND SAVES THE VECTOR STORE TO DISK. 
def build_vectorstore(chunks: list, persist_dir: Path) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embedder(),
        persist_directory=str(persist_dir),
    )

# SEPERATE DIR BY CHUNK SIZE. EMBED EACH ONCE, THEN QUERY WITHOUT REBUILIDNG
def persist_dir_for(chunk_size: int) -> Path:
    return Path(__file__).parent / f"chroma_db_{chunk_size}"

# TO BE RAN ONCE PER CHUNK SIZE FROM CMD LINE. RERUN WHEN SOURCE DATA OR OVERLAP CHANGES
def ingest(chunk_size: int, chunk_overlap: int = 100):
    persist = persist_dir_for(chunk_size)
    if persist.exists():
        shutil.rmtree(persist)  # WIPE BEFORE REBUILD. AVOID ADDITIVE DUPLICATES
    docs = load_documents()
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"EMBEDDING {len(chunks)} CHUNKS INTO {persist.name}")
    build_vectorstore(chunks, persist)
    print("DONE")

# OPEN PERSISTED CHROMA DIR WITHOUT REBUILDING IT. ERROR IF NOT FOUND
def get_vectorstore(chunk_size: int) -> Chroma:
    persist = persist_dir_for(chunk_size)
    if not persist.exists():
        raise SystemExit(
            f"NO STORE AT {persist.name}. RUN: python pipeline.py ingest --chunk-size {chunk_size}"
        )
    return Chroma(persist_directory=str(persist), embedding_function=get_embedder())

# HELPER TO TURN CHUBKS INTO LARGE CONTEXT STRING SO LLM KNOWS WHERE EACH CHUNK CAME FROM
def format_context(chunks: list) -> str:
      parts = []
      for chunk in chunks:
          source = chunk.metadata.get("source", "unknown")
          parts.append(f"[{source}]\n{chunk.page_content}")
      return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    store = get_vectorstore(800)
    # WILL USE SIMILARITY SEARCH WITH K=3
    retriever = store.as_retriever(search_kwargs={"k": 3})
    chunks = retriever.invoke("Who won the 2026 Kerala election?")
    print(format_context(chunks))

