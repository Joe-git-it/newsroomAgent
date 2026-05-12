from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma

DATA_DIR = Path(__file__).parent / "data"
PERSIST_DIR = Path(__file__).parent / "chroma_db"


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


if __name__ == "__main__":
    docs = load_documents()
    print(f"LOADED {len(docs)} DOCS")
    # print(f"FIRST DOC: {docs[0].metadata['source']}")
    # print(f"FIRST 200 CHARS:\n{docs[0].page_content[:200]}")

    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=100)
    print(f"SPLIT INTO {len(chunks)} CHUNKS")
    # print(f"FIRST CHUNK SOURCE: {chunks[0].metadata['source']}")
    # print(f"FIRST CHUNK ({len(chunks[0].page_content)} CHARS):")
    # print(chunks[0].page_content)

    embedder = get_embedder()
    v1 = embedder.embed_query("What city is hosting the 2026 Fifa World Cup?")
    print(f"VECTOR LENGTH: {len(v1)}")
    print(f"VECTOR PREVIEW (FIRST 5 DIMENSIONS): {v1[:5]}")
