from pathlib import Path
from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
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


if __name__ == "__main__":
    docs = load_documents()
    print(f"LOADED {len(docs)} DOCS")
    print(f"FIRST DOC: {docs[0].metadata['source']}")
    print(f"FIRST 200 CHARS:\n{docs[0].page_content[:200]}")