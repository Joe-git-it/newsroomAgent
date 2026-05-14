# RETRIEVAL MODULE WILL OPEN THE PERSISTED STORE, RETRIEVE TOP K CHUNKS, 
# AND INVOKE CLAUDE FOR ANSWER SOURCED FROM THOSE FORMATTED CHUNKS

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from newsroomagent.config import (
    CHAT_MODEL,
    DEFAULT_K,
    PERSIST_DIR,
    PROMPT_TEMPLATE,
)
from newsroomagent.ingest import get_embedder


# OPEN PERSISTED CHROMA DIR WITHOUT REBUILDING IT. ERROR IF NOT FOUND
def get_vectorstore() -> Chroma:
    if not PERSIST_DIR.exists():
        raise SystemExit(
            f"NO STORE AT {PERSIST_DIR.name}. RUN:  uv run python scripts/ingest.py"
        )
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=get_embedder())

def retrieve(question: str, k: int = DEFAULT_K) -> list[Document]:
    store = get_vectorstore()
    retriever = store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)

def format_context(chunks: list[Document]) -> str:
    parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        parts.append(f"[{source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)

def answer(question: str, k: int = DEFAULT_K) -> str:
    chunks = retrieve(question, k=k)
    prompt = PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)
    llm = ChatAnthropic(model=CHAT_MODEL, temperature=0)
    return llm.invoke(prompt).content