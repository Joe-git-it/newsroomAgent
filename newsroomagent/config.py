from pathlib import Path


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"
PERSIST_DIR = ROOT / "data" / "chroma"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "claude-sonnet-4-6"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
DEFAULT_K = 5

PROMPT_TEMPLATE = """You are answering questions about recent world events using the provided context.                            
  Use ONLY the context below. If the answer is not present, say so plainly.                                                           Cite the sources (filenames) you used at the end of your answer.

  Context:
  {context}

  Question: {question}

  Answer:"""