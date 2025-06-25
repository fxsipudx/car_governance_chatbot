
import logging
from pathlib import Path
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Module logger
logger = logging.getLogger(__name__)

# Default directory for persistent Chroma database
DEFAULT_PERSIST_DIR: Path = Path(__file__).parent / "chroma_db"

# Embedding class reference
EMBEDDING_FN = OpenAIEmbeddings


def embed_and_store(
    chunks: List[Document],
    persist_dir: Path = DEFAULT_PERSIST_DIR,
) -> Chroma:

    # Embed document chunks and persist the Chroma vector store to `persist_dir`.
    # Returns: Chroma instance with persisted vectors.

    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = EMBEDDING_FN()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    store.persist()
    logger.info("Persisted %d vectors to %s", len(chunks), persist_dir)
    return store


def load_vectorstore(
    persist_dir: Path = DEFAULT_PERSIST_DIR,
) -> Chroma:
    
    # Load the persisted Chroma vector store from `persist_dir`.
    # Raises: FileNotFoundError: If `persist_dir` is missing.
    
    if not persist_dir.exists():
        raise FileNotFoundError(f"Chroma DB not found: {persist_dir}")
    embeddings = EMBEDDING_FN()
    store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    logger.info("Loaded Chroma store from %s", persist_dir)
    return store


if __name__ == "__main__":
    import argparse
    from split_docs import load_chunks

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Generate or load a Chroma embeddings store",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and overwrite existing store",
    )
    args = parser.parse_args()

    if args.force or not DEFAULT_PERSIST_DIR.exists():
        chunks = load_chunks()
        embed_and_store(chunks, persist_dir=DEFAULT_PERSIST_DIR)
    else:
        load_vectorstore(DEFAULT_PERSIST_DIR)
        logger.info("Vector store already exists. Use --force to overwrite.")