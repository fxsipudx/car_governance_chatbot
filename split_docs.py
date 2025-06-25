
import logging
import pickle
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    # When used as part of the car_gov_bot package
    from .load_docs import load_documents
except ImportError:
    # Fallback when running standalone
    from load_docs import load_documents  # type: ignore

# Logger for this module
logger = logging.getLogger(__name__)

# Default splitter settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Default pickle path
CHUNKS_PATH: Path = Path(__file__).parent / "chunks.pkl"


def split_documents(
    pages: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    # Split each page into overlapping text chunks. Returns a list of Document chunks.
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(pages)
    logger.info("Split %d pages into %d chunks", len(pages), len(chunks))
    return chunks


def save_chunks(
    chunks: List[Document],
    path: Path = CHUNKS_PATH,
) -> None:
    # Pickle chunks list to the given path.

    path.write_bytes(pickle.dumps(chunks))
    logger.info("Saved %d chunks to %s", len(chunks), path)


def load_chunks(
    path: Path = CHUNKS_PATH,
) -> List[Document]:

    # Load chunks list from pickle at given path.

    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    chunks = pickle.loads(path.read_bytes())
    logger.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(
        description="Split PDFs into text chunks and cache in a pickle file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CHUNKS_PATH,
        help="Output pickle path (default: %(default)s)",
    )
    args = parser.parse_args()

    # Load pages, split into chunks, and save
    pages = load_documents()
    chunks = split_documents(pages)
    save_chunks(chunks, args.out)
    logger.info("✅ Saved chunks to %s", args.out)