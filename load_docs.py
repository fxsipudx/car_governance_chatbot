
import logging
from pathlib import Path
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Directory containing PDF files
DOCS_DIR = Path(__file__).parent / "docs"


def load_documents() -> List[Document]:

    # Load every page from each PDF in DOCS_DIR and return as a list of Documents.
    # Raises: FileNotFoundError: If the docs directory does not exist.

    if not DOCS_DIR.is_dir():
        raise FileNotFoundError(f"Docs folder not found: {DOCS_DIR.resolve()}")

    docs: List[Document] = []
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))

    for pdf_path in pdf_paths:
        try:
            # Load pages from PDF; each page becomes a Document
            pages = PyPDFLoader(str(pdf_path)).load()
            docs.extend(pages)
            logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)
        except Exception as err:
            # Skip corrupted or unreadable PDFs
            logger.warning("Failed to load %s: %s", pdf_path.name, err)

    logger.info("Total pages loaded: %d", len(docs))
    return docs


if __name__ == "__main__":
    # Run as script for quick checks
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    documents = load_documents()
    print(f"{len(documents)} pages loaded from {len(sorted(DOCS_DIR.glob('*.pdf')))} PDFs")
