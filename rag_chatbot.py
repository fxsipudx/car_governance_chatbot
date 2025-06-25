"""
rag_chatbot.py
==============

CLI front-end for querying the automotive governance RAG system using Chroma.

Commands:
  ask <question>  Query the knowledge base and show answer with sources.

Examples:
  # Query without rebuilding index
  python rag_chatbot.py ask "Explain ASIL decomposition in ISO 26262"
  # Rebuild embeddings then query
  python embed_store.py --force
  python rag_chatbot.py ask "What are ASPICE maturity levels?"
"""
import logging
from pathlib import Path
from typing import List

import typer
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from embed_store import load_vectorstore  # renamed helper

# ---------------- Configuration ---------------- #
class Settings:
    """Configuration for RAG chain."""
    chroma_dir: Path = Path(__file__).parent / "chroma_db"
    llm_model: str = "gpt-4o"
    retrieval_k: int = 3
    score_threshold: float = 0.7

# Prompt guiding the LLM
AUTOMOTIVE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert automotive governance and safety standards consultant.\n"
        "Use the context below to answer each question accurately and concisely.\n"
        "If the context does not include the answer, say you do not know.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    ),
)

# ---------------- Logging ---------------- #
def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------------- RAG Chain ---------------- #
def make_chain(settings: Settings) -> RetrievalQA:
    """Build RetrievalQA chain with Chroma and OpenAI chat model."""
    # Load Chroma store (raises if missing)
    store = load_vectorstore(settings.chroma_dir)

    # Initialize LLM
    llm = ChatOpenAI(model=settings.llm_model, temperature=0.1, max_tokens=1000)

    # Configure retriever
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": settings.retrieval_k,
            "score_threshold": settings.score_threshold,
        },
    )

    # Build QA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": AUTOMOTIVE_PROMPT},
    )

# ---------------- CLI ---------------- #
app = typer.Typer(help="Ask automotive governance questions via RAG")

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the RAG system"),
    k: int = typer.Option(3, help="Number of chunks to retrieve"),
    verbose: bool = typer.Option(False, "-v", help="Enable debug logging"),
) -> None:
    """Query the knowledge base and display answer with source info."""
    configure_logging(verbose)
    load_dotenv()  # Load API keys from .env

    settings = Settings()
    settings.retrieval_k = k
    qa = make_chain(settings)

    result = qa.invoke({"query": question})

    # Print answer
    typer.secho("\n🧠 Answer:", fg=typer.colors.GREEN, bold=True)
    typer.echo(result["result"])

    # Print sources
    sources: List = result.get("source_documents", [])
    typer.secho(f"\n📎 Sources ({len(sources)}):", fg=typer.colors.BLUE, bold=True)
    for idx, doc in enumerate(sources, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        typer.echo(f"{idx}. {src} • p.{page}")

if __name__ == "__main__":
    app()