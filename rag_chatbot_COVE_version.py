import logging
from pathlib import Path
from typing import List, Dict, Any

import typer
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from embed_store import load_vectorstore  

# Configuration
class Settings:
    # Configuration for RAG chain.
    chroma_dir: Path = Path(__file__).parent / "chroma_db"
    llm_model: str = "gpt-4o"
    retrieval_k: int = 3
    score_threshold: float = 0.7
    enable_verification: bool = True  # New flag to enable/disable CoVe

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

# Verification prompt
VERIFICATION_PROMPT = PromptTemplate(
    input_variables=["original_answer", "question"],
    template=(
        "Given this answer to an automotive governance question, generate 2-3 specific verification questions "
        "that would help validate the accuracy of the answer. Focus on key facts, standards, or processes mentioned.\n\n"
        "Original Question: {question}\n"
        "Original Answer: {original_answer}\n\n"
        "Verification Questions (one per line):"
    ),
)

# Revision prompt
REVISION_PROMPT = PromptTemplate(
    input_variables=["original_answer", "question", "verification_context"],
    template=(
        "You are reviewing an automotive governance answer for accuracy. "
        "Use the verification context to refine or confirm the original answer.\n\n"
        "Original Question: {question}\n"
        "Original Answer: {original_answer}\n"
        "Verification Context: {verification_context}\n\n"
        "Provide the final, verified answer:"
    ),
)

# Logging
def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger(__name__)

# Chain-of-Verification wrapper
class ChainOfVerificationQA:
    def __init__(self, base_chain: RetrievalQA, llm: ChatOpenAI, retriever, settings: Settings):
        self.base_chain = base_chain
        self.llm = llm
        self.retriever = retriever
        self.settings = settings
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Step 1: Get initial answer
        initial_result = self.base_chain.invoke(inputs)
        
        if not self.settings.enable_verification:
            return initial_result
        
        logger.debug("Starting Chain-of-Verification process")
        
        # Step 2: Generate verification questions
        verification_questions = self._generate_verification_questions(
            initial_result["result"], 
            inputs["query"]
        )
        
        if not verification_questions:
            logger.debug("No verification questions generated, returning original answer")
            return initial_result
        
        # Step 3: Retrieve context for verification questions
        verification_context = self._retrieve_verification_context(verification_questions)
        
        # Step 4: Revise answer based on verification
        if verification_context:
            revised_answer = self._revise_answer(
                initial_result["result"],
                inputs["query"],
                verification_context
            )
            
            # Return revised result
            result = initial_result.copy()
            result["result"] = revised_answer
            result["verification_questions"] = verification_questions
            return result
        
        return initial_result
    
    def _generate_verification_questions(self, answer: str, question: str) -> List[str]:
        try:
            prompt = VERIFICATION_PROMPT.format(original_answer=answer, question=question)
            response = self.llm.invoke(prompt)
            
            # Parse verification questions 
            questions = [q.strip() for q in response.content.split('\n') if q.strip() and not q.strip().startswith('Verification')]
            logger.debug(f"Generated {len(questions)} verification questions")
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.warning(f"Failed to generate verification questions: {e}")
            return []
    
    def _retrieve_verification_context(self, questions: List[str]) -> str:
        try:
            all_context = []
            for question in questions:
                # Retrieve relevant documents for each verification question
                docs = self.retriever.get_relevant_documents(question)
                if docs:
                    context = "\n".join([doc.page_content for doc in docs[:2]])  # Top 2 docs per question
                    all_context.append(f"Q: {question}\nContext: {context}")
            
            verification_context = "\n\n".join(all_context)
            logger.debug(f"Retrieved verification context: {len(verification_context)} characters")
            return verification_context
            
        except Exception as e:
            logger.warning(f"Failed to retrieve verification context: {e}")
            return ""
    
    def _revise_answer(self, original_answer: str, question: str, verification_context: str) -> str:
        try:
            prompt = REVISION_PROMPT.format(
                original_answer=original_answer,
                question=question,
                verification_context=verification_context
            )
            response = self.llm.invoke(prompt)
            logger.debug("Generated revised answer")
            return response.content
            
        except Exception as e:
            logger.warning(f"Failed to revise answer: {e}")
            return original_answer

# RAG Chain 
def make_chain(settings: Settings) -> ChainOfVerificationQA:
    # Build RetrievalQA chain with Chroma and OpenAI chat model
    # Load Chroma store 
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

    # Build base QA chain
    base_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": AUTOMOTIVE_PROMPT},
    )
    
    # Wrap with Chain-of-Verification
    return ChainOfVerificationQA(base_chain, llm, retriever, settings)

# CLI 
app = typer.Typer(help="Ask automotive governance questions via RAG")

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the RAG system"),
    k: int = typer.Option(3, help="Number of chunks to retrieve"),
    no_verification: bool = typer.Option(False, "--no-verification", help="Disable Chain-of-Verification"),
    verbose: bool = typer.Option(False, "-v", help="Enable debug logging"),
) -> None:
    # Query the knowledge base and display answer with source info.
    configure_logging(verbose)
    load_dotenv()  # Load API keys from .env

    settings = Settings()
    settings.retrieval_k = k
    settings.enable_verification = not no_verification
    qa = make_chain(settings)

    result = qa.invoke({"query": question})

    # Print answer
    typer.secho("\n🧠 Answer:", fg=typer.colors.GREEN, bold=True)
    typer.echo(result["result"])

    # Print verification info if available
    if settings.enable_verification and "verification_questions" in result:
        typer.secho(f"\n🔍 Verified using {len(result['verification_questions'])} check(s)", fg=typer.colors.YELLOW)

    # Print sources
    sources: List = result.get("source_documents", [])
    typer.secho(f"\n📎 Sources ({len(sources)}):", fg=typer.colors.BLUE, bold=True)
    for idx, doc in enumerate(sources, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        typer.echo(f"{idx}. {src} • p.{page}")

if __name__ == "__main__":
    app()