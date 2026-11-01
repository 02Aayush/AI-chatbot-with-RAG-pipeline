"""Shared RAG logic: used by both the CLI (connect_memory_with_llm.py) and the UI (medibot.py)."""
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ---------- settings (one place to change them) ----------
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # must match create_memory_for_llm.py
TOP_K = 3                                                     # how many chunks to retrieve

PROVIDERS = ["groq", "claude", "huggingface"]
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

SYSTEM_PROMPT = """You are a medical reference assistant.
Answer the user's question using ONLY the context below, taken from the Gale Encyclopedia of Medicine.
If the answer is not in the context, say "I don't know based on the encyclopedia." Do not make anything up.
Start the answer directly. No small talk.

Context:
{context}"""


# ---------- step 1: the knowledge base ----------
def load_vectorstore():
    """Load the FAISS index built by create_memory_for_llm.py."""
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# ---------- step 2: the LLM ----------
def load_llm(provider):
    """Return a LangChain model for 'groq', 'claude' or 'huggingface'."""
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0.2,     # low = factual, repeatable answers
            max_tokens=1024,     # gpt-oss also spends tokens on reasoning, so give it room
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=CLAUDE_MODEL,
            max_tokens=1024,     # no temperature: current Claude models reject sampling params
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    if provider == "huggingface":
        return HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            temperature=0.5,
            max_new_tokens=512,
            huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
        )

    raise ValueError(f"Unknown provider '{provider}'. Use one of {PROVIDERS}.")


# ---------- step 3: wire retriever + prompt + LLM into one chain ----------
def build_rag_chain(vectorstore, llm, k=TOP_K):
    """question -> retrieve k chunks -> fill prompt -> LLM -> answer."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    answer_chain = create_stuff_documents_chain(llm, prompt)   # puts the chunks into {context}
    return create_retrieval_chain(retriever, answer_chain)     # runs the retriever first


# ---------- step 4: ask a question ----------
def ask(chain, question):
    """Run the chain and return (answer_text, list_of_sources)."""
    result = chain.invoke({"input": question})
    # result = {"input": question, "context": [Document, ...], "answer": "..."}
    return result["answer"], format_sources(result["context"])


def format_sources(docs):
    """Turn raw Documents into readable citations."""
    sources = []
    for doc in docs:
        sources.append({
            "file": os.path.basename(doc.metadata.get("source", "unknown")),
            "page": doc.metadata.get("page_label", doc.metadata.get("page", "?")),
            "snippet": doc.page_content[:300].replace("\n", " "),
        })
    return sources