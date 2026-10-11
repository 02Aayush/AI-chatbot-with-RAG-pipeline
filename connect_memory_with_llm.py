import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
# print("HF_TOKEN: =================== ", os.getenv("HF_TOKEN"))

# step 1: setup LLM
# Pick the provider in .env with LLM_PROVIDER = groq (default) | claude | huggingface

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# HUGGINGFACE_REPO_ID = "tiiuae/falcon-7b-instruct"
# HUGGINGFACE_REPO_ID = "meta-llama/Llama-2-7b-chat-hf"

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

def load_llm(provider):
    if provider == "groq":
        # Primary: free tier, very fast Llama inference
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0.2,  # low = more factual, less creative
            max_tokens=512,
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    if provider == "claude":
        # Backup / quality comparison: paid Anthropic API
        # (no temperature here: current Claude models reject sampling params)
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    if provider == "huggingface":
        # Original setup (may fail now that HF serverless moved to Inference Providers)
        return HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            temperature=0.5,
            max_new_tokens=512,
            huggingfacehub_api_token=HF_TOKEN
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use groq, claude or huggingface.")


# step 2: Connect LLM with FAISS and Create chain 

CUSTOM_PROMPT_TEMPLATE = """ 
Use the pieces of information provided in the context to answer user's question.
if you dont know the answer, just say that you dont know, don't try to make up an answer.
Dont provide anything out of the given context.

Context: {context}
Question: {question}

start the answer direectly. NO small talk please.
"""

def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA Chain

qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(LLM_PROVIDER),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs={'k':3}), #search in nearest doc
    return_source_documents= True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Invoke chain with single query
print(f"Using LLM provider: {LLM_PROVIDER}")
user_query = input("write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])

