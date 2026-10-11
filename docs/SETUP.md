# Setup & run (Windows, PowerShell)

Each step says **what** to run and **why**. Run them from VS Code's terminal or PowerShell.

## 0. Prerequisites
- **Python 3.12** (already installed — `python --version` → `Python 3.12.0`).
  *Why:* `torch 2.8`, `faiss-cpu 1.12` and `numpy 2.3` all ship ready-made wheels for 3.12. Avoid 3.13+ for this pinned stack.
- **Git** (repo already cloned).
- ~3 GB free disk (torch + models).

## 1. Go to the project folder
```bash
cd "C:\Projects\AI chatbot with RAG\AI-chatbot-with-RAG-pipeline"
```

## 2. Create a virtual environment
```bash
python -m venv venv
```
*Why:* a venv is a private copy of Python just for this project, so these exact package versions
don't clash with other projects or your global Python. `venv/` is in `.gitignore`.

## 3. Activate it
```bash
.\venv\Scripts\Activate.ps1
```
*Why:* makes `python`/`pip` point at the venv. Your prompt will start with `(venv)`.
If PowerShell blocks scripts, run once (current user only):
```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 4. Install dependencies
```bash
python -m pip install --upgrade pip
```
```bash
pip install -r requirements.txt
```
*Why:* `requirements.txt` pins every version (LangChain 0.3 stack, FAISS, sentence-transformers,
torch, plus the new `langchain-groq` and `langchain-anthropic`). Pinning = reproducible; the code
was written against these APIs. Expect a few minutes — torch is the big one.
**Don't** upgrade to LangChain 1.x yet: `RetrievalQA` moved out of `langchain.chains` there.

Optional, smaller CPU-only torch (skip if the normal install worked):
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

## 5. Add your API keys
```bash
Copy-Item .env.example .env
```
Then open `.env` and fill in:
- `GROQ_API_KEY` — free at <https://console.groq.com/keys> (primary).
- `ANTHROPIC_API_KEY` — your paid Claude key (backup / comparison).
- `HF_TOKEN` — only if you want to try the original Hugging Face path.

*Why:* keys are read by `python-dotenv` at startup and `.env` is git-ignored, so secrets never
get committed. `.env.example` documents what's needed without real values.

## 6. (Optional) Rebuild the vector index
```bash
python create_memory_for_llm.py
```
*Why:* only needed if you change the PDFs, chunk size or embedding model — the index in
`vectorstore/db_faiss` is already built and committed. On CPU this takes several minutes
(embedding ~7,000 chunks). First run downloads the MiniLM model (~90 MB) to your HF cache.

## 7. Ask a question
```bash
python connect_memory_with_llm.py
```
You'll see `Using LLM provider: groq`, then a prompt. Try:
- `What are the symptoms of diabetes?`
- `How is tuberculosis transmitted?`
- `What is a neural network?` (not in a medical encyclopedia → should answer "I don't know")

You get `RESULT:` (the answer) and `SOURCE DOCUMENTS:` (the 3 chunks it used, with page metadata).

## 8. Switch to Claude
In `.env` set:
```
LLM_PROVIDER=claude
```
and run step 7 again. Ask the same questions and compare answers. `CLAUDE_MODEL` defaults to
`claude-opus-5-5` (most capable); set `CLAUDE_MODEL=claude-sonnet-5-5` or `claude-haiku-4-5`
for cheaper runs. Set `LLM_PROVIDER=groq` to go back.

## Troubleshooting
| Symptom | Meaning / fix |
|---|---|
| `ModuleNotFoundError` | venv not activated (step 3) or install failed (step 4). |
| `401` / `invalid api key` | Key missing or wrong in `.env`; no quotes/spaces needed around the value. |
| `429` / rate limit | Groq free tier per-minute limit — wait a minute, or switch provider. |
| `allow_dangerous_deserialization` warning/error | `index.pkl` is a Python pickle. It's safe here because you built it; never load pickles from strangers. |
| FAISS load error about dimensions | Index was built with a different embedding model — rebuild (step 6). |
| Slow first run | Downloading MiniLM + loading torch; later runs are faster. |
| `huggingface` provider fails | Expected — HF moved serverless models to Inference Providers. Use groq/claude. |
