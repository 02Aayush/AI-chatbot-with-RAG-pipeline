"""Terminal version of the chatbot. Run with:  python connect_memory_with_llm.py"""
from rag_core import DEFAULT_PROVIDER, load_vectorstore, load_llm, build_rag_chain, ask


def main():
    print(f"Using LLM provider: {DEFAULT_PROVIDER}")
    chain = build_rag_chain(load_vectorstore(), load_llm(DEFAULT_PROVIDER))

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer, sources = ask(chain, question)
        print("\nANSWER:\n" + answer)
        print("\nSOURCES:")
        for s in sources:
            print(f"  - {s['file']}, p. {s['page']}")


if __name__ == "__main__":
    main()