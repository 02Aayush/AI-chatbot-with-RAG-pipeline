"""MediBot: Streamlit chat UI over the RAG chain. Run with:  streamlit run medibot.py"""
import streamlit as st

from rag_core import PROVIDERS, DEFAULT_PROVIDER, load_vectorstore, load_llm, build_rag_chain, ask

st.set_page_config(page_title="MediBot", page_icon="🩺")


# ---------- cached resources: built once, reused on every rerun ----------
@st.cache_resource(show_spinner="Loading the medical knowledge base...")
def get_vectorstore():
    return load_vectorstore()


@st.cache_resource(show_spinner=False)
def get_chain(provider):
    return build_rag_chain(get_vectorstore(), load_llm(provider))


# ---------- small UI helper ----------
def show_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['file']}, p. {s['page']}**")
            st.caption(s["snippet"] + "...")


# ---------- the page ----------
def main():
    st.title("🩺 MediBot")
    st.caption("Answers come only from the Gale Encyclopedia of Medicine. "
               "Educational use only, not medical advice. In an emergency, call your local emergency number.")

    with st.sidebar:
        provider = st.selectbox("LLM provider", PROVIDERS, index=PROVIDERS.index(DEFAULT_PROVIDER))
        if st.button("Clear chat"):
            st.session_state.messages = []

    # chat history survives reruns inside st.session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                show_sources(msg["sources"])

    question = st.chat_input("Ask a medical question...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the encyclopedia..."):
                answer, sources = ask(get_chain(provider), question)
        except Exception as e:
            st.error(f"Could not get an answer: {e}")
            return
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


main()