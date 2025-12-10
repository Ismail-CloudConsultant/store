import os
import hashlib
import tempfile
import dotenv
import streamlit as st

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load .env (for OPENAI_API_KEY etc.)
dotenv.load_dotenv()
class FlowFormatterParser(BaseOutputParser):
    """
    Cleans LLM output and formats it into a numbered flow.
    Removes markdown artifacts, extra newlines, quotes,
    and enforces a consistent bullet structure.
    """

    def parse(self, text: str) -> str:
        if not text:
            return ""
        
        cleaned = text.strip().strip('"').strip("'")

        
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        cleaned = "\n".join(lines)

        
        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("* ", "- ")
        cleaned = cleaned.replace("•", "- ")

        
        for i in range(1, 21):
            cleaned = cleaned.replace(f"{i}.", f"{i})")

        # 5. Ensure double newlines between top-level items
        final_lines = []
        for line in cleaned.split("\n"):
            # a very simple heuristic: if line starts with "1)" or "2)" etc
            if (len(line) > 1 and line[0:2].isdigit() and ")" in line[:4]) or (
                line and line[0].isdigit() and ")" in line[:3]
            ):
                final_lines.append("\n" + line)
            else:
                final_lines.append(line)
        cleaned = "\n".join(final_lines).lstrip()

        return cleaned

    @property
    def _type(self) -> str:
        return "flow_formatter_parser"

parser = FlowFormatterParser()

prompt_template = PromptTemplate(
    template="""
    You are an expert coding assistant.
    GO THROUGH THE CODE AND UNDERSTAND IT AND GIVE A DETAILED ANSWER
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
""",
    input_variables=["context", "question"],
)

def hash_files(file_contents_list):
    h = hashlib.sha256()
    for name, content in file_contents_list:
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(content.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

def build_vectorstore_from_files(file_contents_list, chunk_size=1000, chunk_overlap=200, embedding_model="text-embedding-3-large"):
    """
    Builds embeddings + FAISS vectorstore from a list of (filename, content) tuples.
    This is cached by Streamlit for given inputs.
    """
    # Create Documents
    docs = []
    for fname, content in file_contents_list:
        docs.append(Document(page_content=content, metadata={"source": fname}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)

    # build embeddings & vectorstore
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

def format_docs(retrieved_docs):
    # join page_content with simple separators and also include filename metadata
    parts = []
    for d in retrieved_docs:
        src = d.metadata.get("source", "unknown")
        parts.append(f"### Source: {src}\n{d.page_content}")
    return "\n\n".join(parts)

st.set_page_config(page_title="Code Q&A (LangChain + Streamlit)", layout="wide")
st.title("Code explainer App   - ismail  :-) ")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size", value=1000, min_value=256, max_value=5000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", value=200, min_value=0, max_value=2000, step=50)
    embedding_model = st.selectbox("Embedding model", options=["text-embedding-3-large", "text-embedding-3-small"], index=0)
    llm_model = st.selectbox("LLM (ChatOpenAI)", options=["gpt-4o", "gpt-4o-mini", "gpt-4o-mini-tts"], index=0)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)
    k = st.slider("Retriever: top k", 1, 60, 30)
    fetch_k = st.slider("Retriever: fetch_k (for mmr)", 1, 120, 60)
    st.markdown("---")
    st.write("Be sure to set `OPENAI_API_KEY` in your environment (or in a .env file).")

uploaded = st.file_uploader("Upload one or more .py files (or drag & drop)", accept_multiple_files=True, type=["py"])
if not uploaded:
    st.info("Drop .py files to begin. The app will index them and let you ask queries.")
    st.stop()

# Read uploaded files into memory
file_contents_list = []
for f in uploaded:
    try:
        raw = f.read().decode("utf-8")
    except Exception:
        # fallback: streamlit returns bytes sometimes
        raw = f.getvalue().decode("utf-8")
    file_contents_list.append((f.name, raw))

# compute a stable key for caching (so embeddings/vectorstore rebuild only when inputs change)
files_hash = hash_files(file_contents_list)

# build or reuse vectorstore (streamlit cache)
with st.spinner("Indexing uploaded files and building vectorstore (this may take a few seconds)..."):
    vectorstore = build_vectorstore_from_files(
        file_contents_list,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        embedding_model=embedding_model,
    )

# Make retriever
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": int(k), "fetch_k": int(fetch_k)})

st.success(f"Loaded {len(uploaded)} file(s). Vectorstore ready.")

st.subheader("Ask a question (answers will be based only on the uploaded files)")
question = st.text_area("Type your question here", height=120)
ask_btn = st.button("Ask")
if ask_btn:
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Retrieving relevant chunks and querying the LLM..."):

            # Retrieve docs using LangChain v0.2+ API
            retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                st.info("No relevant content found in the uploaded files.")
            else:
                # Format retrieved content
                context_text = format_docs(retrieved_docs)

                # Build the prompt text
                prompt_text = prompt_template.format(
                    context=context_text,
                    question=question
                )

                # Call the LLM
                llm = ChatOpenAI(model=llm_model, temperature=float(temperature))

                try:
                    ai_msg = llm.invoke(prompt_text)    # always returns AIMessage
                    llm_output = ai_msg.content         # extract string
                except Exception as e:
                    st.error(f"LLM call failed: {e}")
                    llm_output = ""

                # Parse the LLM output (parser expects a STRING)
                parsed = parser.parse(llm_output)

                # Show final answer
                st.markdown("**Answer (parsed):**")
                st.code(parsed)

                # Show raw output
                with st.expander("Show raw LLM output"):
                    st.text(llm_output)

                # Show retrieved chunks
                with st.expander("Show sources used (retrieved chunks)"):
                    for i, d in enumerate(retrieved_docs, 1):
                        src = d.metadata.get("source", "unknown")
                        st.markdown(f"**{i}. {src}**")
                        snippet = d.page_content[:1000].replace("\n", " ")
                        st.write(snippet + ("..." if len(d.page_content) > 1000 else ""))
