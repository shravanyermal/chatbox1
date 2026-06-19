import os
import json
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Support KB", page_icon="🎧", layout="centered")

# ── Constants ─────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.1-8b-instant"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
RETRIEVER_K   = 4
FAISS_DIR     = "faiss_index"
DOC_MAP_PATH  = f"{FAISS_DIR}/doc_map.json"

Path(FAISS_DIR).mkdir(exist_ok=True)
Path("uploaded_docs").mkdir(exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "doc_map"     not in st.session_state: st.session_state.doc_map     = {}
if "chain"       not in st.session_state: st.session_state.chain       = None
if "chat_store"  not in st.session_state: st.session_state.chat_store  = {}
if "greeted"     not in st.session_state: st.session_state.greeted     = False

# ── Lazy-loaded resources ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    # langchain_huggingface is the correct package — langchain_community.embeddings is deprecated since 0.2.2
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner="Loading language model…")
def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)

# ── FAISS helpers ─────────────────────────────────────────────────────────────
def load_index():
    from langchain_community.vectorstores import FAISS
    if st.session_state.vectorstore is None:
        if Path(f"{FAISS_DIR}/index.faiss").exists():
            st.session_state.vectorstore = FAISS.load_local(
                FAISS_DIR, get_embeddings(), allow_dangerous_deserialization=True
            )
            if Path(DOC_MAP_PATH).exists():
                st.session_state.doc_map = json.loads(Path(DOC_MAP_PATH).read_text())

def save_index():
    st.session_state.vectorstore.save_local(FAISS_DIR)
    Path(DOC_MAP_PATH).write_text(json.dumps(st.session_state.doc_map))

def index_file(file_path: str, filename: str) -> int:
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(file_path).load()
    elif ext == ".docx":
        docs = Docx2txtLoader(file_path).load()
    else:
        docs = TextLoader(file_path, encoding="utf-8").load()

    for doc in docs:
        doc.metadata["source"] = filename

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)

    if not chunks:
        return 0

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(chunks, get_embeddings())
    else:
        st.session_state.vectorstore.add_documents(chunks)

    st.session_state.doc_map[filename] = len(chunks)
    save_index()
    return len(chunks)

def remove_document(filename: str):
    st.session_state.doc_map.pop(filename, None)
    st.session_state.vectorstore = None
    st.session_state.chain = None
    st.session_state.chat_store = {}
    st.session_state.greeted = False

    if not st.session_state.doc_map:
        save_index()
        return

    for fname in list(st.session_state.doc_map.keys()):
        fpath = Path("uploaded_docs") / fname
        if fpath.exists():
            index_file(str(fpath), fname)

# ── Chain builder ─────────────────────────────────────────────────────────────
# Uses only langchain_core LCEL — zero imports from langchain.chains (removed in v1.0)
def get_chain():
    if st.session_state.chain is not None:
        return st.session_state.chain
    if st.session_state.vectorstore is None:
        return None

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.output_parsers import StrOutputParser

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
    )

    def fmt_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Step 1: rephrase the user's latest message into a standalone search query
    # using the conversation history, then retrieve relevant docs
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and the latest user question, "
         "rewrite the question as a standalone search query with no references to prior turns. "
         "Return it unchanged if it is already standalone. Output only the query, nothing else."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Step 2: answer using the retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful and friendly customer support assistant. "
         "Answer the user's question using only the context provided below. "
         "Be concise and specific. "
         "If the answer is not in the context, say you don't have that information "
         "and suggest the user contact support directly.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    llm = get_llm()

    # Full LCEL pipeline — no langchain.chains imports needed
    rag_chain = (
        RunnablePassthrough.assign(
            context=(
                contextualize_prompt
                | llm
                | StrOutputParser()
                | retriever
                | fmt_docs
            )
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in st.session_state.chat_store:
            st.session_state.chat_store[session_id] = InMemoryChatMessageHistory()
        return st.session_state.chat_store[session_id]

    st.session_state.chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return st.session_state.chain

# ── Load existing index on startup ───────────────────────────────────────────
load_index()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎧 Knowledge Base")
    st.caption("Upload company documents to ground the chatbot.")

    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.doc_map:
                with st.spinner(f"Indexing {f.name}…"):
                    save_path = Path("uploaded_docs") / f.name
                    save_path.write_bytes(f.read())
                    try:
                        chunks = index_file(str(save_path), f.name)
                        st.success(f"✅ {f.name} — {chunks} chunks")
                        # Reset chain and greeting so new index takes effect
                        st.session_state.chain = None
                        st.session_state.greeted = False
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")

    st.divider()

    if st.session_state.doc_map:
        st.markdown("**Indexed documents**")
        for name, chunks in st.session_state.doc_map.items():
            col1, col2 = st.columns([4, 1])
            col1.caption(f"{name} · {chunks} chunks")
            if col2.button("🗑️", key=f"del_{name}"):
                with st.spinner(f"Removing {name}…"):
                    remove_document(name)
                st.rerun()
        total = sum(st.session_state.doc_map.values())
        st.caption(f"{total} total chunks in index")
    else:
        st.info("No documents indexed yet.")

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_store = {}
        st.session_state.greeted = False
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Customer Support Assistant")

if not st.session_state.doc_map:
    st.warning("👈 Upload at least one document in the sidebar to start chatting.")
    st.stop()

# ── Opening greeting — shown once after documents are loaded ──────────────────
if not st.session_state.greeted:
    greeting = (
        "Hello! I'm your customer support assistant. "
        "I've loaded your knowledge base and I'm ready to help.\n\n"
        "To get started, could you briefly describe the problem or question you have? "
        "The more detail you give, the better I can assist you."
    )
    st.session_state.messages.append({"role": "assistant", "content": greeting, "sources": []})
    st.session_state.greeted = True

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Describe your problem or ask a question…"):
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            chain = get_chain()
            answer = chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": "default"}}
            )
        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": [],
    })