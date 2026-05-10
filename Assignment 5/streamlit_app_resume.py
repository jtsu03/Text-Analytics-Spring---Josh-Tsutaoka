import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ask My Resume",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Ask My Resume")
st.write(
    "A Retrieval-Augmented Generation chatbot that answers questions about "
    "Josh Tsutaoka's professional background."
)


# ---------------------------------------------------------
# Load API key
# ---------------------------------------------------------
load_dotenv(".env", override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("About This Project")
st.sidebar.write(
    """
    This app answers questions about Josh Tsutaoka's resume, cover letter,
    and LinkedIn About section.

    It uses a RAG pipeline:
    1. Retrieve relevant chunks from career documents
    2. Insert those chunks into a grounded prompt
    3. Generate an answer using an LLM
    """
)

st.sidebar.markdown("### Documents Used")
st.sidebar.write("- Resume")
st.sidebar.write("- Cover letter")
st.sidebar.write("- LinkedIn About section")

st.sidebar.markdown("### Technical Setup")
st.sidebar.write("- Embeddings: sentence-transformers/all-MiniLM-L6-v2")
st.sidebar.write("- Vector store: ChromaDB")
st.sidebar.write("- LLM: OpenAI gpt-4o-mini")
st.sidebar.write("- Retrieval: Top 3 chunks")


# ---------------------------------------------------------
# Load documents
# ---------------------------------------------------------
@st.cache_resource
def load_documents():
    """
    Loads only the intended career documents from the same folder
    as streamlit_app.py.
    """
    app_dir = Path(__file__).parent
    documents = []

    allowed_files = [
        "Resume.pdf",
        "cover_letter.pdf",
        "Linkedin_about.txt"
    ]

    for file_name in allowed_files:
        file_path = app_dir / file_name

        if file_path.exists():
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())

            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents.extend(loader.load())
        else:
            st.sidebar.warning(f"Missing file: {file_name}")

    return documents


# ---------------------------------------------------------
# Create vector store
# ---------------------------------------------------------
@st.cache_resource
def create_vector_store():
    documents = load_documents()

    if len(documents) == 0:
        st.error(
            "No documents were loaded. Make sure Resume.pdf, cover_letter.pdf, "
            "and Linkedin_about.txt are in the same folder as streamlit_app.py."
        )
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build fresh vector store from the current documents
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectordb, len(documents), len(chunks)


# ---------------------------------------------------------
# Initialize LLM
# ---------------------------------------------------------
@st.cache_resource
def initialize_llm():
    if not openai_api_key:
        st.error("OpenAI API key not found. Please check your .env file.")
        st.stop()

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=openai_api_key
    )


# ---------------------------------------------------------
# Final prompt
# ---------------------------------------------------------
prompt_v3 = """
You are a professional career assistant answering questions about Josh Tsutaoka's background.

Use only the provided context from Josh's resume, cover letter, and LinkedIn About section.
Do not use outside knowledge, guess, or add information that is not supported by the context.

If the context does not contain enough information to answer the question, respond:
"The provided documents do not include enough information to answer that."

Keep the response concise, professional, and recruiter-friendly.
Use bullet points when they make the answer easier to read.
For job-fit questions, explain the answer using specific evidence from the provided context.
"""


def build_prompt(question, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    return f"""
{prompt_v3}

Context:
{context}

Question:
{question}

Answer:
"""


def ask_resume_bot(question, vectordb, llm, k=3):
    retrieved_docs = vectordb.similarity_search(question, k=k)
    prompt = build_prompt(question, retrieved_docs)
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "source_documents": retrieved_docs
    }


# ---------------------------------------------------------
# Load resources with friendly error handling
# ---------------------------------------------------------
try:
    vectordb, document_count, chunk_count = create_vector_store()
    llm = initialize_llm()

    st.sidebar.markdown("### App Status")
    st.sidebar.success("App loaded successfully")
    st.sidebar.write(f"Loaded document sections: {document_count}")
    st.sidebar.write(f"Created chunks: {chunk_count}")

except Exception as e:
    st.error(
        "Something went wrong while loading the app. "
        "Please check your documents, packages, API key, or internet connection."
    )
    st.caption(f"Technical detail: {e}")
    st.stop()


# ---------------------------------------------------------
# Sample question buttons
# ---------------------------------------------------------
st.subheader("Try a sample question")

sample_questions = [
    "What technical skills does Josh have?",
    "What projects has Josh worked on?",
    "Describe Josh's most recent role."
]

cols = st.columns(3)

for i, sample_question in enumerate(sample_questions):
    with cols[i]:
        if st.button(sample_question):
            st.session_state["selected_question"] = sample_question


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ---------------------------------------------------------
# Chat interface
# ---------------------------------------------------------
user_question = st.chat_input("Ask a question about Josh's background...")

if st.session_state.selected_question:
    user_question = st.session_state.selected_question
    st.session_state.selected_question = None


if user_question:
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        try:
            result = ask_resume_bot(user_question, vectordb, llm, k=3)

            answer = result["answer"]
            source_documents = result["source_documents"]

            st.write(answer)

            # Display retrieved chunks below each answer
            with st.expander("View retrieved source chunks"):
                for i, doc in enumerate(source_documents, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.markdown(f"**Source:** `{doc.metadata.get('source')}`")
                    st.write(doc.page_content[:1000])
                    st.divider()

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as e:
            st.error(
                "Sorry, something went wrong while generating the answer. "
                "Please check the API key, internet connection, or document setup."
            )
            st.caption(f"Technical detail: {e}")