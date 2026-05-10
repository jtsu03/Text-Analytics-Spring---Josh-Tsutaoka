# Ask My Resume RAG Chatbot

## 1. Project Title and Option

**Assignment 5, Option A: Ask My Resume RAG Chatbot**

This project builds a Retrieval-Augmented Generation chatbot that answers questions about my professional background using my resume, cover letter, and LinkedIn About section.

## 2. Author

**Josh Tsutaoka**  
MS in Business Analytics Candidate  
Loyola Marymount University

## 3. Project Description

The goal of this project was to create a career-focused RAG chatbot that could answer recruiter-style questions using my own professional documents. Instead of relying only on a language model's general knowledge, the chatbot first retrieves relevant chunks from my uploaded documents and then uses those chunks as context to generate a grounded answer.

A user can ask questions such as:

- What technical skills does Josh have?
- What projects has Josh worked on?
- Would Josh be a good fit for a data analyst role?
- What does Josh's cover letter say about his ETL experience?

The chatbot is designed to be useful for recruiters, hiring managers, or networking conversations where someone may want a quick summary of my background, skills, and project experience.

## 4. Setup Instructions

### Clone the Repository

```bash
git clone <your-repository-link>
cd bsan6200-assignment5
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

On Windows:

```bash
venv\Scripts\activate
```

On Mac/Linux:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` File

Create a `.env` file in the main project folder and add your OpenAI API key:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

Do not upload the `.env` file to GitHub.

### Run the Notebook

Open and run:

```text
notebooks/rag_pipeline.ipynb
```

The notebook includes document loading, chunking strategy comparison, embeddings, vector store creation, retrieval chain testing, prompt engineering, and evaluation.

### Run the Streamlit App

From the main project folder, run:

```bash
streamlit run streamlit_app.py
```

## 5. Models and Tools Used

### Programming and App Tools

- Python
- Jupyter Notebook
- Streamlit
- LangChain
- ChromaDB
- python-dotenv
- PyPDFLoader
- TextLoader

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model was used to convert document chunks into vector embeddings for similarity search.

### Vector Store

```text
ChromaDB
```

ChromaDB was used to store embedded document chunks and retrieve relevant context based on user questions.

### Language Model

```text
OpenAI gpt-4o-mini
```

OpenAI `gpt-4o-mini` was used to generate final answers after relevant chunks were retrieved.

## 6. Paid vs. Free Path

This project uses a mixed setup.

### Free Components

- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- ChromaDB for the vector store
- Streamlit for the user interface

### Paid Component

- OpenAI `gpt-4o-mini` for answer generation

I initially explored free LLM options, including Hugging Face-based generation and Ollama. Hugging Face generation caused connection issues in my environment, and I chose not to install a local Ollama model. I used OpenAI `gpt-4o-mini` for a more reliable final chatbot while keeping the embedding and vector store setup lightweight.

## 7. Key Findings

The chatbot performed best on broad factual and inference-based questions. It successfully answered questions about my technical skills, programming languages, tools, education, and general fit for data analyst or process analytics roles.

The chatbot also handled out-of-scope questions well. When asked about information not included in the documents, such as salary expectations or GPA, it avoided guessing and stated that the provided documents did not include enough information.

The main weakness was document-specific retrieval. The chatbot sometimes struggled when asked to answer from one specific document, such as the cover letter or LinkedIn About section. This likely happened because the retriever used general similarity search with `k=3`, which did not always retrieve the exact document-specific chunk needed.

Future improvements would include increasing the number of retrieved chunks, adding metadata filtering by document type, and adding more project write-ups to improve answer depth.

## 8. File Descriptions

```text
bsan6200-assignment5/
├── README.md
├── memo.md
├── ai_log.md
├── requirements.txt
├── .gitignore
├── streamlit_app.py
├── Resume.pdf
├── cover_letter.pdf
├── Linkedin_about.txt
├── notebooks/
│   └── rag_pipeline.ipynb
└── evaluation/
    └── evaluation_results.csv
```

### `README.md`

Provides an overview of the project, setup instructions, tools used, key findings, and file descriptions.

### `memo.md`

Technical memo summarizing the project design, methodology, evaluation results, limitations, and future improvements.

### `ai_log.md`

Documents how AI assistance was used throughout the project, including setup help, debugging, code structure, and documentation support.

### `requirements.txt`

Lists all Python dependencies required to run the notebook and Streamlit app.

### `.gitignore`

Prevents sensitive files such as `.env` and local cache files from being committed to GitHub.

### `streamlit_app.py`

The final Streamlit application. It includes a chat interface, clickable sample questions, retrieved source chunks, error handling, sidebar documentation, and cached resource loading.

### `Resume.pdf`

Primary professional document used as source material for the chatbot.

### `cover_letter.pdf`

Additional career document that provides more detail about internship fit, ETL experience, dashboard building, and process analytics interest.

### `Linkedin_about.txt`

Professional summary used as an additional source document for background, skills, and career goals.

### `notebooks/rag_pipeline.ipynb`

Main development notebook. It includes document loading, chunking strategy comparison, embedding creation, vector store testing, retrieval chain construction, prompt testing, and evaluation.

### `evaluation/evaluation_results.csv`

Stores the chatbot evaluation results across 10 test questions, including retrieval quality, answer faithfulness, and answer quality scores.

## Project Workflow

The final RAG pipeline follows this process:

```text
User question
→ Retrieve relevant document chunks from ChromaDB
→ Insert retrieved chunks into grounded prompt
→ Generate answer with OpenAI gpt-4o-mini
→ Display answer and retrieved chunks in Streamlit
```

## Final Notes

This project demonstrates how RAG can be applied to a practical career-focused use case. The chatbot provides a portfolio-ready example of document loading, chunking, embeddings, vector search, prompt engineering, LLM integration, evaluation, and Streamlit deployment.
