# AI Usage Log

**Project:** Assignment 5, Option A — Ask My Resume RAG Chatbot  
**Student:** Josh Tsutaoka  
**Course:** Text Mining & Social Media Analytics  
**Note:** This log documents how AI assistance was used for debugging, setup support, code structure, and documentation support during the project. Prompt design decisions, evaluation scoring, and final interpretation were reviewed and modified by me.

---

## Entry 1

**Date:** 2026-05-10  
**Tool:** ChatGPT  
**What I asked:** I asked for a step-by-step explanation of what needed to be completed for Assignment 5 Option A.  
**What I used:** I used the breakdown of the assignment requirements, including the need for a RAG pipeline, Streamlit app, evaluation, README, memo, and AI log.  
**What I modified:** I adapted the plan to my own files and decided to use my resume, cover letter, and LinkedIn About section as the source documents.

---

## Entry 2

**Date:** 2026-05-10  
**Tool:** ChatGPT  
**What I asked:** I asked how to install the required libraries in JupyterLab.  
**What I used:** I used the suggested package list, including LangChain, ChromaDB, sentence-transformers, Streamlit, pypdf, python-dotenv, and OpenAI-related packages.  
**What I modified:** I installed only the libraries needed for my final implementation and later updated `requirements.txt` to match the packages used in the project.

---

## Entry 3

**Date:** 2026-05-10  
**Tool:** ChatGPT  
**What I asked:** I asked how to create and use a `.env` file in JupyterLab.  
**What I used:** I used the guidance to store API keys outside the notebook and load them with `python-dotenv`.  
**What I modified:** I added my own Hugging Face and OpenAI API keys locally, confirmed they loaded correctly, and made sure `.env` would not be committed to GitHub.

---

## Entry 4

**Date:** 2026-05-10  
**Tool:** ChatGPT  
**What I asked:** I asked for help fixing a LangChain import error related to `langchain.text_splitter`.  
**What I used:** I used the updated import path from `langchain_text_splitters` and installed the `langchain-text-splitters` package.  
**What I modified:** I updated my notebook imports so the chunking code worked with the newer LangChain package structure.

---

## Entry 5

**Date:** 2026-05-10  
**Tool:** ChatGPT  
**What I asked:** I asked for help comparing two chunking strategies.  
**What I used:** I used the code structure to compare fixed-size chunking and recursive chunking by number of chunks, average chunk length, minimum length, and maximum length.  
**What I modified:** I reviewed the chunk outputs myself and selected recursive chunking as the final strategy because it better preserved resume, cover letter, and LinkedIn context.

---

## Entry 6

**Date:** 2026-05-10  
**Tool:** ChatGPT  
**What I asked:** I asked for help creating embeddings and a vector store using the free path.  
**What I used:** I used `sentence-transformers/all-MiniLM-L6-v2` for embeddings and ChromaDB for the vector store.  
**What I modified:** I tested the vector store with my own similarity search query and confirmed that it retrieved relevant chunks from my resume and cover letter.

---

## Entry 7

**Date:** 2026-05-10  
**Tool:** ChatGPT  
**What I asked:** I asked for help connecting the vector store to an LLM for the retrieval chain.  
**What I used:** I used the manual RAG chain structure: retrieve chunks, build a prompt, call the LLM, and return the answer with source documents.  
**What I modified:** I initially tried Hugging Face options, but connection errors made them unreliable in my environment. I switched to OpenAI `gpt-4o-mini` after adding API credits, which made the retrieval chain run successfully.

---

## Entry 8

**Date:** 2026-05-10  
**Tool:** ChatGPT  
**What I asked:** I asked for help debugging the Streamlit app because the chatbot kept saying the documents did not contain enough information.  
**What I used:** I used the suggestion to force the app to load the intended files from the same folder as `streamlit_app.py` and to rebuild the vector store from the current documents instead of relying on an old ChromaDB folder.  
**What I modified:** I rewrote the document loading section of `streamlit_app.py`, added app status information in the sidebar, deleted the old ChromaDB cache, and confirmed that the chatbot answered correctly afterward.

---

## Entry 9

**Date:** 2026-05-10  
**Tool:** ChatGPT  
**What I asked:** I asked for help organizing the evaluation section after running 10 chatbot questions.  
**What I used:** I used the suggested structure for summarizing where the chatbot succeeded, where it failed, and what I would improve.  
**What I modified:** I based the actual evaluation on my own chatbot outputs. I identified that the chatbot performed well on factual and inference questions but struggled with document-specific questions.

---

## Entry 10

**Date:** 2026-05-10  
**Tool:** ChatGPT  
**What I asked:** I asked for help creating the final `streamlit_app.py` file with the required components.  
**What I used:** I used the Streamlit structure with `st.chat_input()`, `st.chat_message()`, clickable sample questions, retrieved source chunks, sidebar documentation, caching, and friendly error handling.  
**What I modified:** I tested the app locally, fixed the document loading issue, and confirmed that the final app worked with my own resume, cover letter, and LinkedIn About section.

---

## Case Where My Approach Beat the AI Suggestion

At first, the suggested path used the Hugging Face Inference API or a local Hugging Face pipeline for the LLM. In my environment, Hugging Face repeatedly failed because the connection was closed while trying to access or download the model. I also considered Ollama, but I did not want to install a local model manager. My final approach was better for my setup because I used Hugging Face only for embeddings and OpenAI `gpt-4o-mini` for answer generation. This made the app more reliable while still keeping the embedding and vector store setup lightweight.

Another improvement I made was during the Streamlit app debugging. The first app version used a general folder path and appeared to retrieve the wrong or incomplete context. I changed the app to load only the intended files from the same folder as `streamlit_app.py` and rebuilt the vector store fresh. This fixed the issue where the chatbot kept saying the documents did not contain enough information.
