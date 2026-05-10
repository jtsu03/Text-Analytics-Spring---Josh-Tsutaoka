# Technical Memo: Ask My Resume RAG Chatbot

**To:** Text Mining & Social Media Analytics Instructor  
**From:** Josh Tsutaoka  
**Date:** May 10, 2026  
**Subject:** Technical Summary of Ask My Resume RAG Chatbot

---

## Overview

For Assignment 5, I built an “Ask My Resume” Retrieval-Augmented Generation chatbot. The goal of the project was to create a recruiter-facing chatbot that can answer questions about my professional background using my actual career documents. The chatbot uses my resume, cover letter, and LinkedIn About section as the source material.

The final application was built with a RAG pipeline and deployed through Streamlit. A user can ask questions such as “What technical skills does Josh have?” or “Would Josh be a good fit for a data analyst role?” The system retrieves relevant document chunks, passes them into a grounded prompt, and generates an answer using an LLM.

---

## Data Sources

I used three career documents as the chatbot’s knowledge base:

1. **Resume.pdf**  
   Includes education, technical skills, tools, experience, and project background.

2. **cover_letter.pdf**  
   Includes more detailed explanations of my ETL experience, dashboard development, data processing, and interest in process analytics.

3. **Linkedin_about.txt**  
   Provides a professional summary of my academic background, analytics experience, and career goals.

These documents were selected because they represent the types of information a recruiter or hiring manager would want to ask about.

---

## Document Loading and Chunking

The documents were loaded using LangChain document loaders. PDF files were loaded with `PyPDFLoader`, while the LinkedIn About section was loaded with `TextLoader`.

I tested two chunking strategies:

- **Fixed-size chunking**
- **Recursive character chunking**

I selected **RecursiveCharacterTextSplitter** for the final pipeline because it better preserved paragraph and line-break structure. This was important because resume and career documents often contain short sections, bullet points, and grouped information. Recursive chunking helped keep related skills, projects, and experience details together.

The final chunking settings were:

```python
chunk_size = 500
chunk_overlap = 50
separators = ["\n\n", "\n", " ", ""]
