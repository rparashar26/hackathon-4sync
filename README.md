**Policy AI Assistant - Project Overview**

**Team Members Names:**
•	Tejas Tambvekar
•	Rewa Parashar
•	Urmila Bhartal
•	Ishwari Khismatrao

**Summary**
We are building a lightweight AI assistant that answers natural language questions over internal policy/SOP documents without using external APIs by using open-source models and local document embedding + retrieval techniques.

**Example:**
Answer employee queries (like “What is the leave policy?”) by searching through internal SOPs and documents — without using APIs or cloud services.

**System Overview:**
•	Document Loader: Load PDF/Word/Text policy documents.
•	Text Splitter: Break long documents into smaller chunks.
•	Embedding Model: Convert text chunks into numerical vectors.
•	Vector Store: Save and search embeddings locally.
•	LLM (Local Model): Use a local model to answer user questions.
•	Retriever + Prompt: Retrieve top-k relevant chunks and use them to answer.

**Tools & Libraries**
•	Text Extraction: PyMuPDF, docx, pdfplumber
•	Text Splitting: LangChain, nltk, spacy
•	Embedding Model: sentence-transformers
•	Vector Search: FAISS
•	Local LLM (small model): LLama.cpp, Ollama, GPT4All
•	Interface (optional): Streamlit, Flask

**Step-by-Step Guide (Python-based)**
1. Install dependencies
pip install sentence-transformers, langchain, PyMuPDF, pdfplumber streamlit
2. Load & Split SOP Documents
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
Model Selection
Users can choose between phi3, tinyllama, llama3, or gemma:2b models. The app uses `ollama run` to automatically start the selected model.
How Model Internally Works
1.	Step 1: Language Detection – Uses langdetect.
2.	Step 2: Query Answering – Uses the selected LLM.
3.	Step 3: Grammar & Spelling Fix – Uses pyspellchecker.
4.	Step 4: Translate Back – Translates the result back to user's language.
5.	Step 5: Show Sources – Displays the document source of the answer.
   
**Step-by-Step Setup & Usage Guide**
Step 1: Install Python – https://www.python.org/downloads/
Step 2: Set Up Project Folder

internal-policy-assistant/
├── app.py                     # Main Streamlit app
├── requirements.txt           # Dependencies list
├── pdfs/                      # Your internal policy documents
└── chroma/                    # (auto-generated) for vector storage
Step 3: Add requirements.txt
Step 4: Add Internal Documents in 'pdfs/' folder
Step 5: Install and Run Ollama – https://ollama.com/download
Step 6: Run the Streamlit App – `streamlit run app.py`

**Library Explanations**
•	streamlit: For building frontend chat interface.
•	langchain: For document processing, splitting, embeddings, and retrieval.
•	ollama: Runs local LLMs like llama3, phi3, etc.
•	deep_translator: For translating queries/answers without APIs.
•	langdetect: To detect user query language.
•	pyspellchecker: To correct spelling in model responses.
•	os and subprocess: Used for system file handling and background model execution.
