import os

import subprocess

import time

import streamlit as st

import re

from langchain.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import Chroma

from langchain.llms import Ollama

from langchain.chains import RetrievalQA

from spellchecker import SpellChecker

from deep_translator import GoogleTranslator

from langdetect import detect

# Page setup

st.set_page_config(page_title="📄 Policy Assistant", layout="centered")

st.title("📄 Internal Policy AI Assistant")

st.markdown("Ask questions in any language. Answers will match your language.")

# Model selection

model_choice = st.selectbox("🧠 Choose Model", ["phi3", "tinyllama", "llama3", "gemma:2b"])

# Spell checker

spell = SpellChecker()
ALLOWLIST = {"parkar"}

def correct_spelling(text):
   corrected_words = []
   for word in text.split():
       # Check against allowlist (case-insensitive)
       if word.lower() in ALLOWLIST:
           corrected_words.append(word)
       else:
           corrected = spell.correction(word)
           corrected_words.append(corrected if corrected else word)
   return ' '.join(corrected_words)

# Grammar correction

def fix_grammar(text):

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    corrected_sentences = [s.capitalize() if s else s for s in sentences]

    return ' '.join(corrected_sentences)

# Translation

def detect_language(text):

    try:

        return detect(text)

    except:

        return "en"

def translate_to_english(text):

    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_back(text, lang_code):

    return GoogleTranslator(source='en', target=lang_code).translate(text)

# Run model in background

def ensure_model_running(model_name):

    try:

        subprocess.Popen(["ollama", "run", model_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(2)

    except Exception as e:

        st.error(f"❌ Couldn't run model: {e}")

with st.spinner(f"🔁 Starting model `{model_choice}`..."):

    ensure_model_running(model_choice)

# Vector DB config

CHROMA_DIR = "chroma"

@st.cache_resource

def load_vectorstore():

    if os.path.exists(os.path.join(CHROMA_DIR, "index")):

        return Chroma(persist_directory=CHROMA_DIR, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    docs = []

    for file in os.listdir("pdfs"):

        path = os.path.join("pdfs", file)

        if file.endswith(".pdf"):

            loader = PyMuPDFLoader(path)

        elif file.endswith(".txt"):

            loader = TextLoader(path)

        elif file.endswith(".docx"):

            loader = Docx2txtLoader(path)

        else:

            continue

        docs.extend(loader.load())

    if not docs:

        st.error("❌ No supported files found in `pdfs/` folder.")

        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)

# Load LLM and vectorstore

vectorstore = load_vectorstore()

if vectorstore:

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = Ollama(model=model_choice)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Chat session state

    if "messages" not in st.session_state:

        st.session_state.messages = []

    if st.button("🧹 Clear Chat"):

        st.session_state.messages = []

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

    query = st.chat_input("Ask your question...")

    if query:

        lang_code = detect_language(query)

        with st.chat_message("user"):

            st.markdown(query)

        st.session_state.messages.append({"role": "user", "content": query})

        if lang_code != "en":

            with st.spinner("🌍 Translating your question to English..."):

                translated_query = translate_to_english(query)

        else:

            translated_query = query

        with st.spinner("🤖 Thinking..."):

            result = qa_chain(translated_query)

            english_answer = result["result"]

            sources = result.get("source_documents", [])

        if not sources or english_answer.strip().lower() in ["i don't know", "not sure", ""]:

            fallback = "❓ Sorry, I couldn't find a relevant answer based on the internal documents."

            st.warning(fallback)

            st.session_state.messages.append({"role": "assistant", "content": fallback})

        else:

            corrected = correct_spelling(english_answer)

            final = fix_grammar(corrected)

            if lang_code != "en":

                with st.spinner("🌐 Translating answer to your language..."):

                    translated_answer = translate_back(final, lang_code)

            else:

                translated_answer = final

            with st.chat_message("assistant"):

                st.markdown(translated_answer)

            st.session_state.messages.append({"role": "assistant", "content": translated_answer})

            with st.expander("📚 Sources"):

                for doc in sources:

                    st.markdown(f"**{os.path.basename(doc.metadata['source'])}**")

                    st.write(doc.page_content[:300] + "...")
 