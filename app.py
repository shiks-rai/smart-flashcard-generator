import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gpt4all import GPT4All

st.set_page_config(page_title="📚 Exam Flashcards Generator", page_icon=":books:")
st.title("📚 Exam Flashcards Generator from PDF (Local Model)")

# Upload PDF
uploaded_file = st.file_uploader("Upload your syllabus PDF", type="pdf")

if uploaded_file:
    with st.spinner("📖 Reading and splitting PDF..."):
        # Save uploaded PDF temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        # Delete temp file
        os.remove(temp_path)

    st.success(f"✅ Loaded {len(pages)} pages; split into {len(docs)} chunks.")

    # Path to your downloaded GGUF model (change path if yours is different)
    model_path = "C:\\model\\mistral-7b-instruct-v0.1.Q4_0.gguf"

    flashcards = []

    with st.spinner("🧠 Generating exam-style flashcards..."):
        with GPT4All(
            model_name=model_path,
            n_ctx=512,
            allow_download=False,
            verbose=False
        ) as llm:
            for i, doc in enumerate(docs[:2]):  # limit to first 8 chunks for demo
                context = doc.page_content.strip()

                # Quick cleaning: remove obvious admin / email lines
                for remove_text in ["Email:", "Course Code:", "Course Title:", "Dear Students,", "Reference Book", "Online Resources"]:
                    context = context.replace(remove_text, "")

                # Prompt for question
                prompt_q = f"""
You are an expert examiner.
Ignore teacher names, emails, course codes, greetings like 'Dear students', and references.
Focus ONLY on technical concepts, definitions, explanations from the text that would appear in an exam.
Write exactly ONE useful, exam-style question.
Text:
{context}

Write only the question:
"""
                question = llm.generate(prompt_q, max_tokens=100).strip()

                # Prompt for answer
                prompt_a = f"""
Write a clear, short model answer to this question:
Question: {question}
Ignore any irrelevant text like emails, course codes, reference book names.
Text:
{context}
"""
                answer = llm.generate(prompt_a, max_tokens=150).strip()

                flashcards.append((question, answer))

    # Show flashcards
    if flashcards:
        st.success("✅ Your generated flashcards:")
        for idx, (q, a) in enumerate(flashcards):
            with st.container():
                st.markdown(f"""
<div style="background-color: #1a1a1a; padding: 16px; border-radius: 10px; margin-bottom:10px; border: 1px solid #444;">
<b>Flashcard #{idx+1}</b><br><br>
<b>Q:</b> {q}<br>
<b>A:</b> {a}
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("⚠ No flashcards generated.")
else:
    st.info("👆 Upload a PDF above to get started!")
