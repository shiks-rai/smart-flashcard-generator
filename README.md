# 📚 Smart Flashcard Generator from PDF (Local Model)

A Python + Streamlit app that **automatically generates flashcards** from any syllabus or notes PDF, using a local LLM model (Mistral 7B).

✨ Built as a portfolio / resume project to show:
- LangChain document loading & text splitting
- Local LLM generation with [GPT4All](https://github.com/nomic-ai/gpt4all)
- Clean Streamlit UI with flashcard cards

---

## 🚀 Features
✅ Upload your syllabus or notes as PDF  
✅ Extracts text, splits into chunks  
✅ Uses local language model to generate flashcard questions & answers  
✅ Displays them in a clean, dark-themed interface

Works **completely offline** (after downloading the model).

---

## 📦 Requirements

Python 3.8+ and these Python packages:
- streamlit
- langchain
- gpt4all

Install with:
```bash
pip install -r requirements.txt
