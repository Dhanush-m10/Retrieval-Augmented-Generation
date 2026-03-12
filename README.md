# Retrieval-Augmented-Generation (RAG)(only answers based on machine learning)

A Streamlit-based RAG project with two apps:

1. **Text RAG App** (`main.py`) – asks questions from a local text file (`documents.txt`).
2. **PDF RAG + OCR App** (`pdf.py`) – lets you upload PDF files and ask questions, including scanned PDFs using OCR.

---

## Features

- Retrieval-Augmented Generation using FAISS vector store
- Embedding models via Sentence Transformers
- LLM responses via Groq API (`llama-3.1-8b-instant`)
- PDF text extraction with PyMuPDF (`fitz`)
- OCR fallback for scanned PDFs using Tesseract + `pytesseract`
- Streamlit web UI

---

## Project Structure

```text
RAG2/
├── main.py
├── pdf.py
├── documents.txt
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── 1️⃣ Install Tesseract OCR.txt
```

---

## Prerequisites

- Python 3.10+ (recommended: 3.11/3.12/3.13)
- Git
- A Groq API key
- (For `pdf.py`) Tesseract OCR installed on Windows

---

## Step-by-Step Setup (Windows)

### 1) Clone the repository

```bash
git clone https://github.com/Dhanush-m10/Retrieval-Augmented-Generation.git
cd Retrieval-Augmented-Generation
```

### 2) Create and activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install Python dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

If any dependency is missing at runtime, install these explicitly:

```powershell
pip install langchain-community langchain-core langchain-text-splitters groq PyMuPDF pytesseract
```

### 4) Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 5) Install Tesseract OCR (required for `pdf.py`)

- Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default path: `C:\Program Files\Tesseract-OCR\`
- During install, enable **Add Tesseract to PATH** (recommended)

This project already sets the Tesseract executable path in `pdf.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

If your install path is different, update that line.

---

## Running the Apps

### Option A: Text RAG App

```powershell
streamlit run main.py
```

What it does:
- Reads content from `documents.txt`
- Creates chunks + embeddings
- Stores vectors in FAISS
- Answers your question from retrieved chunks

### Option B: PDF RAG + OCR App

```powershell
streamlit run pdf.py
```

What it does:
- Upload a PDF in the UI
- Extracts text directly from pages
- If a page has no text layer, falls back to OCR
- Builds FAISS index and answers based on retrieved chunks

---

## Usage Flow (PDF App)

1. Run `streamlit run pdf.py`
2. Open the local URL shown in terminal (usually `http://localhost:8501` or similar)
3. Upload a PDF
4. Wait for **"PDF processed successfully!"**
5. Ask questions in chat input

---

## Common Issues & Fixes

### 1) `ModuleNotFoundError` (example: `fitz`, `pytesseract`)

Install missing packages:

```powershell
pip install PyMuPDF pytesseract
```

### 2) `GROQ_API_KEY not found in .env file`

- Ensure `.env` exists in project root
- Add `GROQ_API_KEY=...`
- Restart Streamlit

### 3) `Port xxxx is not available`

Run on another port:

```powershell
streamlit run pdf.py --server.port 8503
```

### 4) OCR not working

- Verify `C:\Program Files\Tesseract-OCR\tesseract.exe` exists
- Ensure `pytesseract.pytesseract.tesseract_cmd` points to correct path

---

## GitHub Workflow

After changes:

```bash
git add .
git commit -m "Update project"
git push origin main
```


### WORKABLE OUTPUT LINK 

workable links based on my personal api key used via streamlit application(web)
--to add api key to streamlit go to settings->secrets->past u r api key in there and deploy the app 

**Text RAG App** (`main.py`) – asks questions from a local text file (`about machine learning`)-
link - https://retrieval-augmented-generation-de7z2mrqwx9mp4j5vgdjif.streamlit.app/

**PDF RAG + OCR App** (`pdf.py`) – lets you upload PDF files and ask questions, including scanned PDFs using OCR.
link - https://retrieval-augmented-generation-nxjtg6mxhf55do9zmtjsji.streamlit.app/


## Notes

- `.env` and virtual environment are ignored by `.gitignore`.
- For deployment, use Streamlit Cloud or any VM/container with Tesseract installed for OCR support.

---

## Author

Dhanush
