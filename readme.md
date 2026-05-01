# DeepLeaf / PageIndex RAG

This project uses PageIndex to index a PDF and then uses a separate RAG script to answer questions from that PDF.

The workflow has two main steps:

```text
1. run_pageindex.py  -> creates the PageIndex structure JSON
2. pageindex_rag.py  -> uses the PDF + structure JSON to answer questions
```

PageIndex reads a PDF, extracts page-level text, detects or generates a table-of-contents-like structure, assigns page ranges to sections, and saves the result as a structured JSON file. The RAG script then uses that JSON structure to find relevant parts of the PDF before sending context to the local LLM.

---

## Relevant Files

```text
run_pageindex.py
```

Runs the PageIndex indexing step. It takes a PDF and creates a structured JSON file inside the `results/` folder.

```text
pageindex_rag.py
```

Runs the question-answering step. It takes the original PDF, the generated PageIndex JSON file, and a user question. It retrieves relevant context and sends it to the local LLM.

```text
pageindex/page_index.py
```

Contains the main PageIndex algorithm. This is where the PDF is parsed, the TOC/structure is detected, page indexes are assigned, large nodes can be split, and the final JSON tree is built.

```text
pageindex/utils.py
```

Contains helper functions for PDF text extraction, token counting, JSON parsing, node ID creation, node summaries, and LLM calls. The LLM calls are routed through the local vLLM OpenAI-compatible endpoint.

```text
docs/
```

Folder containing the PDF documents to index.

```text
results/
```

Folder where generated PageIndex structure JSON files are saved.

---

## Environment Setup

Before running the program, make sure your local vLLM server is running and that the OpenAI-compatible environment variables point to it.

```bash
export OPENAI_BASE_URL=http://localhost:8010/v1
export OPENAI_API_KEY=dummy
```

You can check that vLLM is reachable with:

```bash
curl http://localhost:8010/v1/models
```

The model used in this setup is:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

When calling PageIndex through LiteLLM/OpenAI-compatible routing, the model can be passed with the `openai/` prefix:

```text
openai/Qwen/Qwen2.5-7B-Instruct-AWQ
```

---

## Step 1: Index the PDF with PageIndex

Run this command from the project root:

```bash
python run_pageindex.py \
    --pdf_path docs/DAOD6000-0.pdf \
    --model openai/Qwen/Qwen2.5-7B-Instruct-AWQ \
    --toc-check-pages 0 \
    --if-add-node-id yes \
    --if-add-node-summary yes
```

This creates a structured JSON file in the `results/` folder.

Expected output file:

```text
results/DAOD6000-0_structure.json
```

The generated JSON contains the PDF structure, including section titles, start/end page indexes, optional node IDs, and optional node summaries.

---

## Step 2: Ask a Question with PageIndex RAG

After the structure JSON is created, run:

```bash
python pageindex_rag.py \
    --pdf-path docs/DAOD6000-0.pdf \
    --structure-json results/DAOD6000-0_structure.json \
    --question "What authority does ADM(IM) have?" \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --verbose
```

This script uses:

```text
- the original PDF
- the PageIndex structure JSON
- the user question
- the local vLLM model
```

The script uses the PageIndex JSON structure to identify relevant sections or page ranges, extracts the corresponding text from the PDF, and sends that context to the LLM to generate an answer.

---

## Full Example

Copy and paste this full example to index the PDF and ask a question:

```bash
export OPENAI_BASE_URL=http://localhost:8010/v1
export OPENAI_API_KEY=dummy

python run_pageindex.py \
    --pdf_path docs/DAOD6000-0.pdf \
    --model openai/Qwen/Qwen2.5-7B-Instruct-AWQ \
    --toc-check-pages 0 \
    --if-add-node-id yes \
    --if-add-node-summary yes

python pageindex_rag.py \
    --pdf-path docs/DAOD6000-0.pdf \
    --structure-json results/DAOD6000-0_structure.json \
    --question "What authority does ADM(IM) have?" \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --verbose
```

---

## Program Flow

```text
PDF document
    ↓
run_pageindex.py
    ↓
PageIndex builds structure JSON
    ↓
results/DAOD6000-0_structure.json
    ↓
pageindex_rag.py
    ↓
Relevant PDF pages/sections are selected
    ↓
Selected context is sent to local vLLM
    ↓
Final answer is generated
```

---

## Current Limitation

The current workflow runs against one selected PDF and one selected PageIndex JSON file at a time.

To support multiple documents, the workflow could be extended to index every PDF in `docs/`, store each structure JSON in `results/`, compare the user question against all document structures, and retrieve context from the most relevant document or documents.
