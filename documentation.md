# DeepLeaf / PageIndex Documentation

## PageIndex Overview

PageIndex takes a PDF document as input and outputs a structured JSON representation of the document’s contents.

The main function that runs this algorithm is:

```text
page_index.py
```

It does the following:

1. Extracts text page-by-page.
2. Detects whether the PDF has a table of contents (TOC).
3. Uses the TOC, or generates a TOC-like structure if no TOC exists.
4. Figures out which physical PDF page each section starts on.
5. Verifies and fixes wrong page numbers.
6. Converts the flat section list into a tree.
7. Optionally attaches extra information to each node depending on the YAML settings:
    - Node IDs, if `if_add_node_id: "yes"`
    - Node summaries using an LLM, if `if_add_node_summary: "yes"`
    - Document description, if `if_add_doc_description: "yes"`
    - Extracted section text, if `if_add_node_text: "yes"`
8. Saves the structure JSON in a result folder.

---

## Example PageIndex JSON Output

Example of a JSON file with options `a-d` set to `"yes"`:

```json
{
    "doc_name": "string",
    "doc_description": "string",
    "structure": [
        {
            "title": "string",
            "node_id": "string",
            "start_index": "integer",
            "end_index": "integer",
            "summary": "string",
            "text": "string",
            "nodes": [
                {
                    "title": "string",
                    "node_id": "string",
                    "start_index": "integer",
                    "end_index": "integer",
                    "summary": "string",
                    "text": "string",
                    "nodes": []
                }
            ]
        }
    ]
}
```

---

## PageIndexClient Overview

`PageIndexClient` is a wrapper class around `page_index`.

When a JSON is generated with the wrapper, it produces output similar to this:

```json
{
    "id": "12345678-abcd-4321-abcd-123456789abc",
    "type": "pdf",
    "path": "/home/kevinz/project/documents/example_document.pdf",
    "doc_name": "example_document",
    "doc_description": "This document explains the main topic, methods, and conclusions.",
    "page_count": 12,
    "structure": [
        {
            "title": "Introduction",
            "node_id": "0000",
            "start_index": 1,
            "end_index": 3,
            "summary": "This section introduces the purpose and background of the document."
        },
        {
            "title": "Methods",
            "node_id": "0001",
            "start_index": 4,
            "end_index": 10,
            "summary": "This section describes the methods used in the document.",
            "nodes": [
                {
                    "title": "Dataset",
                    "node_id": "0002",
                    "start_index": 5,
                    "end_index": 6,
                    "summary": "This subsection describes the dataset."
                }
            ]
        }
    ],
    "pages": [
        {
            "page": 1,
            "content": "Raw extracted text from physical PDF page 1..."
        }
    ]
}
```

---

## Running `page_index.py`

To run `page_index.py`, use this command:

```bash
python run_pageindex.py --pdf_path path_to_pdf.pdf
```

This generates a JSON file inside the `results/` folder.

---

## Using `PageIndexClient` in a Custom Program

To use `PageIndexClient` in a custom program, import the client class and create a client object.

The client provides a higher-level interface around the PageIndex algorithm. Instead of only generating a standalone JSON file in `results/`, it indexes the PDF, stores the indexed document in a workspace folder, and provides helper functions to retrieve the document structure and selected page content.

```python
from pageindex.client import PageIndexClient
import json

MODEL = "openai/mistralai/Mistral-7B-Instruct-v0.2"

client = PageIndexClient(
    model=MODEL,
    retrieve_model=MODEL,
    workspace=".deep_leaf_workspace",
)

doc_id = client.index("path_to_pdf.pdf", mode="pdf")

structure = json.loads(client.get_document_structure(doc_id))
pages = json.loads(client.get_page_content(doc_id, pages="2-4"))

print(json.dumps(structure, indent=2))
print(json.dumps(pages, indent=2))
```

Here:

- `client.index()` runs the PageIndex algorithm and saves the indexed document to the workspace.
- `client.get_document_structure()` returns the hierarchical PageIndex tree, including section titles, node IDs, summaries, and page ranges.
- `client.get_page_content()` retrieves the raw text from specific physical PDF pages.

This makes `PageIndexClient` useful when building a custom retrieval or question-answering program, because the program can first inspect the tree, select relevant nodes or page ranges, and then retrieve only the needed page content for the LLM.

---

# My Changes to the Original Repository

## Replacement of OpenAI with vLLM

Some files, particularly:

```text
utils.py
page_index.py
```

were modified so the project can run with a local model through vLLM instead of using the OpenAI API directly.

Within `utils.py`, the following functions were adjusted to use an OpenAI-compatible local endpoint:

```text
llm_completion()
llm_acompletion()
```

Instead of sending requests to OpenAI, the functions read environment variables such as:

```bash
OPENAI_BASE_URL=http://localhost:8010/v1
OPENAI_API_KEY=dummy
```

`OPENAI_BASE_URL` points to the local vLLM server, and `OPENAI_API_KEY` can be set to a dummy value.

The configuration file was also adjusted to use the local vLLM model name:

```yaml
model: "openai/mistralai/Mistral-7B-Instruct-v0.2"
retrieve_model: "openai/mistralai/Mistral-7B-Instruct-v0.2"
```

This allows the original PageIndex code to keep using the same function calls, while the actual LLM request is redirected to the local vLLM server.

---

## Addition of Question-Answering Workflows

I created an `ask.py` file that adds a question-answering workflow on top of `PageIndexClient`.

It uses the client functions from the original PageIndex program to:

1. Index the PDF.
2. Retrieve the document tree.
3. Retrieve selected page text.
4. Send the retrieved context to the LLM.

When a question is asked, `ask.py` first creates a `PageIndexClient` object.

It then checks whether the PDF has already been indexed in the workspace.

If the document already exists, it reuses the existing document ID.

If not, it calls:

```python
client.index()
```

to run the PageIndex algorithm and save the indexed JSON document.

After the document is indexed, `ask.py` calls:

```python
client.get_document_structure()
```

to retrieve the PageIndex tree.

The user question and the tree are then sent to the LLM, which selects and returns the page range that should be used to answer the question.

Once the page range is selected, `ask.py` calls:

```python
client.get_page_content()
```

to retrieve the raw text from those pages.

This retrieved page text becomes the context for the final answer.

The question, selected nodes, and retrieved page content are then sent to the LLM, which generates the final response.

---

## Current Limitation

The question-answering workflow is currently hardcoded to only retrieve data from one specific PDF document.

Changes could be made so it can parse through the entire collection of DAOD PDF documents inside the `docs/` folder and select the relevant ones.

For example, the improved workflow could:

1. Loop through all PDFs in `docs/`.
2. Index each PDF using the original PageIndex algorithm.
3. Store each document’s PageIndex JSON in the workspace.
4. Compare the user question against all document trees.
5. Select the most relevant document or documents.
6. Retrieve the needed page ranges.
7. Send the selected context to the LLM for final answer generation.

This would allow the project to answer questions across multiple DAOD documents without modifying the original PageIndex algorithm itself.
