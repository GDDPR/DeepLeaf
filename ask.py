import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

from pageindex.client import PageIndexClient
from pageindex.utils import extract_json, llm_completion


DEFAULT_MODEL = "openai/mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_WORKSPACE = "ask_results"


def parse_json(raw: Any) -> Any:
    """
    PageIndexClient retrieval functions return JSON strings.
    This safely converts them back into Python objects.
    """
    if isinstance(raw, (dict, list)):
        return raw

    if not isinstance(raw, str):
        return raw

    try:
        return json.loads(raw)
    except Exception:
        return raw


def recover_page_selection_from_raw_output(raw: str) -> Dict[str, Any]:
    """
    Recover the pages field when a local LLM returns almost-valid JSON.

    Example problem:
        "reason": "the "Authorities" section is relevant"

    The unescaped quotes around Authorities break json.loads(), but the pages
    value is often still present and usable.
    """
    if not isinstance(raw, str):
        return {}

    pages_match = re.search(
        r'"pages"\s*:\s*"(?P<pages>[^"\n]+)"',
        raw,
        flags=re.IGNORECASE,
    )

    if not pages_match:
        return {}

    pages = pages_match.group("pages").strip()

    safe_pages_match = re.search(r'[0-9,\-\s]+', pages)
    pages = safe_pages_match.group(0).strip() if safe_pages_match else ""

    if not pages:
        return {}

    selected_nodes = []
    node_ids = re.findall(r'"node_id"\s*:\s*"(?P<node_id>[^"]+)"', raw)
    titles = re.findall(r'"title"\s*:\s*"(?P<title>[^"]+)"', raw)
    starts = re.findall(r'"start_index"\s*:\s*(?P<start>\d+)', raw)
    ends = re.findall(r'"end_index"\s*:\s*(?P<end>\d+)', raw)

    for index, node_id in enumerate(node_ids):
        node = {"node_id": node_id}

        if index < len(titles):
            node["title"] = titles[index]
        if index < len(starts):
            node["start_index"] = int(starts[index])
        if index < len(ends):
            node["end_index"] = int(ends[index])

        selected_nodes.append(node)

    return {
        "pages": pages,
        "reason": "Recovered page range from malformed JSON returned by the local LLM.",
        "selected_nodes": selected_nodes,
    }


def find_existing_doc_id(client: PageIndexClient, pdf_path: str) -> Optional[str]:
    """
    Reuse an already-indexed PDF from the PageIndexClient workspace.

    This prevents run.py from rebuilding the PageIndex tree every time
    you ask a new question.
    """
    target_path = os.path.abspath(os.path.expanduser(pdf_path))

    for doc_id, doc in client.documents.items():
        existing_path = doc.get("path", "")

        if not existing_path:
            continue

        existing_path = os.path.abspath(os.path.expanduser(existing_path))

        if existing_path == target_path:
            return doc_id

    return None


def get_or_index_document(
    client: PageIndexClient,
    pdf_path: str,
    reindex: bool = False,
) -> str:
    """
    Either reuse an existing PageIndex document or index the PDF.
    """
    if not reindex:
        existing_doc_id = find_existing_doc_id(client, pdf_path)

        if existing_doc_id:
            print(f"Using existing indexed document: {existing_doc_id}")
            return existing_doc_id

    print("Indexing document with PageIndexClient...")
    doc_id = client.index(pdf_path, mode="pdf")
    print(f"Indexed document ID: {doc_id}")

    return doc_id


def compact_structure_for_prompt(
    structure: List[Dict[str, Any]],
    max_chars: int = 12000,
) -> str:
    """
    Convert the PageIndex tree into a prompt-friendly JSON string.

    PageIndexClient saves PDF structure without full text fields,
    so this usually includes titles, node_ids, start/end pages, summaries,
    and child nodes.
    """
    structure_text = json.dumps(structure, ensure_ascii=False, indent=2)

    if len(structure_text) > max_chars:
        print(
            f"Warning: structure prompt trimmed from "
            f"{len(structure_text)} chars to {max_chars} chars."
        )
        structure_text = structure_text[:max_chars]

    return structure_text


def choose_pages_from_structure(
    question: str,
    structure: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    PageIndex-style retrieval step.

    Instead of keyword-searching raw PDF pages, this asks the LLM
    to inspect the PageIndex tree and choose relevant nodes/page ranges.
    """
    structure_text = compact_structure_for_prompt(structure)

    prompt = f"""
You are selecting relevant document sections from a PageIndex tree.

You are given:
1. A user question.
2. A hierarchical document tree.

Each tree node may contain:
- title
- node_id
- start_index
- end_index
- summary
- nodes

Your job:
Choose the most relevant section nodes for answering the question.

Rules:
- Prefer specific child nodes over broad parent nodes when possible.
- Use only page ranges that appear in the tree.
- If multiple sections are needed, include multiple page ranges.
- If unsure, choose the closest relevant high-level section.
- Do not answer the question yet.
- Return only valid JSON.
- Do not wrap the JSON in markdown.
- Do not put double quotation marks inside any string values. Use single quotes if needed.
- Keep the reason fields short.

Return format:
{{
    "pages": "page range string like 5-7 or 3,8 or 5-7,10-12",
    "reason": "brief reason for selecting these pages",
    "selected_nodes": [
        {{
            "node_id": "string or null",
            "title": "section title",
            "start_index": 1,
            "end_index": 3,
            "reason": "brief reason"
        }}
    ]
}}

Question:
{question}

PageIndex tree:
{structure_text}
"""

    raw = llm_completion(model=model, prompt=prompt)
    selected = extract_json(raw)

    if not isinstance(selected, dict):
        selected = recover_page_selection_from_raw_output(raw)

    if isinstance(selected, dict) and not selected.get("pages"):
        recovered = recover_page_selection_from_raw_output(raw)
        if recovered.get("pages"):
            selected = recovered

    if not isinstance(selected, dict):
        raise ValueError(
            "The LLM did not return a JSON object when selecting pages.\n"
            f"Raw output:\n{raw}"
        )

    if not selected.get("pages"):
        raise ValueError(
            "The LLM did not select any pages.\n"
            f"Parsed output:\n{json.dumps(selected, indent=2, ensure_ascii=False)}\n"
            f"Raw output:\n{raw}"
        )

    selected["pages"] = str(selected["pages"]).strip()
    return selected


def retrieve_page_context(
    client: PageIndexClient,
    doc_id: str,
    pages: str,
) -> List[Dict[str, Any]]:
    """
    Retrieve actual page text using PageIndexClient.
    """
    raw_page_content = client.get_page_content(doc_id, pages)
    page_content = parse_json(raw_page_content)

    if isinstance(page_content, dict) and "error" in page_content:
        raise ValueError(page_content["error"])

    if not isinstance(page_content, list):
        raise ValueError(
            "Unexpected result from client.get_page_content().\n"
            f"Pages requested: {pages}\n"
            f"Result:\n{raw_page_content}"
        )

    return page_content


def build_context(
    page_content: List[Dict[str, Any]],
    max_chars: int = 12000,
) -> str:
    """
    Convert retrieved pages into a final context block for the answer LLM.
    """
    parts = []
    current_len = 0

    for page in page_content:
        page_num = page.get("page")
        content = page.get("content", "")

        chunk = f"[Page {page_num}]\n{content}\n"

        if current_len + len(chunk) > max_chars and parts:
            break

        parts.append(chunk)
        current_len += len(chunk)

    return "\n\n".join(parts)


def clean_answer_pages_line(answer: str, pages: str) -> str:
    """
    Force a clean final Pages used line.

    Some local models try to generate their own page citation line,
    sometimes like:
        Pages used:
        [Page 2]

    This removes the model-generated citation line and appends the
    selected PageIndex page range instead.
    """
    if "Pages used:" in answer:
        answer = answer.split("Pages used:")[0].rstrip()

    return answer.rstrip() + f"\n\nPages used: {pages}"


def answer_from_pages(
    question: str,
    pages: str,
    page_content: List[Dict[str, Any]],
    selected_info: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Ask the LLM to answer using only the pages retrieved by PageIndexClient.
    """
    context = build_context(page_content)
    selected_nodes = selected_info.get("selected_nodes", [])

    prompt = f"""
You are answering a question about a PDF.

Use only the provided retrieved page context.
If the answer is unclear from the retrieved pages, say you are not sure based on the retrieved pages.
Be concise.

Question:
{question}

Selected PageIndex nodes:
{json.dumps(selected_nodes, ensure_ascii=False, indent=2)}

Retrieved page context:
{context}

Answer:
"""

    answer = llm_completion(model=model, prompt=prompt)
    return clean_answer_pages_line(answer, pages)


def ask_question(
    pdf_path: str,
    question: str,
    model: str = DEFAULT_MODEL,
    workspace: str = DEFAULT_WORKSPACE,
    reindex: bool = False,
) -> None:
    """
    Compatibility function for your existing run.py.

    Your run.py calls:
        ask_question(DEFAULT_PDF_PATH, question, DEFAULT_MODEL)

    This function keeps that interface, but internally uses PageIndexClient.
    """
    client = PageIndexClient(
        model=model,
        retrieve_model=model,
        workspace=workspace,
    )

    doc_id = get_or_index_document(
        client=client,
        pdf_path=pdf_path,
        reindex=reindex,
    )

    raw_structure = client.get_document_structure(doc_id)
    structure = parse_json(raw_structure)

    if isinstance(structure, dict) and "error" in structure:
        raise ValueError(structure["error"])

    if not isinstance(structure, list):
        raise ValueError(
            "Unexpected structure returned by PageIndexClient.\n"
            f"Result:\n{raw_structure}"
        )

    selected_info = choose_pages_from_structure(
        question=question,
        structure=structure,
        model=model,
    )

    pages = selected_info["pages"]

    print("\nSelected pages:")
    print(pages)

    print("\nReason:")
    print(selected_info.get("reason", ""))

    if selected_info.get("selected_nodes"):
        print("\nSelected PageIndex nodes:")
        print(json.dumps(selected_info["selected_nodes"], indent=2, ensure_ascii=False))

    page_content = retrieve_page_context(
        client=client,
        doc_id=doc_id,
        pages=pages,
    )

    answer = answer_from_pages(
        question=question,
        pages=pages,
        page_content=page_content,
        selected_info=selected_info,
        model=model,
    )

    print("\nAnswer:\n")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions using PageIndexClient tree retrieval."
    )

    parser.add_argument(
        "--pdf_path",
        required=True,
        help="Path to the PDF file.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the PDF.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LiteLLM/OpenAI-compatible model name.",
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help="Workspace folder for indexed documents.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing even if the PDF already exists in the workspace.",
    )

    args = parser.parse_args()

    ask_question(
        pdf_path=args.pdf_path,
        question=args.question,
        model=args.model,
        workspace=args.workspace,
        reindex=args.reindex,
    )


if __name__ == "__main__":
    main()