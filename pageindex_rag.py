"""
Local Agentic PageIndex RAG runner (no OpenAI Agents SDK).

This follows the same PageIndex agentic/tool pattern as the official demo:
    - get_document()
    - get_document_structure()
    - get_page_content(pages="5-7")

But it does NOT import or use the OpenAI Agents SDK.
It uses LiteLLM to call your local OpenAI-compatible vLLM server.

Example with local vLLM:
    export OPENAI_API_KEY=dummy
    export OPENAI_BASE_URL=http://localhost:8010/v1

    python agentic_pageindex_rag_local.py \
        --pdf-path docs/DAOD6000-0.pdf \
        --structure-json results/DAOD6000-0_structure.json \
        --question "What authority does ADM(IM) have?" \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --verbose

Notes:
    - You can pass --model Qwen/Qwen2.5-7B-Instruct-AWQ without the openai/ prefix.
      Internally, LiteLLM still needs an OpenAI-compatible route for vLLM, but all calls go to
      OPENAI_BASE_URL, e.g. http://localhost:8010/v1. Nothing is sent to OpenAI.
    - The script uses an existing PageIndex structure JSON by default. If you omit
      --structure-json, it will try to run PageIndexClient.index(), which may trigger indexing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm
import PyPDF2

try:
    from pageindex import PageIndexClient
except Exception:
    PageIndexClient = None  # Only needed if --structure-json is omitted.


SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.

You have access to these tools:
1. get_document()
   Returns document metadata.
2. get_document_structure()
   Returns the PageIndex tree structure without full page text.
3. get_page_content(pages="5-7")
   Returns the text from specific PDF pages. Use tight page ranges only.

Rules:
- First call get_document.
- Then call get_document_structure.
- Then call get_page_content with the most relevant page range(s).
- Never fetch the whole document unless the document is tiny and no specific section is identifiable.
- If a node has end_index smaller than start_index, treat the node as covering start_index only.
- Answer only from retrieved page content.
- Be concise.

At each step, reply ONLY with valid JSON in one of these forms:

Tool call:
{
  "reason": "short reason for this tool call",
  "action": "get_document",
  "args": {}
}

{
  "reason": "short reason for this tool call",
  "action": "get_document_structure",
  "args": {}
}

{
  "reason": "short reason for this tool call",
  "action": "get_page_content",
  "args": {"pages": "5-7"}
}

Final answer:
{
  "final_answer": "answer text here"
}
""".strip()


KNOWN_LITELLM_PREFIXES = (
    "openai/",
    "ollama/",
    "ollama_chat/",
    "hosted_vllm/",
    "vllm/",
    "anthropic/",
    "azure/",
    "gemini/",
)


def litellm_model_name(model: str) -> str:
    """
    Let the user pass the raw vLLM-served model id, e.g.
        Qwen/Qwen2.5-7B-Instruct-AWQ

    LiteLLM needs a provider prefix for OpenAI-compatible servers, so internally
    we use openai/<model> with api_base=OPENAI_BASE_URL. This still calls local
    vLLM, not OpenAI, because api_base points to localhost.
    """
    if model.startswith(KNOWN_LITELLM_PREFIXES):
        return model
    return f"openai/{model}"


def call_llm(model: str, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
    api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY", "dummy")
    llm_model = litellm_model_name(model)

    last_error: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            response = litellm.completion(
                model=llm_model,
                messages=messages,
                temperature=0,
                api_base=api_base,
                api_key=api_key,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse a model response that should contain exactly one JSON object."""
    if not text:
        return {}

    raw = text.strip()
    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    if raw.startswith("```"):
        raw = raw[len("```"):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    # Local models sometimes escape underscores as if writing Markdown.
    raw = raw.replace("\\_", "_")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting the first JSON object from surrounding text.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            candidate = match.group(0).replace("\\_", "_")
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    return {}


def read_structure_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Structure JSON must be a JSON object.")
    if not isinstance(data.get("structure"), list):
        raise ValueError("Structure JSON must contain a top-level 'structure' list.")
    return data


def extract_pdf_pages(pdf_path: str | Path) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append({
                "page": page_number,
                "content": page.extract_text() or "",
            })
    return pages


def fix_invalid_ranges(node_or_nodes: Any) -> Any:
    """Normalize impossible ranges like 5-4 to 5-5 for retrieval safety."""
    if isinstance(node_or_nodes, list):
        for item in node_or_nodes:
            fix_invalid_ranges(item)
        return node_or_nodes

    if isinstance(node_or_nodes, dict):
        start = node_or_nodes.get("start_index")
        end = node_or_nodes.get("end_index")
        if isinstance(start, int) and isinstance(end, int) and end < start:
            node_or_nodes["end_index"] = start
        if isinstance(node_or_nodes.get("nodes"), list):
            fix_invalid_ranges(node_or_nodes["nodes"])
    return node_or_nodes


def parse_pages(pages: str, max_page: int) -> List[int]:
    """Parse strings like '5-7', '3,8', '5-7,10' into sorted page numbers."""
    selected: set[int] = set()
    if not pages or not isinstance(pages, str):
        return []

    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start = int(left.strip())
                end = int(right.strip())
            except ValueError:
                continue
            if end < start:
                end = start
            for page in range(start, end + 1):
                if 1 <= page <= max_page:
                    selected.add(page)
        else:
            try:
                page = int(part)
            except ValueError:
                continue
            if 1 <= page <= max_page:
                selected.add(page)

    return sorted(selected)


class LocalPageIndexTools:
    def __init__(self, pdf_path: str | Path, structure_json: Optional[str | Path], fix_ranges: bool = True):
        self.pdf_path = Path(pdf_path).expanduser().resolve()
        self.pages = extract_pdf_pages(self.pdf_path)
        self.doc_id = str(uuid.uuid4())

        if structure_json:
            data = read_structure_json(structure_json)
            self.doc_name = data.get("doc_name") or self.pdf_path.name
            self.doc_description = data.get("doc_description", "")
            self.structure = data["structure"]
            if fix_ranges:
                self.structure = fix_invalid_ranges(self.structure)
            self.client = None
        else:
            if PageIndexClient is None:
                raise RuntimeError("PageIndexClient could not be imported. Provide --structure-json instead.")
            self.client = PageIndexClient(workspace="agentic_pageindex_local_workspace")
            self.doc_id = self.client.index(str(self.pdf_path), mode="pdf")
            self.doc_name = self.pdf_path.name
            self.doc_description = ""
            self.structure = json.loads(self.client.get_document_structure(self.doc_id))
            if fix_ranges:
                self.structure = fix_invalid_ranges(self.structure)

    def get_document(self) -> str:
        result = {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "doc_description": self.doc_description,
            "page_count": len(self.pages),
            "status": "loaded",
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    def get_document_structure(self) -> str:
        return json.dumps(self.structure, ensure_ascii=False, indent=2)

    def get_page_content(self, pages: str) -> str:
        page_numbers = parse_pages(pages, max_page=len(self.pages))
        result = []
        for page_number in page_numbers:
            page = self.pages[page_number - 1]
            result.append({
                "page": page["page"],
                "content": page["content"],
            })
        return json.dumps(result, ensure_ascii=False, indent=2)


def run_agent(
    tools: LocalPageIndexTools,
    question: str,
    model: str,
    max_steps: int = 6,
    verbose: bool = False,
) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    used_page_content = False

    for step in range(1, max_steps + 1):
        raw = call_llm(model=model, messages=messages)
        action = parse_json_response(raw)

        if verbose:
            print(f"\n[step {step} raw model output]\n{raw}\n")

        if not action:
            messages.append({
                "role": "assistant",
                "content": raw,
            })
            messages.append({
                "role": "user",
                "content": "Your last response was not valid JSON. Reply only with one valid JSON object using the required schema.",
            })
            continue

        if "final_answer" in action:
            final = str(action["final_answer"])
            if not used_page_content:
                messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
                messages.append({
                    "role": "user",
                    "content": "You must call get_page_content before final_answer. Choose the relevant page range now.",
                })
                continue
            return final

        tool_name = action.get("action")
        args = action.get("args") or {}
        reason = action.get("reason", "")

        if verbose:
            print(f"[tool call] {tool_name}({args})")
            if reason:
                print(f"[reason] {reason}")

        if tool_name == "get_document":
            observation = tools.get_document()
        elif tool_name == "get_document_structure":
            observation = tools.get_document_structure()
        elif tool_name == "get_page_content":
            pages = str(args.get("pages", "")).strip()
            observation = tools.get_page_content(pages)
            used_page_content = True
        else:
            observation = json.dumps({
                "error": f"Unknown tool: {tool_name}",
                "available_tools": ["get_document", "get_document_structure", "get_page_content"],
            })

        if verbose:
            preview_chars = 5000
            preview = observation[:preview_chars] + "..." if len(observation) > preview_chars else observation
            print(f"[tool output preview]\n{preview}\n")

        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"Tool output for {tool_name}:\n{observation}"})

    # Fallback: ask for final answer from accumulated tool outputs.
    messages.append({
        "role": "user",
        "content": "Now provide the final answer as valid JSON: {\"final_answer\": \"...\"}. Use only retrieved page content.",
    })
    raw = call_llm(model=model, messages=messages)
    parsed = parse_json_response(raw)
    if "final_answer" in parsed:
        return str(parsed["final_answer"])
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Local agentic PageIndex RAG runner without OpenAI Agents SDK")
    parser.add_argument("--pdf-path", required=True, help="Path to the PDF file")
    parser.add_argument("--structure-json", help="Path to PageIndex structure JSON generated by run_pageindex.py")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--model", required=True, help="Local vLLM model id, e.g. Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum agent tool-use steps")
    parser.add_argument("--verbose", action="store_true", help="Print tool calls and previews")
    parser.add_argument("--no-fix-invalid-ranges", action="store_true", help="Do not normalize impossible ranges like 5-4")
    args = parser.parse_args()

    tools = LocalPageIndexTools(
        pdf_path=args.pdf_path,
        structure_json=args.structure_json,
        fix_ranges=not args.no_fix_invalid_ranges,
    )

    answer = run_agent(
        tools=tools,
        question=args.question,
        model=args.model,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
