import os
from ask import ask_question, DEFAULT_MODEL

DEFAULT_PDF_PATH = "./docs/DAOD6000-0.pdf"


def main() -> None:
    if not os.path.isfile(DEFAULT_PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {DEFAULT_PDF_PATH}")

    print("DeepLeaf QA")
    print("Ask a question about the PDF.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            print("Please enter a question.\n")
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        ask_question(DEFAULT_PDF_PATH, question, DEFAULT_MODEL)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()