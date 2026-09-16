"""
Loader: reads .txt and .docx files from a folder.
"""
import os
from docx import Document


def _read_docx(path: str) -> str:
    """Reads a .docx file and joins its paragraphs with blank lines between
    them, so downstream chunking still sees natural paragraph boundaries."""
    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_documents(data_dir: str) -> list[dict]:
    """
    Returns a list of {"source": filename, "text": content}
    """
    docs = []
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)

        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"source": filename, "text": text})

        elif filename.endswith(".docx"):
            text = _read_docx(path)
            docs.append({"source": filename, "text": text})

    return docs