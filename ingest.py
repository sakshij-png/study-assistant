from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

CHUNK_SIZE = 800
OVERLAP = 150


def load_pdfs(folder="data"):
    pages = []
    for pdf in Path(folder).glob("*.pdf"):
        for i, page in enumerate(PdfReader(pdf).pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "source": pdf.name, "page": i})
    return pages


def chunk_pages(pages):
    chunks = []
    for p in pages:
        t = p["text"]
        for start in range(0, len(t), CHUNK_SIZE - OVERLAP):
            piece = t[start:start + CHUNK_SIZE]
            if len(piece) > 100:
                chunks.append({**p, "text": piece})
    return chunks


def build_index(folder="data", db_path="db", model=None):
    pages = load_pdfs(folder=folder)
    if not pages:
        return 0, 0

    chunks = chunk_pages(pages)
    if not chunks:
        return len(pages), 0

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection("notes")  # rebuild from scratch each run
    except Exception:
        pass
    col = client.create_collection("notes", metadata={"hnsw:space": "cosine"})

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    col.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return len(pages), len(chunks)


if __name__ == "__main__":
    pages, chunks = build_index()
    print(f"Loaded {pages} pages")
    if pages == 0:
        raise SystemExit("No text found. Check that PDFs are in the data folder and are not scanned images.")

    print(f"Created {chunks} chunks")
    print("Index built and saved in the db folder.")