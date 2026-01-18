"""Embedding generation and vector storage with ChromaDB."""

import hashlib
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .config import settings
from .models import EmailMessage

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # Try to break at sentence or word boundary
        if end < text_len:
            # Look for sentence end
            for sep in [". ", ".\n", "\n\n", "\n", " "]:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


class EmbeddingEngine:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding model."""
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


class VectorStore:
    """ChromaDB-based vector storage."""

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        collection_name: Optional[str] = None,
    ):
        """Initialize the vector store."""
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.collection_name = collection_name or settings.chroma_collection_name

        # Initialize ChromaDB with persistent storage
        chroma_path = settings.get_chroma_path()
        chroma_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"Initialized ChromaDB collection '{self.collection_name}' "
            f"with {self.collection.count()} documents"
        )

    def _make_chunk_id(self, email_id: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk."""
        return hashlib.sha256(f"{email_id}:{chunk_index}".encode()).hexdigest()[:16]

    def add_email(self, email: EmailMessage) -> int:
        """Add an email to the vector store. Returns number of chunks added."""
        full_text = email.full_text
        chunks = chunk_text(full_text, settings.chunk_size, settings.chunk_overlap)

        if not chunks:
            return 0

        # Generate embeddings
        embeddings = self.embedding_engine.embed(chunks)

        # Prepare metadata for each chunk
        ids = []
        metadatas = []
        documents = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = self._make_chunk_id(email.message_id, i)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "email_id": email.message_id,
                    "chunk_index": i,
                    "subject": email.subject or "",
                    "from_address": email.from_address or "",
                    "date": email.date.isoformat() if email.date else "",
                    "folder": email.folder_name or "",
                    "has_attachments": len(email.attachments) > 0,
                    "attachment_types": ",".join(
                        set(a.attachment_type.value for a in email.attachments)
                    ),
                }
            )

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        limit: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Search for similar documents."""
        query_embedding = self.embedding_engine.embed_single(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                formatted.append(
                    {
                        "id": chunk_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    }
                )

        return formatted

    def get_email_chunks(self, email_id: str) -> list[dict]:
        """Get all chunks for a specific email."""
        results = self.collection.get(
            where={"email_id": email_id},
            include=["documents", "metadatas"],
        )

        formatted = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                formatted.append(
                    {
                        "id": chunk_id,
                        "document": results["documents"][i],
                        "metadata": results["metadatas"][i],
                    }
                )

        return formatted

    def delete_email(self, email_id: str) -> None:
        """Delete all chunks for an email."""
        # Get all chunk IDs for this email
        results = self.collection.get(
            where={"email_id": email_id},
            include=[],
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
