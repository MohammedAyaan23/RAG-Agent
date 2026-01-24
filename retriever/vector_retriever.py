import chromadb
from chromadb.config import Settings
from typing import List
import logging

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Vector-based retriever using ChromaDB.
    """

    def __init__(
        self,
        collection_name: str,
        persist_path: str = "./injestion/chroma_db",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, query: str, n_chunks: int) -> List[str]:
        """
        Retrieve top-N most relevant chunks for a query.
        """

        logger.info(f"Retrieving {n_chunks} chunks for query: {query}")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_chunks,
        )

        # Chroma returns a list per query
        documents = results.get("documents", [])

        if not documents or not documents[0]:
            return []

        return documents[0]
