import uuid
import re

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..schemas.documents import PolicyChunk, RetrievalResult
from ..config.settings import settings


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a string chunk ID to a deterministic UUID for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class HybridRetriever:
    """
    Hybrid retriever combining BM25 sparse search and dense vector search
    with Reciprocal Rank Fusion (RRF) for score combination.

    BM25 catches exact keyword matches while dense retrieval captures
    semantic similarity; RRF merges both rankings without requiring
    compatible score scales.
    """

    def __init__(
        self,
        qdrant_url: str = settings.qdrant_url,
        qdrant_api_key: str | None = settings.qdrant_api_key,
        collection_name: str = settings.collection_name,
        embedding_model: str = settings.embedding_model,
    ) -> None:
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim: int = self.embedder.get_sentence_embedding_dimension()
        self.collection_name = collection_name

        # BM25 state — rebuilt after every add_documents call
        self.bm25: BM25Okapi | None = None
        self.bm25_corpus: list[list[str]] = []
        self.bm25_ids: list[str] = []  # UUID strings matching corpus order
        self.bm25_payload: dict[str, dict] = {}  # id -> payload for post-filtering

        self.rrf_k: int = settings.rrf_k
        self.alpha: float = settings.hybrid_search_alpha

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not yet exist."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    )
                },
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list[PolicyChunk]) -> None:
        """Embed and upsert document chunks, then rebuild BM25 index."""
        if not chunks:
            return

        texts = [chunk.full_text for chunk in chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)

        points = [
            PointStruct(
                id=_chunk_id_to_uuid(chunk.chunk_id),
                vector={"dense": embedding.tolist()},
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "text": chunk.text,
                    "context_prefix": chunk.context_prefix,
                    "full_text": chunk.full_text,
                    **chunk.metadata,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        """Reload all payloads from Qdrant (with pagination) and rebuild the BM25 index."""
        all_points = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=10_000,
                offset=offset,
                with_payload=True,
            )
            if not batch:
                break
            all_points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            return

        self.bm25_ids = [str(p.id) for p in all_points]
        self.bm25_payload = {str(p.id): (p.payload or {}) for p in all_points}
        self.bm25_corpus = [
            self._tokenize(self.bm25_payload[str(p.id)].get("text", ""))
            for p in all_points
        ]
        self.bm25 = BM25Okapi(self.bm25_corpus)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, strip punctuation, and split into tokens."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 2]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[str, float]],
        dense_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """
        Combine BM25 and dense rankings with Reciprocal Rank Fusion.

        RRF score = Σ weight / (k + rank)

        alpha controls the balance:
            0 → pure BM25, 1 → pure dense
        """
        rrf_scores: dict[str, float] = {}

        bm25_weight = 1.0 - self.alpha
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores.setdefault(doc_id, 0.0)
            rrf_scores[doc_id] += bm25_weight / (self.rrf_k + rank + 1)

        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores.setdefault(doc_id, 0.0)
            rrf_scores[doc_id] += self.alpha / (self.rrf_k + rank + 1)

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: int = settings.retrieval_top_k,
        filter_policy_type: str | None = None,
        exclude_document_id: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieval: BM25 + dense vectors fused with RRF.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            filter_policy_type: Restrict results to this policy type.
            exclude_document_id: Skip chunks from this document.

        Returns:
            List of RetrievalResult sorted by descending RRF score.
        """
        # Build Qdrant filter
        filter_conditions: list[FieldCondition] = []
        if filter_policy_type:
            filter_conditions.append(
                FieldCondition(
                    key="policy_type",
                    match=MatchValue(value=filter_policy_type),
                )
            )
        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        # --- BM25 retrieval ---
        if self.bm25 and self.bm25_corpus:
            query_tokens = self._tokenize(query)
            bm25_scores_arr = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(bm25_scores_arr)[::-1][: top_k * 2]
            bm25_results: list[tuple[str, float]] = [
                (self.bm25_ids[i], float(bm25_scores_arr[i]))
                for i in top_indices
                if bm25_scores_arr[i] > 0
            ]
            # Apply the same policy_type filter as the dense search
            if filter_policy_type:
                bm25_results = [
                    (uid, score) for uid, score in bm25_results
                    if self.bm25_payload.get(uid, {}).get("policy_type") == filter_policy_type
                ]
        else:
            bm25_results = []

        # --- Dense retrieval ---
        query_embedding = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        dense_raw = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_embedding),
            limit=top_k * 2,
            query_filter=query_filter,
            with_payload=True,
        )
        dense_results: list[tuple[str, float]] = [
            (str(hit.id), hit.score) for hit in dense_raw
        ]

        # --- RRF fusion ---
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results)

        # --- Build result objects ---
        bm25_lookup = dict(bm25_results)
        dense_lookup = dict(dense_results)
        payload_lookup = {str(hit.id): hit.payload for hit in dense_raw}

        results: list[RetrievalResult] = []
        for point_uuid, rrf_score in fused[:top_k]:
            payload = payload_lookup.get(point_uuid)

            if not payload:
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_uuid],
                    with_payload=True,
                )
                if points:
                    payload = points[0].payload

            if not payload:
                continue

            doc_id = payload.get("document_id", "")

            # Honour exclusion filter for the current policy document
            if exclude_document_id and doc_id == exclude_document_id:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=payload.get("chunk_id", point_uuid),
                    document_id=doc_id,
                    content=payload.get("text", ""),
                    section_title=payload.get("section_title", ""),
                    bm25_score=bm25_lookup.get(point_uuid, 0.0),
                    dense_score=dense_lookup.get(point_uuid, 0.0),
                    rrf_score=rrf_score,
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in {"text", "full_text", "context_prefix"}
                    },
                )
            )

        return results

    def collection_count(self) -> int:
        """Return the number of points currently in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0
