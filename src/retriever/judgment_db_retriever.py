import threading
from typing import Any, List, TypedDict

import chromadb
from src.config import config
from src.llm.factory import get_embedding_function

from src.retriever.crime_db_retriever import CrimeNotFoundError, crime_db_retriever


class JudgmentRecord(TypedDict):
    """单条类案检索结果。"""

    id: str
    document: str
    metadata: dict[str, Any]


RRF_K = 60  # RRF 平滑常数，抑制头部名次的绝对优势


def rrf_fuse(
    ranked_lists: List[List[JudgmentRecord]],
    top_k: int,
    weights: List[float] | None = None,
    k: int = RRF_K,
) -> List[JudgmentRecord]:
    """按加权 Reciprocal Rank Fusion 融合多路召回结果。

    每条记录得分 = Σ weight_r * 1 / (k + 名次_r)，只用名次不用原始分数，因此不需要归一化。
    weights 与 ranked_lists 一一对应，缺省时每路权重为 1.0。
    同分并列时按 id 升序，保证结果稳定。
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[str, float] = {}
    records: dict[str, JudgmentRecord] = {}

    for weight, ranked in zip(weights, ranked_lists):
        for rank, record in enumerate(ranked, start=1):
            doc_id = record["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            records.setdefault(doc_id, record)

    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [records[doc_id] for doc_id in ordered[:top_k]]


class JudgmentDBRetriever:
    def __init__(
        self,
        vector_path: str = config.VECTOR_DB_PATH,
        collection: str = config.COLLECTION_NAME,
    ) -> None:
        self._vector_path = vector_path
        self._collection_name = collection
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._client_lock = threading.Lock()
        self._collection_lock = threading.Lock()

    @property
    def client(self) -> chromadb.ClientAPI:
        client = self._client
        if client is None:
            with self._client_lock:
                client = self._client
                if client is None:
                    client = chromadb.PersistentClient(path=self._vector_path)
                    self._client = client
        return client

    @property
    def collection(self) -> chromadb.Collection:
        collection = self._collection
        if collection is None:
            client = self.client
            with self._collection_lock:
                collection = self._collection
                if collection is None:
                    collection = client.get_collection(
                        name=self._collection_name,
                        embedding_function=get_embedding_function(),
                    )
                    self._collection = collection
        return collection

    def query_judgments_by_ids(self, ids: List[str]) -> List[JudgmentRecord]:
        result = self.collection.get(ids=ids)
        return [
            JudgmentRecord(id=id, document=document, metadata=dict(metadata))
            for id, document, metadata in zip(
                result["ids"], result["documents"] or [], result["metadatas"] or []
            )
        ]

    def query_judgments_hybrid(self, query: str, cause: str, tags: List[str]) -> List[JudgmentRecord]:
        """语义召回 + 标签召回，按 RRF 融合后返回 top-k。

        cause 校验、空标签判断都由被调用的两个方法负责，这里只做融合。
        """
        ranked_lists = [
            self.query_judgments_by_text(query, cause),
            self.query_judgments_by_tags(cause, tags),
        ]
        weights = [config.VECTOR_RECALL_WEIGHT, config.TAG_RECALL_WEIGHT]
        return rrf_fuse(ranked_lists, config.HYBRID_RESULT_SIZE, weights)

    def _ensure_supported_cause(self, cause: str) -> None:
        if cause not in crime_db_retriever.get_all_crime_name():
            raise CrimeNotFoundError(cause)

    def query_judgments_by_text(self, query: str, cause: str) -> List[JudgmentRecord]:
        self._ensure_supported_cause(cause)
        res = self.collection.query(
            query_texts=[query],
            n_results=config.VECTOR_RECALL_SIZE,
            where={"cause": cause},
            include=["documents", "metadatas"],
        )
        if not res["ids"] or not res["ids"][0]:
            return []
        return [
            JudgmentRecord(id=doc_id, document=document, metadata=metadata)
            for doc_id, document, metadata in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
            )
        ]

    def query_judgments_by_tags(
        self, cause: str, tags: List[str]
    ) -> List[JudgmentRecord]:
        self._ensure_supported_cause(cause)
        wanted = [tag.strip() for tag in tags if tag and tag.strip()]
        if not wanted:
            return []

        tag_clauses = [{"tags": {"$contains": tag}} for tag in wanted]
        tag_filter = tag_clauses[0] if len(tag_clauses) == 1 else {"$or": tag_clauses}
        where = {"$and": [{"cause": cause}, tag_filter]} if cause else tag_filter

        res = self.collection.get(where=where, include=["documents", "metadatas"])
        scored: list[tuple[int, str, JudgmentRecord]] = []
        for doc_id, document, metadata in zip(
            res["ids"], res["documents"] or [], res["metadatas"] or []
        ):
            stored_tags = (metadata or {}).get("tags") or []
            overlap = sum(1 for tag in wanted if tag in stored_tags)
            scored.append(
                (
                    overlap,
                    doc_id,
                    JudgmentRecord(id=doc_id, document=document, metadata=dict(metadata)),
                )
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[: config.TAG_RECALL_SIZE]]

judgment_db_retriever = JudgmentDBRetriever()
