# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Type

import more_itertools
import numpy as np
from google.cloud.firestore import (  # type: ignore
    Client,
    CollectionReference,
    DocumentSnapshot,
)
from google.cloud.firestore_v1.base_query import BaseFilter  # type: ignore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure  # type: ignore
from google.cloud.firestore_v1.vector import Vector  # type: ignore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .common import client_with_user_agent
from .document_converter import convert_firestore_document
from .version import __version__

USER_AGENT = "langchain-google-firestore-python:vectorstore" + __version__
WRITE_BATCH_SIZE = 500
DEFAULT_TOP_K = 4


class FirestoreVectorStore(VectorStore):
    """Interface for vector store."""

    def __init__(
        self,
        collection: CollectionReference | str,
        embedding_service: Embeddings,
        client: Optional[Client] = None,
        content_field: str = "content",
        metadata_field: str = "metadata",
        embedding_field: str = "embedding",
        distance_strategy: Optional[DistanceMeasure] = DistanceMeasure.COSINE,
        filters: Optional[BaseFilter] = None,
    ) -> None:
        """Constructor for FirestoreVectorStore.

        Args:
            source (CollectionReference | str): The source collection or document
            reference to store the data.
            embedding (Embeddings): The embeddings to use for the vector store.
            client (Optional[Client]): The Firestore client to use. If not provided,
            a new client will be created.
            content_field (str): The field name to store the content data.
            metadata_field (str): The field name to store the metadata.
            embedding_field (str): The field name to store the text embeddings.
            distance_strategy (Optional[DistanceMeasure]): The distance strategy to use for
            calculating distances between vectors. Defaults to DistanceStrategy.COSINE.
            filters (Optional[BaseFilter]): The pre-filters to apply to the query. Defaults to None.
        """

        self.client = client_with_user_agent(USER_AGENT, client)

        if isinstance(collection, str):
            self.collection = self.client.collection(collection)
        else:
            self.collection = collection

        self.embedding_service = embedding_service
        self.content_field = content_field
        self.metadata_field = metadata_field
        self.embedding_field = embedding_field
        self.distance_strategy = distance_strategy
        self.filters = filters

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding_service

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add or update texts in the vector store. If the `ids` are provided, and
        a Firestore document with the same id exists, it will be updated.
        Otherwise, a new Firestore document will be created.

        Args:
            texts: The texts to add to the vector store.
            metadatas: The metadata to add to the vector store. Defaults to None.
            ids: The document ids to use for the new documents. If not provided, new
            document ids will be generated.

        Returns:
            List[str]: The list of document ids added to the vector store.
        """
        texts_len = len(list(texts))
        ids_len_match = not ids or len(ids) == texts_len
        metadatas_len_match = not metadatas or len(metadatas) == texts_len

        if texts_len == 0:
            raise ValueError("No texts provided to add to the vector store.")

        if not metadatas_len_match:
            raise ValueError(
                "The length of metadatas must be the same as the length of texts or zero."
            )

        if not ids_len_match:
            raise ValueError(
                "The length of ids must be the same as the length of texts or zero."
            )

        _ids: List[str] = []
        db_batch = self.client.batch()

        for batch in more_itertools.chunked(texts, WRITE_BATCH_SIZE):
            texts_embs = self.embedding_service.embed_documents(batch)
            for i, text in enumerate(batch):
                doc_id = ids[i] if ids else None
                doc = self.collection.document(doc_id)
                _ids.append(doc.id)

                data = {
                    self.content_field: text,
                    self.embedding_field: Vector(texts_embs[i]),
                    self.metadata_field: metadatas[i] if metadatas else None,
                }

                db_batch.set(doc, data, merge=True)

            db_batch.commit()

        return _ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete documents from the vector store.

        Args:
            ids: The document ids to delete from the vector store.
        """

        if not ids or len(ids) == 0:
            return

        for batch in more_itertools.chunked(ids, WRITE_BATCH_SIZE):
            db_batch = self.client.batch()
            for doc_id in batch:
                doc_ref = self.collection.document(doc_id)
                db_batch.delete(doc_ref)
            db_batch.commit()

    def _similarity_search(
        self,
        query: List[float],
        k: int = DEFAULT_TOP_K,
        filters: Optional[BaseFilter] = None,
    ) -> List[DocumentSnapshot]:
        _filters = filters or self.filters

        wfilters = None

        if _filters is not None:
            wfilters = self.collection.where(filter=_filters)

        results = (wfilters or self.collection).find_nearest(
            vector_field=self.embedding_field,
            query_vector=Vector(query),
            distance_measure=self.distance_strategy,
            limit=k,
        )

        return results.get()

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filters: Optional[BaseFilter] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Firestore.

        Raises:
            FailedPrecondition: If the index is not created.

        Args:
            query: The query text.
            k: The number of documents to return. Defaults to 4.
            filters: The pre-filter to apply to the query. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """

        docs = self._similarity_search(
            self.embedding_service.embed_query(query), k, filters=filters
        )
        return [
            convert_firestore_document(doc, page_content_fields=[self.content_field])
            for doc in docs
        ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filters: Optional[BaseFilter] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Firestore using a vector.

        Raises:
            FailedPrecondition: If the index is not created.

        Args:
            embedding: The query vector.
            k: The number of documents to return. Defaults to 4.
            filters: The pre-filter to apply to the query. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query vector.
        """

        docs = self._similarity_search(embedding, k, filters=filters)
        return [
            convert_firestore_document(doc, page_content_fields=[self.content_field])
            for doc in docs
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filters: Optional[BaseFilter] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run max marginal relevance search on the results of Firestore nearest
        neighbor search.

        Raises:
            FailedPrecondition: If the index is not created.

        Args:
            query: The query text.
            k: The number of documents to return. Defaults to 4.
            fetch_k: The number of documents to fetch. Defaults to 20.
            lambda_mult: The lambda multiplier. Defaults to 0.5.
            filters: The pre-filter to apply to the query. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        query_embedding = self.embedding_service.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filters=filters,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filters: Optional[BaseFilter] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run max marginal relevance search on the results of Firestore nearest
        neighbor search using a vector. This method will throw if the index is
        not created, in which case you will be prompted to create the index.

        Raises:
            FailedPrecondition: If the index is not created.

        Args:
            embedding: The query vector.
            k: The number of documents to return. Defaults to 4.
            fetch_k: The number of documents to fetch. Defaults to 20.
            lambda_mult: The lambda multiplier. Defaults to 0.5.
            filters: The pre-filter to apply to the query. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query vector.
        """
        doc_results = self._similarity_search(embedding, fetch_k, filters=filters)
        doc_embeddings = [
            self._vector_to_list(d.to_dict()[self.embedding_field]) for d in doc_results
        ]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            doc_embeddings,
            lambda_mult=lambda_mult,
            k=k,
        )
        return [convert_firestore_document(doc_results[i]) for i in mmr_doc_indexes]

    @classmethod
    def from_texts(
        cls: Type[FirestoreVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection: Optional[str | CollectionReference] = None,
        **kwargs: Any,
    ) -> FirestoreVectorStore:
        """Create a FirestoreVectorStore instance and add texts to it.

        Args:
            texts: The texts to add to the vector store.
            embedding: The embeddings to use to generate the vectors from texts.
            metadatas: The metadata to add to the vector store. Defaults to None.

        Returns:
            FirestoreVectorStore: The FirestoreVectorStore instance.
        """
        if collection is None:
            raise ValueError("Must provide 'collection' named parameter.")

        vs_obj = cls(collection=collection, embedding_service=embedding, **kwargs)
        vs_obj.add_texts(texts, metadatas, ids, **kwargs)
        return vs_obj

    def _vector_to_list(self, vector: Vector) -> List[float]:
        return vector.to_map_value()["value"]
