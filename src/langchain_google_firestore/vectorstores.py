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

import asyncio
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, TypeVar

import more_itertools
from google.cloud.firestore import (  # type: ignore
    Client,
    CollectionGroup,
    DocumentReference,
    Query,
)
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

USER_AGENT = "langchain-google-firestore-python:vectorstore"
IMPORT_ERROR_MSG = """`google-cloud-firestore` package not found,
please run `pip3 install google-cloud-firestore`"""
WRITE_BATCH_SIZE = 500


VST = TypeVar("VST", bound=VectorStore)


class FirestoreVectorStore(VectorStore):
    """Interface for vector store."""

    _DEFAULT_FIRESTORE_DATABASE = "(default)"

    def __init__(
        self,
        source: Query | CollectionGroup | DocumentReference | str,
        embedding: Embeddings,
        client: Optional[Client] = None,
        content_field="content",
        metadata_fields: Optional[List[str]] = None,
        ignore_metadata_fields: Optional[List[str]] = None,
        text_embedding_field: Optional[str] = "embeddings",
        distance_strategy: Optional[DistanceMeasure] = DistanceMeasure.COSINE,
    ) -> None:
        """Constructor for FirestoreVectorStore.

        Args:
            source (Query | CollectionGroup | DocumentReference | str): The source
                collection or document reference to store the data.
            embeddings (Embeddings): The embeddings to use for the vector store.
            client (Optional[Client], optional): The Firestore client to use. If
                not provided, a new client will be created. Defaults to None.
            content_field (str, optional): The field name to store the content
                data. Defaults to "content".
            metadata_fields (Optional[List[str]], optional): The list of metadata
                fields to store. Defaults to None.
            ignore_metadata_fields (Optional[List[str]], optional): The list of
                metadata fields to ignore. Defaults to None.
            text_embedding_field (Optional[str], optional): The field name to
                store the text embeddings. Defaults to "embeddings".
            distance_strategy (Optional[DistanceStrategy], optional): The distance
                strategy to use for calculating distances between vectors.
                Defaults to DistanceStrategy.COSINE.

        Raises:
            ImportError: If the `firestore` package is not found.

        Example:
            .. code-block:: python

            from langchain_google_firestore.vectorstores import Firestore
            from langchain.embeddings import GooglePalmEmbeddings

            embeddings = GooglePalmEmbeddings()
            firestore_vecstore = Firestore(source="my_collection", embeddings=embeddings)
        """
        try:
            from google.cloud import firestore
        except ModuleNotFoundError as exc:
            raise ImportError(IMPORT_ERROR_MSG) from exc

        # Check if the client is provided, otherwise create a new client with
        # the default client info.
        self.client = client or firestore.Client()

        client_agent = self.client._client_info.user_agent
        if not client_agent:
            self.client._client_info.user_agent = USER_AGENT
        elif USER_AGENT not in client_agent:
            self.client._client_info.user_agent = " ".join([client_agent, USER_AGENT])

        if isinstance(source, str):
            self.source = self.client.collection(source)
        elif isinstance(source, DocumentReference):
            self.source = source.parent
        else:
            self.source = source

        self.embedding = embedding
        self.content_field = content_field
        self.metadata_fields = metadata_fields
        self.ignore_metadata_fields = ignore_metadata_fields
        self.text_embedding_field = text_embedding_field
        self.distance_strategy = distance_strategy

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        collection: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        if isinstance(self.source, CollectionGroup) and not collection:
            raise ValueError(
                "The `collection` path must be provided when using CollectionGroup."
            )

        ids = []
        db_batch = self.client.batch()

        for batch in more_itertools.chunked(texts, WRITE_BATCH_SIZE):
            texts_embs = self.embedding.embed_documents(batch)
            for i, text in enumerate(batch):
                doc = self.source.document()
                ids.append(doc.id)
                data = {
                    self.content_field: text,
                    self.text_embedding_field: Vector(texts_embs[i]),
                }
                if metadatas:
                    data.update(metadatas[i])
                db_batch.set(doc, data)
            db_batch.commit()

        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        collection: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.add_texts, **kwargs), texts, metadatas, collection
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        raise NotImplementedError

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        raise NotImplementedError

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search, query, k=k, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError

    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search_with_score, *args, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search_by_vector, embedding, k=k, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(
            self.max_marginal_relevance_search,
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        raise NotImplementedError

    @classmethod
    async def afrom_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(cls.from_texts, **kwargs), texts, embedding, metadatas
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        raise NotImplementedError
