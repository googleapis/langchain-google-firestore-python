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

import itertools
from typing import TYPE_CHECKING, Iterator, List, Optional

from google.cloud import firestore  # type: ignore
from google.cloud.firestore import DocumentReference  # type: ignore
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from more_itertools import chunked

from .common import client_with_user_agent
from .document_converter import (
    DOC_REF,
    FIRESTORE_TYPE,
    convert_firestore_document,
    convert_langchain_document,
)
from .version import __version__

USER_AGENT_LOADER = "langchain-google-firestore-python:document_loader/" + __version__
USER_AGENT_SAVER = "langchain-google-firestore-python:document_saver/" + __version__
WRITE_BATCH_SIZE = 500


if TYPE_CHECKING:
    from google.cloud.firestore import Client, CollectionGroup, Query


class FirestoreLoader(BaseLoader):
    def __init__(
        self,
        source: Query | CollectionGroup | DocumentReference | str,
        page_content_fields: List[str] = [],
        metadata_fields: List[str] = [],
        client: Client = None,
    ) -> None:
        """Document Loader for Google Cloud Firestore.

        Args:
            source: The source to load the documents. It can be an instance of Query,
                CollectionGroup, DocumentReference or the single `/`-delimited path to
                a Firestore collection.
            page_content_fields: The document field names to write into the `page_content`.
                If an empty or None list is provided all fields will be written into
                `page_content`. When only one field is provided only the value is written.
            metadata_fields: The document field names to write into the `metadata`. By default
                it will write all fields that are not in `page_content`into `metadata`.
            client: Client for interacting with the Google Cloud Firestore API.
        """
        self.client = client_with_user_agent(USER_AGENT_LOADER, client)
        self.source = source
        self.page_content_fields = page_content_fields
        self.metadata_fields = metadata_fields

    def load(self) -> List[Document]:
        """Load Documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        query = None
        if isinstance(self.source, DocumentReference):
            self.source._client = client_with_user_agent(
                USER_AGENT_LOADER, self.source._client
            )
            yield convert_firestore_document(self.source.get())
        else:
            if isinstance(self.source, str):
                query = self.client.collection(self.source)
            else:
                query = self.source
                client_with_user_agent(USER_AGENT_LOADER, query._client)

            for document_snapshot in query.stream():
                yield convert_firestore_document(
                    document_snapshot,
                    self.page_content_fields,
                    self.metadata_fields,
                )


class FirestoreSaver:
    """Write into Google Cloud Platform `Firestore`."""

    def __init__(
        self,
        collection: Optional[str] = None,
        client: Optional[Client] = None,
    ) -> None:
        """Document Saver for Google Cloud Firestore.

        Args:
            collection: The single `/`-delimited path to a Firestore collection. If
                this value is present it will write documents with an auto generated id.
            client: Client for interacting with the Google Cloud Firestore API.
        """
        self.collection = collection
        self.client = client_with_user_agent(USER_AGENT_SAVER, client)

    def upsert_documents(
        self,
        documents: List[Document],
        merge: Optional[bool] = False,
        document_ids: Optional[List[str]] = None,
    ) -> None:
        """Create / merge documents into the Firestore database.

        Args:
            documents: List of documents to be written into Firestore.
            merge: To merge data with an existing document (creating if
                the document does not exist).
            document_ids: List of document ids to be used. By default it
                will try to construct the document paths using the `reference`
                from the Document.
        """
        db_batch = self.client.batch()

        if document_ids and len(document_ids) != len(documents):
            raise ValueError(
                "`documents` and `document_ids` parameters must be the same length"
            )

        docs_list = itertools.zip_longest(documents, document_ids or [])

        for batch in chunked(docs_list, WRITE_BATCH_SIZE):
            for doc, doc_id in batch:
                document_dict = convert_langchain_document(doc, self.client)
                if self.collection:
                    doc_ref = self.client.collection(self.collection).document()
                elif doc_id:
                    doc_ref = DocumentReference(*doc_id.split("/"), client=self.client)
                elif document_dict.get("reference", {}).get(FIRESTORE_TYPE) == DOC_REF:
                    doc_ref = DocumentReference(
                        *document_dict["reference"]["path"].split("/"),
                        client=self.client,
                    )
                else:
                    raise ValueError(
                        "Unable to construct document_path for document: " + str(doc)
                    )

                db_batch.set(
                    reference=doc_ref,
                    document_data=document_dict["data"],
                    merge=merge,
                )
            db_batch.commit()

    def delete_documents(
        self,
        documents: List[Document],
        document_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete documents from the Firestore database.

        Args:
            documents: List of documents to be deleted from Firestore.
                It will try to extract the {document_path} from the `reference`
                in the document metadata.
            document_ids: List of documents ids to be deleted from Firestore.
                If provided the `documents` argument will be ignored.

        """
        db_batch = self.client.batch()

        docs_list = itertools.zip_longest(documents, document_ids or [])

        for batch in chunked(docs_list, WRITE_BATCH_SIZE):
            for doc, doc_id in batch:
                document_path = None
                if doc_id:
                    document_path = doc_id
                elif doc:
                    document_dict = convert_langchain_document(doc, self.client)
                    if (
                        document_dict.get("reference", {}).get(FIRESTORE_TYPE)
                        == DOC_REF
                    ):
                        document_path = document_dict["reference"]["path"]
                if not document_path:
                    raise ValueError(
                        "Unable to construct document_path for document: "
                        + str(doc)
                        + "or doc_id: "
                        + str(doc_id)
                    )
                doc_ref = DocumentReference(
                    *document_path.split("/"), client=self.client
                )
                db_batch.delete(doc_ref)
            db_batch.commit()
