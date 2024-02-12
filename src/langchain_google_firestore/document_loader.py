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

from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    List,
    Optional,
)

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from .utility.document_converter import DocumentConverter

DEFAULT_FIRESTORE_DATABASE = "(default)"
USER_AGENT = "LangChain"
IMPORT_ERROR_MSG = (
    "`firestore` package not found, please run `pip3 install google-cloud-firestore`"
)
WRITE_BATCH_SIZE = 500

if TYPE_CHECKING:
    from google.cloud.firestore_v1.client import Client
    from google.cloud.firestore_v1.document import DocumentReference
    from google.cloud.firestore_v1.query import Query, CollectionGroup


class FirestoreLoader(BaseLoader):
    def __init__(
        self,
        source: Query | CollectionGroup | DocumentReference | str,
        page_content_fields: List[str] = None,
        metadata_fields: Optional[List[str]] = None,
        client: Optional[Client] = None,
    ) -> None:
        """Document Loader for Google Cloud Firestore.
        Args:
            source: The source to load the documents. It can be an instance of Query,
                CollectionGroup, DocumentReference or the single `/`-delimited path to
                a Firestore collection.
            page_content_fields: The document field names to write into the `page_content`.
                If an empty or None list is provided all fields will be written into
                `page_content`. When only one field is provided only the value is written.
            metadata_fields: The document field names to write into the `metadata`.
                By default it will write all fields that are not in `page_content` into `metadata`.
            client: Client for interacting with the Google Cloud Firestore API.
        """
        try:
            from google.cloud import firestore
            from google.cloud.firestore_v1.services.firestore.transports.base import (
                DEFAULT_CLIENT_INFO,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        client_info = DEFAULT_CLIENT_INFO
        client_info.user_agent = USER_AGENT
        if client:
            self.client = client
            self.client._user_agent = USER_AGENT
        else:
            self.client = firestore.Client(client_info=client_info)
        self.source = source
        self.page_content_fields = page_content_fields
        self.metadata_fields = metadata_fields

    def load(self) -> List[Document]:
        """Load Documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        try:
            from google.cloud.firestore_v1.document import DocumentReference
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if isinstance(self.source, DocumentReference):
            self.source._client._client_info.user_agent = USER_AGENT
            doc = self._load_document()
            if doc:
                yield self._load_document()
            return
        elif isinstance(self.source, str):
            query = self.client.collection(self.source)
        else:
            query = self.source

        query._client._client_info.user_agent = USER_AGENT

        for document_snapshot in query.stream():
            yield DocumentConverter.convertFirestoreDocument(
                document_snapshot, self.page_content_fields, self.metadata_fields
            )

    def _load_document(self) -> Document:
        doc = self.source.get()
        if doc:
            return DocumentConverter.convertFirestoreDocument(doc)
        else:
            return None


class FirestoreSaver:
    """Write into Google Cloud Platform `Firestore`."""

    def __init__(
        self,
        collection: Optional[str] = None,
        client: Optional[Client] = None,
    ) -> None:
        """Document Saver for Google Cloud Firestore.
        Args:
            collection: The single `/`-delimited path to a Firestore collection. If this
              value is present it will write documents with an auto generated id.
            client: Client for interacting with the Google Cloud Firestore API.
        """
        try:
            from google.cloud import firestore
            from google.cloud.firestore_v1.services.firestore.transports.base import (
                DEFAULT_CLIENT_INFO,
            )
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self.collection = collection
        client_info = DEFAULT_CLIENT_INFO
        client_info.user_agent = USER_AGENT
        if client:
            self.client = client
            self.client._user_agent = USER_AGENT
        else:
            self.client = firestore.Client(client_info=client_info)

    def upsert_documents(
        self,
        documents: List[Document],
        merge: Optional[bool] = False,
        document_ids: Optional[List[str]] = None,
    ) -> None:
        """Create / merge documents into the Firestore database.
        Args:
         documents: List of documents to be written into Firestore.
         merge: To merge data iwth an existing document (creating if the document does
          not exist).
         document_ids: List of document ids to be used. By default it will try to
          construct the document paths using the `reference` from the Document.
        """
        try:
            from google.cloud.firestore_v1.document import DocumentReference
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        db_batch = self.client.batch()

        if document_ids and (len(document_ids) != len(documents)):
            raise ValueError("Document ids and docs must have the same size")

        if document_ids:
            docs_list = tuple(zip(documents, document_ids))
        else:
            docs_list = documents

        for batch in self._batched(docs_list, WRITE_BATCH_SIZE):
            for elem in batch:
                if document_ids:
                    doc = elem[0]
                    doc_id = elem[1]
                else:
                    doc = elem
                    doc_id = None
                document_dict = DocumentConverter.convertLangChainDocument(
                    doc, self.client
                )
                if self.collection:
                    doc_ref = self.client.collection(self.collection).document()
                elif doc_id:
                    doc_ref = DocumentReference(*doc_id.split("/"), client=self.client)
                elif document_dict["path"]:
                    doc_ref = DocumentReference(
                        *document_dict["path"].split("/"), client=self.client
                    )
                else:
                    continue

                db_batch.set(
                    reference=doc_ref, document_data=document_dict["data"], merge=merge
                )
            db_batch.commit()

    def delete_documents(
        self, documents: List[Document], document_ids: Optional[List[str]] = None
    ) -> None:
        """Delete documents from the Firestore database.
        Args:
          documents: List of documents to be deleted from Firestore. It will try to extract
            the {document_path} from the `reference` in the document metadata.
          document_ids: List of documents ids to be deleted from Firestore. If provided
            the `documents` argument will be ignored.

        """
        try:
            from google.cloud.firestore_v1.document import DocumentReference
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        db_batch = self.client.batch()

        if document_ids:
            iter_docs = self._batched(document_ids, WRITE_BATCH_SIZE)
        else:
            iter_docs = self._batched(documents, WRITE_BATCH_SIZE)

        for batch in iter_docs:
            for elem in batch:
                if document_ids:
                    document_path = elem
                else:
                    document_dict = DocumentConverter.convertLangChainDocument(
                        elem, self.client
                    )
                    document_path = document_dict["path"]
                if not document_path:
                    continue
                doc_ref = DocumentReference(
                    *document_path.split("/"), client=self.client
                )
                db_batch.delete(doc_ref)
            db_batch.commit()

    def _batched(self, lst: List[Any], n: int) -> Iterator[Any]:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]
