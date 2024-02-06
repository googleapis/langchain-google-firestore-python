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

if TYPE_CHECKING:
    from google.cloud.firestore_v1.client import Client
    from google.cloud.firestore_v1.document import DocumentReference
    from google.cloud.firestore_v1.query import Query, CollectionGroup

class FirestoreLoader(BaseLoader):
    def __init__(
        self,
        source: Query|CollectionGroup|DocumentReference|str,
        page_content_fields: List[str] = None,
        metadata_fields: Optional[List[str]] = None,
        client: Optional[Client] = None
    ) -> None:
        """Document Loader for Cloud Firestore.
        
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
            return [self._load_document()]
        elif isinstance(self.source, str):
            query = self.client.collection(self.source)
        else:
            query = self.query
    
        query._client._client_info.user_agent = USER_AGENT 
        
        for document_snapshot in query.stream():
            yield DocumentConverter.convertFirestoreDocument(document_snapshot, self._page_content_fields, self._metadata_fields)
    
    def _load_document(self) -> Document:
        return DocumentConverter.convertFirestoreDocument(self.source.get())
    
