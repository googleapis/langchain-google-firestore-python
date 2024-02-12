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

import json

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
)

USER_AGENT = "LangChain"
DEFAULT_COLLECTION = "ChatHistory"
IMPORT_ERROR_MSG = (
    "`firestore` package not found, please run `pip3 install google-cloud-firestore`"
)

if TYPE_CHECKING:
    from google.cloud.firestore import Client, DocumentReference, Query, CollectionGroup


class FirestoreChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self, session_id: str, collection: str = DEFAULT_COLLECTION, client: Optional[Client] = None
    ) -> None:
        """Chat Message History for Google Cloud Firestore.
        Args:
            session_id: Arbitrary key that is used to store the messages of a single
                chat session. This is the document_path of a document.
            collection: The single `/`-delimited path to a Firestore collection.
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
        self.session_id = session_id
        self.doc_ref = self.client.collection(collection).document(session_id)
        self.messages: List[BaseMessage] = []
        self.load_messages()

    def load_messages(self) -> None:
        doc = self.doc_ref.get()
        if doc.exists:
            encoded_messages = doc.to_dict()["messages"]
            if "messages" in encoded_messages:
                self.messages = self._decode_messages(encoded_messages)

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self.upsert_messages()

    def upsert_messages(self) -> None:
        self.doc_ref.set({"messages": self._encode_messages(self.messages)})

    def clear(self) -> None:
        self.messages = []
        self.doc_ref.delete()

    def _encode_messages(self, messages: List[BaseMessage]) -> List[str]:
        return [str.encode(m.json()) for m in messages]

    def _decode_messages(self, messages: List[str]) -> List[BaseMessage]:
        dict_messages = [json.loads(m) for m in messages]
        return messages_from_dict(
            [{"type": m["type"], "data": m} for m in dict_messages]
        )
