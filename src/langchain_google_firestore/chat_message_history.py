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

import json
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

from google.cloud import firestore  # type: ignore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict

from .version import __version__

USER_AGENT = "langchain-google-firestore-python:chat_history/" + __version__
DEFAULT_COLLECTION = "ChatHistory"

if TYPE_CHECKING:
    from google.cloud.firestore import Client  # type: ignore


class FirestoreChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        collection: str = DEFAULT_COLLECTION,
        client: Optional[Client] = None,
    ) -> None:
        """Chat Message History for Google Cloud Firestore.
        Args:
            session_id: Arbitrary key that is used to store the messages of a single
                chat session. This is the document_path of a document.
            collection: The single `/`-delimited path to a Firestore collection.
            client: Client for interacting with the Google Cloud Firestore API.
        """
        self.client = client or firestore.Client()
        client_agent = self.client._client_info.user_agent
        if not client_agent:
            self.client._client_info.user_agent = USER_AGENT
        elif USER_AGENT not in client_agent:
            self.client._client_info.user_agent = " ".join([client_agent, USER_AGENT])

        self.session_id = session_id
        self.doc_ref = self.client.collection(collection).document(session_id)
        self.messages: List[BaseMessage] = []
        self._load_messages()

    def _load_messages(self) -> None:
        doc = self.doc_ref.get()
        if doc.exists:
            data_messages = doc.to_dict()
            if "messages" in data_messages:
                self.messages = decode_messages(data_messages["messages"])

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self._upsert_messages()

    def _upsert_messages(self) -> None:
        self.doc_ref.set({"messages": encode_messages(self.messages)})

    def clear(self) -> None:
        self.messages = []
        self.doc_ref.delete()


def encode_messages(messages: List[BaseMessage]) -> List[bytes]:
    return [str.encode(m.json()) for m in messages]


def decode_messages(messages: List[bytes]) -> List[BaseMessage]:
    dict_messages = [json.loads(m.decode()) for m in messages]
    return messages_from_dict([{"type": m["type"], "data": m} for m in dict_messages])
