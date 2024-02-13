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

import ast

from typing import (
    TYPE_CHECKING,
    Any,
    List,
)

from langchain_core.documents import Document
from google.cloud.firestore import DocumentReference, GeoPoint

if TYPE_CHECKING:
    from google.cloud.firestore import Client, DocumentSnapshot, DocumentReference


class DocumentConverter:

    @staticmethod
    def convertFirestoreDocument(
        document: DocumentSnapshot,
        page_content_fields: List[str] = None,
        metadata_fields: List[str] = None,
    ) -> Document:
        data_doc = document.to_dict()
        metadata = {"reference": {"path": document.reference.path}}

        set_page_fields = set(page_content_fields or [])
        set_metadata_fields = set(metadata_fields or [])
        shared_keys = set_metadata_fields & set_page_fields

        page_content = {}
        for k in sorted(shared_keys):
            if k in data_doc:
                val = DocumentConverter._convertFromFirestore(data_doc.pop(k))
                page_content[k] = val
                metadata[k] = val

        metadata.update(
            {
                k: DocumentConverter._convertFromFirestore(data_doc.pop(k))
                for k in sorted(set_metadata_fields - shared_keys)
                if k in data_doc
            }
        )

        if not set_page_fields:
            # write all fields
            keys = sorted(data_doc.keys())
            page_content = {
                k: DocumentConverter._convertFromFirestore(data_doc.pop(k))
                for k in keys
            }
        else:
            page_content.update(
                {
                    k: DocumentConverter._convertFromFirestore(data_doc.pop(k))
                    for k in sorted(set_page_fields - shared_keys)
                    if k in data_doc
                }
            )

        if len(page_content) == 1:
            page_content = page_content.popitem()[1]

        if not set_metadata_fields:
            # metadata fields not specified. Write remaining fields into metadata
            metadata.update(
                {
                    k: DocumentConverter._convertFromFirestore(v)
                    for k, v in sorted(data_doc.items())
                }
            )

        return Document(page_content=str(page_content), metadata=metadata)

    @staticmethod
    def convertLangChainDocument(document: Document, client: Client) -> dict:
        metadata = document.metadata
        path = None
        data = {}

        if metadata:
            data.update(DocumentConverter._convertFromLangChain(metadata, client))
        if (
            ("reference" in metadata)
            and ("path" in metadata["reference"])
            and (len(metadata["reference"]) == 1)
        ):
            path = metadata["reference"]["path"]
            data.pop("reference")

        if document.page_content:
            try:
                content_dict = ast.literal_eval(document.page_content)
            except (ValueError, SyntaxError):
                content_dict = {"page_content": document.page_content}
            converted_page = DocumentConverter._convertFromLangChain(
                content_dict, client
            )
            data.update(converted_page)

        return {"path": path, "data": data}

    @staticmethod
    def _convertFromFirestore(val: Any) -> Any:
        val_converted = val
        if isinstance(val, DocumentReference):
            val_converted = {"path": val.path}
        elif isinstance(val, GeoPoint):
            val_converted = {"latitude": val.latitude, "longitude": val.longitude}
        elif isinstance(val, dict):
            val_converted = {
                k: DocumentConverter._convertFromFirestore(v) for k, v in val.items()
            }
        elif isinstance(val, list):
            val_converted = [DocumentConverter._convertFromFirestore(v) for v in val]

        return val_converted

    @staticmethod
    def _convertFromLangChain(val: Any, client: Client) -> Any:
        val_converted = val
        if isinstance(val, dict):
            l = len(val)
            if (l == 1) and ("path" in val) and isinstance(val["path"], str):
                val_converted = DocumentReference(
                    *val["path"].split("/"), client=client
                )
            elif (l == 2) and ("latitude" in val) and ("longitude" in val):
                val_converted = GeoPoint(val["latitude"], val["longitude"])
            else:
                val_converted = {
                    k: DocumentConverter._convertFromLangChain(v, client)
                    for k, v in val.items()
                }
        elif isinstance(val, list):
            val_converted = [
                DocumentConverter._convertFromLangChain(v, client) for v in val
            ]

        return val_converted
