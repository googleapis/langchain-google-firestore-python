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
from typing import TYPE_CHECKING, Any, List

from google.cloud.firestore import DocumentReference, GeoPoint  # type: ignore
from langchain_core.documents import Document

if TYPE_CHECKING:
    from google.cloud.firestore import Client, DocumentReference, DocumentSnapshot

TYPE = "firestore_type"


class DocumentConverter:
    @staticmethod
    def convert_firestore_document(
        document: DocumentSnapshot,
        page_content_fields: List[str] = [],
        metadata_fields: List[str] = [],
    ) -> Document:
        data_doc = document.to_dict()
        metadata = {"reference": {"path": document.reference.path, "type": TYPE}}

        set_page_fields = set(
            page_content_fields or (data_doc.keys() - set(metadata_fields))
        )
        set_metadata_fields = set(
            metadata_fields or (data_doc.keys() - set_page_fields)
        )

        page_content = {}

        metadata.update(
            {
                k: DocumentConverter._convert_from_firestore(data_doc[k])
                for k in sorted(set_metadata_fields)
                if k in data_doc
            }
        )

        page_content.update(
            {
                k: DocumentConverter._convert_from_firestore(data_doc[k])
                for k in sorted(set_page_fields)
                if k in data_doc
            }
        )

        if len(page_content) == 1:
            page_content = page_content.popitem()[1]

        return Document(page_content=str(page_content), metadata=metadata)

    @staticmethod
    def convert_langchain_document(document: Document, client: Client) -> dict:
        metadata = document.metadata
        path = None
        data = {}

        if metadata:
            data.update(DocumentConverter._convert_from_langchain(metadata, client))

        if (
            ("reference" in metadata)
            and ("path" in metadata["reference"])
            and ("type" in metadata["reference"])
            and (metadata["reference"]["type"] == TYPE)
        ):
            path = metadata["reference"]
            data.pop("reference")

        if document.page_content:
            try:
                content_dict = ast.literal_eval(document.page_content)
            except (ValueError, SyntaxError):
                content_dict = {"page_content": document.page_content}
            converted_page = DocumentConverter._convert_from_langchain(
                content_dict, client
            )
            data.update(converted_page)

        return {"reference": path, "data": data}

    @staticmethod
    def _convert_from_firestore(val: Any) -> Any:
        val_converted = val
        if isinstance(val, DocumentReference):
            val_converted = {"path": val.path, "type": TYPE}
        elif isinstance(val, GeoPoint):
            val_converted = {
                "latitude": val.latitude,
                "longitude": val.longitude,
                "type": TYPE,
            }
        elif isinstance(val, dict):
            val_converted = {
                k: DocumentConverter._convert_from_firestore(v) for k, v in val.items()
            }
        elif isinstance(val, list):
            val_converted = [DocumentConverter._convert_from_firestore(v) for v in val]

        return val_converted

    @staticmethod
    def _convert_from_langchain(val: Any, client: Client) -> Any:
        val_converted = val
        if isinstance(val, dict):
            l = len(val)
            if (
                ("path" in val)
                and isinstance(val["path"], str)
                and ("type" in val)
                and (val["type"] == TYPE)
            ):
                val_converted = DocumentReference(
                    *val["path"].split("/"), client=client
                )
            elif (
                ("latitude" in val)
                and ("longitude" in val)
                and ("type" in val)
                and (val["type"] == TYPE)
            ):
                val_converted = GeoPoint(val["latitude"], val["longitude"])
            else:
                val_converted = {
                    k: DocumentConverter._convert_from_langchain(v, client)
                    for k, v in val.items()
                }
        elif isinstance(val, list):
            val_converted = [
                DocumentConverter._convert_from_langchain(v, client) for v in val
            ]

        return val_converted
