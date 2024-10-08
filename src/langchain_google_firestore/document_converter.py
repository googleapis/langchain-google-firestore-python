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
from typing import TYPE_CHECKING, Any, List, Optional

from google.cloud.firestore import DocumentReference, GeoPoint  # type: ignore
from google.cloud.firestore_v1.vector import Vector  # type: ignore
from langchain_core.documents import Document

if TYPE_CHECKING:
    from google.cloud.firestore import Client, DocumentSnapshot


FIRESTORE_TYPE = "firestore_type"
DOC_REF = "document_reference"
GEOPOINT = "geopoint"
VECTOR = "vector"


def convert_firestore_document(
    document: DocumentSnapshot,
    page_content_fields: Optional[List[str]] = None,
    metadata_fields: Optional[List[str]] = None,
) -> Document:
    data_doc = document.to_dict()
    metadata = {
        "reference": {
            "path": document.reference.path,
            FIRESTORE_TYPE: DOC_REF,
        }
    }

    # Check for vector fields and move them from the data_doc to the metadata
    vector_keys = [k for k in data_doc if isinstance(data_doc[k], Vector)]
    for k in vector_keys:
        metadata[k] = _convert_from_firestore(data_doc.pop(k))

    set_page_fields = set(
        page_content_fields or (data_doc.keys() - set(metadata_fields or []))
    )
    set_metadata_fields = set(metadata_fields or (data_doc.keys() - set_page_fields))

    page_content = {}

    for k in sorted(set_metadata_fields):
        if k in data_doc:
            metadata[k] = _convert_from_firestore(data_doc[k])
    for k in sorted(set_page_fields):
        if k in data_doc:
            page_content[k] = _convert_from_firestore(data_doc[k])

    if len(page_content) == 1:
        page_content = str(page_content.popitem()[1])  # type: ignore
    else:
        page_content = json.dumps(page_content)  # type: ignore

    return Document(page_content=page_content, metadata=metadata)  # type: ignore


def convert_langchain_document(document: Document, client: Client) -> dict:
    metadata = document.metadata
    path = None
    data = {}

    if metadata:
        data.update(_convert_from_langchain(metadata, client))

    if metadata.get("reference", {}).get(FIRESTORE_TYPE) == DOC_REF:
        path = metadata["reference"]
        data.pop("reference")

    if document.page_content:
        try:
            content_dict = json.loads(document.page_content)
        except (ValueError, SyntaxError):
            content_dict = {"page_content": document.page_content}
        converted_page = _convert_from_langchain(content_dict, client)
        data.update(converted_page)

    return {"reference": path, "data": data}


def _convert_from_firestore(val: Any) -> Any:
    val_converted = val
    if isinstance(val, dict):
        val_converted = {k: _convert_from_firestore(v) for k, v in val.items()}
    elif isinstance(val, list):
        val_converted = [_convert_from_firestore(v) for v in val]
    elif isinstance(val, DocumentReference):
        val_converted = {
            "path": val.path,
            FIRESTORE_TYPE: DOC_REF,
        }
    elif isinstance(val, GeoPoint):
        val_converted = {
            "latitude": val.latitude,
            "longitude": val.longitude,
            FIRESTORE_TYPE: GEOPOINT,
        }
    elif isinstance(val, Vector):
        vector_map = val.to_map_value()
        val_converted = {
            "values": list(vector_map["value"]),
            FIRESTORE_TYPE: VECTOR,
        }

    return val_converted


def _convert_from_langchain(val: Any, client: Client) -> Any:
    val_converted = val
    if isinstance(val, list):
        val_converted = [_convert_from_langchain(v, client) for v in val]
    elif isinstance(val, dict):
        if val.get(FIRESTORE_TYPE) == DOC_REF:
            val_converted = DocumentReference(*val["path"].split("/"), client=client)
        elif val.get(FIRESTORE_TYPE) == GEOPOINT:
            val_converted = GeoPoint(val["latitude"], val["longitude"])
        elif val.get(FIRESTORE_TYPE) == VECTOR:
            val_converted = Vector(val["values"])
        else:
            val_converted = {
                k: _convert_from_langchain(v, client) for k, v in val.items()
            }
    return val_converted
