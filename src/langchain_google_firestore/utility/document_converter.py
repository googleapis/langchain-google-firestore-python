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
    List,
)

from langchain_core.documents import Document

IMPORT_ERROR_MSG = (
    "`firestore` package not found, please run `pip3 install google-cloud-firestore`"
)

if TYPE_CHECKING:
    from google.cloud.firestore_v1.base_document import DocumentSnapshot

class DocumentConverter():

    @staticmethod
    def convertFirestoreDocument(document: DocumentSnapshot,
                                 page_content_fields: List[str] = None,
                                 metadata_fields: List[str] = None) -> Document:
        data_doc = document.to_dict()
        metadata = {'reference': document.reference.path}
        if page_content_fields:
            set_page_fields = set(page_content_fields)
        else:
            set_page_fields = set()
        if metadata_fields:
            set_metadata_fields = set(metadata_fields)
        else:
            set_metadata_fields = set()
        shared_keys = set_metadata_fields & set_page_fields

        page_content = {}
        for k in shared_keys:
            if k in data_doc:
                val = DocumentConverter._convertFromFirestore(data_doc.pop(k))
                page_content[k] = val
                metadata[k] = val

        metadata.update({k: DocumentConverter._convertFromFirestore(data_doc.pop(k)) for k in (set_metadata_fields-shared_keys) if k in data_doc})

        if not set_page_fields:
            # write all fields
            page_content = {k: DocumentConverter._convertFromFirestore(data_doc.pop(k)) for k in data_doc.keys()}
        else:
            page_content.update({k: DocumentConverter._convertFromFirestore(data_doc.pop(k)) for k in (set_page_fields-shared_keys) if k in data_doc})

        if not set_metadata_fields:
            # metadata fields not specified. Write remaining fields into metadata
            metadata.update({k: DocumentConverter._convertFromFirestore(v) for k,v in data_doc.items()})

    @staticmethod
    def _convertFromFirestore(val):
        try:
            from google.cloud.firestore_v1.document import DocumentReference
            from google.cloud.firestore_v1._helpers import GeoPoint
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)
        
        if isinstance(val, DocumentReference):
            val_converted = val
        elif isinstance(val, GeoPoint):
            val_converted = val
        elif isinstance(val, dict):
            val_converted = {k: DocumentConverter._convertFromFirestore(v) for k,v in val.items()}
            pass
        elif isinstance(val, list):
            val_converted = [DocumentConverter._convertFromFirestore(v) for v in val]

        return val_converted