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

import pytest

from langchain_core.documents import Document
from langchain_google_firestore.utility.document_converter import DocumentConverter
from google.cloud import firestore
from google.cloud.firestore_v1.base_document import DocumentSnapshot
from google.cloud.firestore_v1.document import DocumentReference
from google.cloud.firestore_v1._helpers import GeoPoint


@pytest.mark.parametrize(
    "document_snapshot,langchain_doc",
    [
        (
            DocumentSnapshot(
                reference=DocumentReference("foo", "bar"),
                data={"field_1": "data_1", "field_2": 2},
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            ),
            Document(
                page_content="{'field_1': 'data_1', 'field_2': 2}",
                metadata={"reference": {"path": "foo/bar"}},
            ),
        ),
        (
            DocumentSnapshot(
                reference=DocumentReference("foo", "bar"),
                data={
                    "field_1": GeoPoint(1, 2),
                    "field_2": ["data", 2, {"nested": DocumentReference("abc", "xyz")}],
                },
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            ),
            Document(
                page_content="{'field_1': {'latitude': 1, 'longitude': 2}, "
                + "'field_2': ['data', 2, {'nested': {'path': 'abc/xyz'}}]}",
                metadata={"reference": {"path": "foo/bar"}},
            ),
        ),
    ],
)
def test_convert_firestore_document_default_fields(
    document_snapshot, langchain_doc
) -> None:
    return_doc = DocumentConverter.convertFirestoreDocument(document_snapshot)

    assert return_doc == langchain_doc


@pytest.mark.parametrize(
    "document_snapshot,langchain_doc,page_content_fields,metadata_fields",
    [
        (
            DocumentSnapshot(
                reference=DocumentReference("abc", "xyz"),
                data={"data_field": "data", "extra_field": 1},
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            ),
            Document(
                page_content="data",
                metadata={"reference": {"path": "abc/xyz"}, "data_field": "data"},
            ),
            ["data_field"],
            ["data_field"],
        ),
        (
            DocumentSnapshot(
                reference=DocumentReference("abc", "xyz"),
                data={"field_1": 1, "field_2": "val"},
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            ),
            Document(
                page_content="val",
                metadata={"reference": {"path": "abc/xyz"}, "field_1": 1},
            ),
            ["field_2"],
            ["field_1"],
        ),
        (
            DocumentSnapshot(
                reference=DocumentReference("abc", "xyz"),
                data={
                    "field_1": "val_1",
                    "field_2": "val_2",
                    "field_3": "val_3",
                    "field_4": "val_4",
                },
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            ),
            Document(
                page_content="{'field_2': 'val_2', 'field_3': 'val_3'}",
                metadata={"reference": {"path": "abc/xyz"}, "field_1": "val_1"},
            ),
            ["field_2", "field_3"],
            ["field_1"],
        ),
        (
            DocumentSnapshot(
                reference=DocumentReference("abc", "xyz"),
                data={
                    "field_1": "val_1",
                    "field_2": "val_2",
                    "field_3": "val_3",
                    "field_4": "val_4",
                },
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            ),
            Document(
                page_content="{'field_2': 'val_2', 'field_3': 'val_3'}",
                metadata={
                    "reference": {"path": "abc/xyz"},
                    "field_1": "val_1",
                    "field_4": "val_4",
                },
            ),
            [],
            ["field_1", "field_4"],
        ),
        (
            DocumentSnapshot(
                reference=DocumentReference("abc", "xyz"),
                data={
                    "field_1": "val_1",
                    "field_2": "val_2",
                    "field_3": "val_3",
                    "field_4": "val_4",
                },
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            ),
            Document(
                page_content="{'field_2': 'val_2', 'field_4': 'val_4'}",
                metadata={
                    "reference": {"path": "abc/xyz"},
                    "field_1": "val_1",
                    "field_3": "val_3",
                },
            ),
            ["field_2", "field_4"],
            [],
        ),
    ],
)
def test_convert_firestore_document_with_filters(
    document_snapshot, langchain_doc, page_content_fields, metadata_fields
) -> None:
    return_doc = DocumentConverter.convertFirestoreDocument(
        document_snapshot, page_content_fields, metadata_fields
    )

    assert return_doc == langchain_doc


@pytest.mark.parametrize(
    "langchain_doc,firestore_doc",
    [
        (
            Document(page_content="value", metadata={"reference": {"path": "foo/bar"}}),
            {"path": "foo/bar", "data": {"page_content": "value"}},
        ),
        (
            Document(page_content="value", metadata={"reference": {}}),
            {"path": None, "data": {"page_content": "value", "reference": {}}},
        ),
        (
            Document(
                page_content="value",
                metadata={"reference": {"path": "foo/bar", "unexpected_field": "data"}},
            ),
            {
                "path": None,
                "data": {
                    "page_content": "value",
                    "reference": {"path": "foo/bar", "unexpected_field": "data"},
                },
            },
        ),
        (
            Document(
                page_content="value",
                metadata={
                    "reference": {"path": "foo/bar"},
                    "metadata_field": {"path": "abc/xyz"},
                },
            ),
            {
                "path": "foo/bar",
                "data": {
                    "page_content": "value",
                    "metadata_field": DocumentReference(
                        *["abc", "xyz"], client=pytest.client
                    ),
                },
            },
        ),
        (
            Document(
                page_content='{"field_1": "val_1", "field_2": "val_2"}',
                metadata={"reference": {"path": "foo/bar"}, "field_3": "val_3"},
            ),
            {
                "path": "foo/bar",
                "data": {"field_1": "val_1", "field_2": "val_2", "field_3": "val_3"},
            },
        ),
        (
            Document(page_content="", metadata={"reference": {"path": "foo/bar"}}),
            {"path": "foo/bar", "data": {}},
        ),
        (
            Document(
                page_content="",
                metadata={
                    "reference": {"path": "foo/bar"},
                    "point": {"latitude": 1, "longitude": 2},
                    "field_2": "val_2",
                },
            ),
            {"path": "foo/bar", "data": {"point": GeoPoint(1, 2), "field_2": "val_2"}},
        ),
        (Document(page_content="", metadata={}), {"path": None, "data": {}}),
        (
            Document(
                page_content='{"array":[1, "data", {"k_1":"v_1", "k_point": {"latitude":1, "longitude":0}}], "f_2":2}',
                metadata={},
            ),
            {
                "path": None,
                "data": {
                    "array": [1, "data", {"k_1": "v_1", "k_point": GeoPoint(1, 0)}],
                    "f_2": 2,
                },
            },
        ),
    ],
)
def test_convert_langchain_document(langchain_doc, firestore_doc):
    return_doc = DocumentConverter.convertLangChainDocument(
        langchain_doc, pytest.client
    )

    assert return_doc == firestore_doc


@pytest.mark.parametrize(
    "firestore_doc",
    [
        (
            DocumentSnapshot(
                reference=DocumentReference(*["foo", "bar"], client=pytest.client),
                data={
                    "field_1": GeoPoint(1, 2),
                    "field_2": [
                        "data",
                        2,
                        {
                            "nested": DocumentReference(
                                *["abc", "xyz"], client=pytest.client
                            )
                        },
                    ],
                },
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            )
        ),
    ],
)
def test_roundtrip_firestore(firestore_doc):
    langchain_doc = DocumentConverter.convertFirestoreDocument(firestore_doc)
    roundtrip_doc = DocumentConverter.convertLangChainDocument(
        langchain_doc, pytest.client
    )

    assert roundtrip_doc["data"] == firestore_doc.to_dict()
    assert roundtrip_doc["path"] == firestore_doc.reference.path
