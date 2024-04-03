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

from unittest.mock import patch

import pytest
from google.cloud.firestore import (  # type: ignore
    DocumentReference,
    DocumentSnapshot,
    GeoPoint,
)
from google.cloud.firestore_v1.vector import Vector  # type: ignore
from langchain_core.documents import Document

from langchain_google_firestore.document_converter import (
    DOC_REF,
    FIRESTORE_TYPE,
    GEOPOINT,
    VECTOR,
    convert_firestore_document,
    convert_langchain_document,
)


@pytest.fixture(scope="module", autouse=True)
def firestore_client():
    with patch("google.cloud.firestore.Client") as _fixture:
        yield _fixture


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
                page_content='{"field_1": "data_1", "field_2": 2}',
                metadata={
                    "reference": {
                        "path": "foo/bar",
                        FIRESTORE_TYPE: DOC_REF,
                    }
                },
            ),
        ),
        (
            DocumentSnapshot(
                reference=DocumentReference("foo", "bar"),
                data={
                    "field_1": GeoPoint(1, 2),
                    "field_2": [
                        "data",
                        2,
                        {"nested": DocumentReference("abc", "xyz")},
                    ],
                },
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            ),
            Document(
                page_content='{"field_1": {"latitude": 1, "longitude": 2, "firestore_type": "geopoint"}, '
                + '"field_2": ["data", 2, {"nested": {"path": "abc/xyz", "firestore_type": "document_reference"}}]}',
                metadata={
                    "reference": {
                        "path": "foo/bar",
                        FIRESTORE_TYPE: DOC_REF,
                    }
                },
            ),
        ),
    ],
)
def test_convert_firestore_document_default_fields(
    document_snapshot, langchain_doc
) -> None:
    return_doc = convert_firestore_document(document_snapshot)

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
                metadata={
                    "reference": {
                        "path": "abc/xyz",
                        FIRESTORE_TYPE: DOC_REF,
                    },
                    "data_field": "data",
                },
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
                metadata={
                    "reference": {
                        "path": "abc/xyz",
                        FIRESTORE_TYPE: DOC_REF,
                    },
                    "field_1": 1,
                },
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
                page_content='{"field_2": "val_2", "field_3": "val_3"}',
                metadata={
                    "reference": {
                        "path": "abc/xyz",
                        FIRESTORE_TYPE: DOC_REF,
                    },
                    "field_1": "val_1",
                },
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
                page_content='{"field_2": "val_2", "field_3": "val_3"}',
                metadata={
                    "reference": {
                        "path": "abc/xyz",
                        FIRESTORE_TYPE: DOC_REF,
                    },
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
                page_content='{"field_2": "val_2", "field_4": "val_4"}',
                metadata={
                    "reference": {
                        "path": "abc/xyz",
                        FIRESTORE_TYPE: DOC_REF,
                    },
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
    return_doc = convert_firestore_document(
        document_snapshot, page_content_fields, metadata_fields
    )

    assert return_doc == langchain_doc


@pytest.mark.parametrize(
    "langchain_doc,firestore_doc",
    [
        (
            Document(
                page_content="value",
                metadata={
                    "reference": {
                        "path": "foo/bar",
                        FIRESTORE_TYPE: DOC_REF,
                    }
                },
            ),
            {
                "reference": {
                    "path": "foo/bar",
                    FIRESTORE_TYPE: DOC_REF,
                },
                "data": {"page_content": "value"},
            },
        ),
        (
            Document(page_content="value", metadata={"reference": {}}),
            {
                "reference": None,
                "data": {"page_content": "value", "reference": {}},
            },
        ),
        (
            Document(
                page_content="value",
                metadata={"reference": {"path": "foo/bar", "unexpected_field": "data"}},
            ),
            {
                "reference": None,
                "data": {
                    "page_content": "value",
                    "reference": {
                        "path": "foo/bar",
                        "unexpected_field": "data",
                    },
                },
            },
        ),
        (
            Document(
                page_content="value",
                metadata={
                    "reference": {
                        "path": "foo/bar",
                        FIRESTORE_TYPE: DOC_REF,
                    },
                    "metadata_field": {
                        "path": "abc/xyz",
                        FIRESTORE_TYPE: DOC_REF,
                    },
                },
            ),
            {
                "reference": {
                    "path": "foo/bar",
                    FIRESTORE_TYPE: DOC_REF,
                },
                "data": {
                    "page_content": "value",
                    "metadata_field": DocumentReference(
                        *["abc", "xyz"], client=firestore_client
                    ),
                },
            },
        ),
        (
            Document(
                page_content='{"field_1": "val_1", "field_2": "val_2"}',
                metadata={
                    "reference": {
                        "path": "foo/bar",
                        FIRESTORE_TYPE: DOC_REF,
                    },
                    "field_3": "val_3",
                },
            ),
            {
                "reference": {
                    "path": "foo/bar",
                    FIRESTORE_TYPE: DOC_REF,
                },
                "data": {
                    "field_1": "val_1",
                    "field_2": "val_2",
                    "field_3": "val_3",
                },
            },
        ),
        (
            Document(
                page_content="",
                metadata={
                    "reference": {
                        "path": "foo/bar",
                        FIRESTORE_TYPE: DOC_REF,
                    }
                },
            ),
            {
                "reference": {
                    "path": "foo/bar",
                    FIRESTORE_TYPE: DOC_REF,
                },
                "data": {},
            },
        ),
        (
            Document(
                page_content="",
                metadata={
                    "reference": {
                        "path": "foo/bar",
                        FIRESTORE_TYPE: DOC_REF,
                    },
                    "point": {
                        "latitude": 1,
                        "longitude": 2,
                        FIRESTORE_TYPE: GEOPOINT,
                    },
                    "field_2": "val_2",
                },
            ),
            {
                "reference": {
                    "path": "foo/bar",
                    FIRESTORE_TYPE: DOC_REF,
                },
                "data": {"point": GeoPoint(1, 2), "field_2": "val_2"},
            },
        ),
        (
            Document(page_content="", metadata={}),
            {"reference": None, "data": {}},
        ),
        (
            Document(
                page_content='{"array":[1, "data", {"k_1":"v_1", "k_point": {"latitude":1, "longitude":0, "firestore_type": "geopoint"}}], "f_2":2}',
                metadata={},
            ),
            {
                "reference": None,
                "data": {
                    "array": [
                        1,
                        "data",
                        {"k_1": "v_1", "k_point": GeoPoint(1, 0)},
                    ],
                    "f_2": 2,
                },
            },
        ),
    ],
)
def test_convert_langchain_document(langchain_doc, firestore_doc):
    return_doc = convert_langchain_document(langchain_doc, firestore_client)
    assert return_doc == firestore_doc


@pytest.mark.parametrize(
    "firestore_doc",
    [
        (
            DocumentSnapshot(
                reference=DocumentReference(*["foo", "bar"], client=firestore_client),
                data={
                    "field_1": GeoPoint(1, 2),
                    "field_2": [
                        "data",
                        2,
                        {
                            "nested": DocumentReference(
                                *["abc", "xyz"], client=firestore_client
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
    langchain_doc = convert_firestore_document(firestore_doc)
    roundtrip_doc = convert_langchain_document(langchain_doc, firestore_client)

    assert roundtrip_doc["data"] == firestore_doc.to_dict()
    assert roundtrip_doc["reference"]["path"] == firestore_doc.reference.path


@pytest.mark.parametrize(
    "firestore_doc, langchain_doc",
    [
        (
            DocumentSnapshot(
                reference=DocumentReference(*["foo", "bar"], client=firestore_client),
                data={
                    "embedding": Vector([1, 2, 3]),
                    "content": "test_doc2",
                },
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            ),
            Document(
                page_content="test_doc2",
                metadata={
                    "reference": {
                        "path": "foo/bar",
                        FIRESTORE_TYPE: DOC_REF,
                    },
                    "embedding": {
                        "values": [1, 2, 3],
                        FIRESTORE_TYPE: VECTOR,
                    },
                },
            ),
        ),
    ],
)
def test_vector_type_from_firestore(firestore_doc, langchain_doc):
    """
    Test vector type conversion from Firestore to LangChain.
    """

    assert convert_firestore_document(firestore_doc) == langchain_doc


@pytest.mark.parametrize(
    "langchain_doc, firestore_doc",
    [
        (
            Document(
                page_content="test_doc2",
                metadata={
                    "reference": {
                        "path": "foo/bar",
                        FIRESTORE_TYPE: DOC_REF,
                    },
                    "embedding": {
                        "values": [1, 2, 3],
                        FIRESTORE_TYPE: VECTOR,
                    },
                },
            ),
            {
                "reference": {
                    "path": "foo/bar",
                    FIRESTORE_TYPE: DOC_REF,
                },
                "data": {
                    "page_content": "test_doc2",
                    "embedding": Vector([1, 2, 3]),
                },
            },
        ),
    ],
)
def test_vector_type_to_firestore(langchain_doc, firestore_doc):
    """
    Test vector type conversion from LangChain to Firestore.
    """

    assert convert_langchain_document(langchain_doc, firestore_client) == firestore_doc


@pytest.mark.parametrize(
    "firestore_doc",
    [
        (
            DocumentSnapshot(
                reference=DocumentReference(*["foo", "bar"], client=firestore_client),
                data={
                    "field_1": GeoPoint(1, 2),
                    "field_2": [
                        "data",
                        2,
                        {
                            "nested": DocumentReference(
                                *["abc", "xyz"], client=firestore_client
                            )
                        },
                    ],
                    "field_3": Vector([1, 2, 3]),
                },
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None,
            )
        ),
    ],
)
def test_vector_type_roundtrip(firestore_doc):
    """
    Test vector type roundtrip conversion between LangChain and Firestore.
    """

    langchain_doc = convert_firestore_document(firestore_doc)
    roundtrip_doc = convert_langchain_document(langchain_doc, firestore_client)

    assert roundtrip_doc["data"] == firestore_doc.to_dict()
    assert roundtrip_doc["reference"]["path"] == firestore_doc.reference.path
