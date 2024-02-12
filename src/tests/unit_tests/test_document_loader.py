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

import time
import unittest.mock as mock
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreLoader, FirestoreSaver
from google.cloud.firestore import CollectionGroup, FieldFilter


def test_firestore_write_roundtrip_and_load() -> None:
    saver = FirestoreSaver("WriteRoundTrip")
    loader = FirestoreLoader("WriteRoundTrip")

    docs = [Document(page_content="data", metadata={})]

    saver.upsert_documents(docs)
    # wait 1s for consistency
    time.sleep(1)
    written_docs = loader.load()
    saver.delete_documents(written_docs)
    # wait 1s for consistency
    time.sleep(1)

    deleted_docs = loader.load()

    assert len(written_docs) == 1
    assert written_docs[0].page_content == "data"
    assert written_docs[0].metadata != {}
    assert "reference" in written_docs[0].metadata
    assert len(deleted_docs) == 0


def test_firestore_write_load_batch() -> None:
    saver = FirestoreSaver("WriteBatch")
    loader = FirestoreLoader("WriteBatch")
    NUM_DOCS = 1000

    docs = []
    expected_docs = []
    for i in range(NUM_DOCS):
        docs.append(Document(page_content=f"content {i}"))
        expected_docs.append(
            Document(
                page_content=f"content {i}", metadata={"reference": {"path": mock.ANY}}
            )
        )

    saver.upsert_documents(docs)
    # wait 5s for consistency
    time.sleep(5)
    docs_after_write = loader.load()
    saver.delete_documents(docs_after_write)
    # wait 5s for consistency
    time.sleep(5)
    docs_after_delete = loader.load()

    pytest.case.assertCountEqual(expected_docs, docs_after_write)
    assert len(docs_after_delete) == 0


def test_firestore_write_with_reference() -> None:
    saver = FirestoreSaver()
    loader = FirestoreLoader("WriteRef")

    expected_doc = [
        Document(
            page_content="{'f1': 1, 'f2': 2}",
            metadata={"reference": {"path": "WriteRef/doc"}},
        )
    ]
    saver.upsert_documents(expected_doc)
    # wait 1s for consistency
    time.sleep(1)
    written_doc = loader.load()
    saver.delete_documents(written_doc)
    # wait 1s for consistency
    time.sleep(1)
    deleted_doc = loader.load()

    pytest.case.assertCountEqual(expected_doc, written_doc)
    assert len(deleted_doc) == 0


def test_firestore_write_doc_id_error() -> None:
    saver = FirestoreSaver()
    doc_to_insert = [Document(page_content="{'f1': 1, 'f2': 2}")]
    doc_id = ["a/b", "c/d"]

    pytest.case.assertRaises(
        ValueError, saver.upsert_documents, documents=doc_to_insert, document_ids=doc_id
    )


def test_firestore_write_with_doc_id() -> None:
    saver = FirestoreSaver()
    loader = FirestoreLoader("WriteId")

    doc_to_insert = [
        Document(
            page_content="{'f1': 1, 'f2': 2}",
            metadata={"reference": {"path": "foo/bar"}},
        )
    ]

    expected_doc = [
        Document(
            page_content="{'f1': 1, 'f2': 2}",
            metadata={"reference": {"path": "WriteId/doc"}},
        )
    ]
    doc_id = ["WriteId/doc"]
    saver.upsert_documents(documents=doc_to_insert, document_ids=doc_id)
    # wait 1s for consistency
    time.sleep(1)
    written_doc = loader.load()
    saver.delete_documents(written_doc, doc_id)
    # wait 1s for consistency
    time.sleep(1)
    deleted_doc = loader.load()

    pytest.case.assertCountEqual(expected_doc, written_doc)
    assert len(deleted_doc) == 0


@pytest.mark.parametrize(
    "page_fields,metadata_fields,expected_page_content,expected_metadata",
    [
        ([], [], "{'f1': 'v1', 'f2': 'v2', 'f3': 'v3'}", {"reference": mock.ANY}),
        (["f1"], [], "v1", {"reference": mock.ANY, "f2": "v2", "f3": "v3"}),
        ([], ["f2"], "{'f1': 'v1', 'f3': 'v3'}", {"reference": mock.ANY, "f2": "v2"}),
        (["f1"], ["f2"], "v1", {"reference": mock.ANY, "f2": "v2"}),
        (["f2"], ["f2"], "v2", {"reference": mock.ANY, "f2": "v2"}),
    ],
)
def test_firestore_load_with_fields(
    page_fields, metadata_fields, expected_page_content, expected_metadata
):
    saver = FirestoreSaver("WritePageFields")
    loader = FirestoreLoader(
        source="WritePageFields",
        page_content_fields=page_fields,
        metadata_fields=metadata_fields,
    )

    doc_to_insert = [
        Document(page_content="{'f1': 'v1', 'f2': 'v2', 'f3': 'v3'}", metadata={})
    ]
    expected_doc = [
        Document(page_content=expected_page_content, metadata=expected_metadata)
    ]

    saver.upsert_documents(doc_to_insert)
    # wait 1s for consistency
    time.sleep(1)
    loaded_doc = loader.load()
    saver.delete_documents(loaded_doc)
    # wait 1s for consistency
    time.sleep(1)
    deleted_docs = loader.load()

    pytest.case.assertCountEqual(expected_doc, loaded_doc)
    assert len(deleted_docs) == 0


def test_firestore_load_from_subcollection():
    saver = FirestoreSaver()
    loader = FirestoreLoader("collection/doc/subcol")

    doc_to_insert = [
        Document(
            page_content="data",
            metadata={"reference": {"path": "collection/doc/subcol/sdoc"}},
        )
    ]

    saver.upsert_documents(doc_to_insert)
    # wait 1s for consistency
    time.sleep(1)
    loaded_doc = loader.load()
    saver.delete_documents(loaded_doc)
    # wait 1s for consistency
    time.sleep(1)
    deleted_docs = loader.load()

    pytest.case.assertCountEqual(doc_to_insert, loaded_doc)
    assert len(deleted_docs) == 0


def test_firestore_load_from_query():
    saver = FirestoreSaver("WriteQuery")
    loader_cleanup = FirestoreLoader("WriteQuery")

    docs_to_insert = [
        Document(page_content="{'num': 20, 'region': 'west_coast'}"),
        Document(page_content="{'num': 20, 'region': 'south_coast'}"),
        Document(page_content="{'num': 30, 'region': 'west_coast'}"),
        Document(page_content="{'num': 0, 'region': 'east_coast'}"),
    ]
    expected_docs = [
        Document(
            page_content="{'num': 20, 'region': 'west_coast'}",
            metadata={"reference": mock.ANY},
        ),
        Document(
            page_content="{'num': 30, 'region': 'west_coast'}",
            metadata={"reference": mock.ANY},
        ),
    ]

    col_ref = pytest.client.collection("WriteQuery")
    query = col_ref.where(filter=FieldFilter("region", "==", "west_coast"))
    loader = FirestoreLoader(query)

    saver.upsert_documents(docs_to_insert)
    # wait 1s for consistency
    time.sleep(1)
    loaded_docs = loader.load()
    saver.delete_documents(loader_cleanup.load())
    # wait 1s for consistency
    time.sleep(1)
    deleted_docs = loader.load()

    pytest.case.assertCountEqual(expected_docs, loaded_docs)
    assert len(deleted_docs) == 0


def test_firestore_load_from_col_group():
    saver = FirestoreSaver()

    docs_to_insert = [
        Document(
            page_content="data_A",
            metadata={"reference": {"path": "ColA/doc/ColGroup/doc1"}},
        ),
        Document(
            page_content="data_B",
            metadata={"reference": {"path": "ColB/doc/ColGroup/doc2"}},
        ),
        Document(page_content="data_C", metadata={"reference": {"path": "foo/bar"}}),
    ]
    expected_docs = [
        Document(
            page_content="data_A",
            metadata={"reference": {"path": "ColA/doc/ColGroup/doc1"}},
        ),
        Document(
            page_content="data_B",
            metadata={"reference": {"path": "ColB/doc/ColGroup/doc2"}},
        ),
    ]

    saver.upsert_documents(docs_to_insert)
    # wait 1s for consistency
    time.sleep(1)

    col_ref = pytest.client.collection("ColGroup")
    collection_group = CollectionGroup(col_ref)
    loader = FirestoreLoader(collection_group)
    loaded_docs = loader.load()
    saver.delete_documents(docs_to_insert)

    pytest.case.assertCountEqual(expected_docs, loaded_docs)


def test_firestore_load_from_doc_ref():
    saver = FirestoreSaver()

    doc_to_insert = [
        Document(page_content="data", metadata={"reference": {"path": "foo/bar"}})
    ]

    doc_ref = pytest.client.collection("foo").document("bar")

    saver.upsert_documents(doc_to_insert)
    # wait 1s for consistency
    time.sleep(1)
    loader = FirestoreLoader(doc_ref)
    loaded_doc = loader.load()
    saver.delete_documents(doc_to_insert)

    pytest.case.assertCountEqual(doc_to_insert, loaded_doc)


def test_firestore_empty_load():
    loader = FirestoreLoader("Empty")

    loaded_docs = loader.load()

    assert len(loaded_docs) == 0
