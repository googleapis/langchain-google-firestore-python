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

from unittest.mock import Mock

import pytest
from google.cloud import firestore  # type: ignore
from google.cloud.firestore import CollectionReference  # type: ignore
from langchain_community.embeddings import FakeEmbeddings

from langchain_google_firestore.vectorstores import FirestoreVectorStore

TEST_COLLECTION = "test_collection"


def get_embeddings():
    return FakeEmbeddings(size=100)


def get_client():
    return firestore.Client()


@pytest.fixture(scope="session", autouse=True)
def cleanup_firestore():
    client = get_client()
    for doc in client.collection(TEST_COLLECTION).stream():
        doc.reference.delete()


def test_firestore_vectorstore_initialization():
    """
    Tests FirestoreVectorStore initialization with mocked embeddings,
    focusing on correct attribute setting and potential errors.

    This test uses `unittest.mock` and manual patching.
    """

    # Mock Embeddings class and its attributes
    mocked_embeddings = Mock()

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore("my_collection", mocked_embeddings)

    # Assertions to verify attribute values and error handling
    assert isinstance(firestore_store.collection, CollectionReference)
    assert firestore_store.embeddings == mocked_embeddings


def test_firestore_add_vectors():
    """
    An end-to-end test for adding vectors to FirestoreVectorStore.
    """
    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(
        TEST_COLLECTION, get_embeddings(), client=get_client()
    )

    # Add vectors to Firestore
    firestore_store.add_texts(
        ["test_doc1", "test_doc2"],
        metadatas=[{"foo": "bar"}, {"baz": "qux"}],
        ids=["1", "2"],
    )

    # Verify that the vectors were added to Firestore
    docs = firestore_store.collection.stream()
    for doc in docs:
        data = doc.to_dict()
        if doc.id == "1":
            assert data["content"] == "test_doc1"
            assert data["metadata"] == {"foo": "bar"}
        elif doc.id == "2":
            assert data["content"] == "test_doc2"
            assert data["metadata"] == {"baz": "qux"}


def test_firestore_add_vectors_auto_id():
    """
    An end-to-end test for adding vectors to FirestoreVectorStore.
    """
    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(
        TEST_COLLECTION, get_embeddings(), client=get_client()
    )

    # Add vectors to Firestore
    ids = firestore_store.add_texts(
        ["test_doc_1", "test_doc_2"],
    )

    # Verify that the first document was added to Firestore
    # with a generated ID that matches the returned ID
    doc = firestore_store.collection.document(ids[0]).get()
    data = doc.to_dict()
    assert doc.id == ids[0]
    assert data["content"] == "test_doc_1"

    # Verify that the second document was added to Firestore
    # with a generated ID that matches the returned ID
    doc = firestore_store.collection.document(ids[1]).get()
    data = doc.to_dict()
    assert doc.id == ids[1]
    assert data["content"] == "test_doc_2"


def test_firestore_add_vectors_assertions():
    """
    Tests assertions in FirestoreVectorStore.add_vectors method.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(
        TEST_COLLECTION, get_embeddings(), client=get_client()
    )

    # Test assertions
    with pytest.raises(AssertionError):
        firestore_store.add_texts(["test_doc1", "test_doc2"], ids=["1", "2", "3"])

    with pytest.raises(AssertionError):
        firestore_store.add_texts(
            ["test_doc1", "test_doc2"], metadatas=[{"foo": "bar"}]
        )

    with pytest.raises(AssertionError):
        firestore_store.add_texts(["test_doc1", "test_doc2"], ids=["1"])

    with pytest.raises(AssertionError):
        firestore_store.add_texts([])


def test_firestore_update_vectors():
    """
    An end-to-end test for updating vectors in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(
        TEST_COLLECTION, get_embeddings(), client=get_client()
    )

    # Add vectors to Firestore
    firestore_store.add_texts(["test1", "test2"], ids=["1", "2"])

    # Verify that the vectors were updated in Firestore
    docs = firestore_store.collection.stream()
    for doc in docs:
        data = doc.to_dict()
        if doc.id == "1":
            assert data["content"] == "test1"
            assert data["metadata"] == {"foo": "bar"}
        elif doc.id == "2":
            assert data["content"] == "test2"
            assert data["metadata"] == {"baz": "qux"}


def test_firestore_similarity_search():
    """
    An end-to-end test for similarity search in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(
        TEST_COLLECTION, get_embeddings(), client=get_client()
    )

    # Add vectors to Firestore
    firestore_store.add_texts(["test1", "test2"], ids=["1", "2"])

    # Perform similarity search
    results = firestore_store.similarity_search("test1", k=2)

    # Verify that the search results are as expected
    assert len(results) == 2
