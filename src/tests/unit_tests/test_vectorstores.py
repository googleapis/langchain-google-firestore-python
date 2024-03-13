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

from google.api_core.client_options import ClientOptions
from google.cloud import firestore
from google.cloud.firestore import CollectionReference  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pytest

from langchain_google_firestore.vectorstores import FirestoreVectorStore


client = firestore.Client()

TEST_COLLECTION = "test_collection"


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


@pytest.fixture(scope="session", autouse=True)
def cleanup_firestore():
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
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, get_embeddings())

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


def test_firestore_update_vectors():
    """
    An end-to-end test for updating vectors in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, get_embeddings())

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
