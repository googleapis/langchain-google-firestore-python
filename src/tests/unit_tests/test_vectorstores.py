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

from unittest import TestCase
from unittest.mock import Mock

import pytest
from google.cloud import firestore
from google.cloud.firestore import CollectionReference
from google.cloud.firestore_v1 import FieldFilter
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

from langchain_google_firestore.document_converter import DOC_REF, VECTOR
from langchain_google_firestore.vectorstores import FirestoreVectorStore

TEST_COLLECTION = "test_collection"


@pytest.fixture(autouse=True, name="test_case")
def init_test_case() -> TestCase:
    """Returns a TestCase instance."""
    return TestCase()


@pytest.fixture(scope="session", autouse=True, name="embeddings")
def get_embeddings():
    """Returns a FakeEmbeddings instance with a size of 100."""
    return FakeEmbeddings(size=100)


@pytest.fixture(scope="session", autouse=True, name="client")
def firestore_client():
    """Returns a Firestore client."""
    return firestore.Client(
        client_options={"api_endpoint": "test-firestore.sandbox.googleapis.com"}
    )


@pytest.fixture(autouse=True)
def cleanup_firestore(client: firestore.Client):
    """Deletes all documents in the test collection. Will be run before each test."""
    collection = client.collection(TEST_COLLECTION)
    snapshots = collection.list_documents()

    for doc in snapshots:
        doc.delete()

    count = collection.count().get()

    assert count[0][0].value == 0


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


def test_firestore_add_vectors(client, embeddings: FakeEmbeddings):
    """
    An end-to-end test for adding vectors to FirestoreVectorStore.
    """
    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, embeddings, client=client)

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


def test_firestore_add_vectors_auto_id(client, embeddings: FakeEmbeddings):
    """
    An end-to-end test for adding vectors to FirestoreVectorStore.
    """
    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, embeddings, client=client)

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


def test_firestore_add_vectors_assertions(client, embeddings: FakeEmbeddings):
    """
    Tests assertions in FirestoreVectorStore.add_vectors method.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, embeddings, client=client)

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


def test_firestore_update_vectors(client, embeddings: FakeEmbeddings):
    """
    An end-to-end test for updating vectors in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, embeddings, client=client)

    # Add vectors to Firestore
    firestore_store.add_texts(["test1", "test2"], ids=["1", "2"])

    # Verify that the vectors were updated in Firestore
    docs = firestore_store.collection.stream()
    for doc in docs:
        data = doc.to_dict()
        if doc.id == "1":
            assert data["content"] == "test1"
        elif doc.id == "2":
            assert data["content"] == "test2"


def test_firestore_delete(client, embeddings: FakeEmbeddings):
    """
    An end-to-end test for deleting vectors in FirestoreVectorStore.
    """

    firestore_store = FirestoreVectorStore(TEST_COLLECTION, embeddings, client=client)

    firestore_store.add_texts(["test1", "test2"], ids=["1", "2"])

    # Verify that the vectors were added to Firestore
    docs = firestore_store.collection.stream()
    for doc in docs:
        data = doc.to_dict()
        if doc.id == "1":
            assert data["content"] == "test1"
        elif doc.id == "2":
            assert data["content"] == "test2"

    # Delete vectors from Firestore
    firestore_store.delete(["1", "2"])

    # Verify that the vectors were deleted from Firestore
    docs = firestore_store.collection.stream()
    assert len(list(docs)) == 0


def test_firestore_similarity_search(client, embeddings: FakeEmbeddings):
    """
    An end-to-end test for similarity search in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, embeddings, client=client)

    # Add vectors to Firestore
    firestore_store.add_texts(["test1", "test2"], ids=["1", "2"])

    # Perform similarity search
    results = firestore_store.similarity_search("test1", k=2)

    # Verify that the search results are as expected
    assert len(results) == 2


def test_firestore_similarity_search_with_filters(
    test_case: TestCase, client: firestore.Client, embeddings: FakeEmbeddings
):
    """
    An end-to-end test for similarity search in FirestoreVectorStore with filters.
    Requires an index on the filter field in Firestore.

    To create an index, run the following command in the Cloud Shell:
    ```
    gcloud alpha firestore indexes composite create \
        --collection-group=test_collection \
        --query-scope=COLLECTION \
        --field-config=order=ASCENDING,field-path=metadata.foo \
        --field-config=vector-config='{"dimension":"100","flat": "{}"}',field-path=embedding
    ```
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, embeddings, client=client)

    # Add vectors to Firestore
    firestore_store.add_texts(
        ["test1", "test2"], ids=["1", "2"], metadatas=[{"foo": "bar"}, {"foo": "baz"}]
    )

    # Get the vector of the first document for test assertions
    query_vector = firestore_store.collection.document("1").get().to_dict()["embedding"]

    # Perform similarity search
    results = firestore_store.similarity_search(
        "test1", k=2, filters=FieldFilter("metadata.foo", "==", "bar")
    )

    # Verify that the search results are as expected with the filter applied
    test_case.assertEqual(
        results[0],
        Document(
            page_content="test1",
            metadata={
                "reference": {"path": "test_collection/1", "firestore_type": DOC_REF},
                "embedding": {
                    "values": query_vector.value,
                    "firestore_type": VECTOR,
                },
                "metadata": {"foo": "bar"},
            },
        ),
    )


def test_firestore_similarity_search_by_vector(client, embeddings: FakeEmbeddings):
    """
    An end-to-end test for similarity search in FirestoreVectorStore using a vector query.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, embeddings, client=client)

    # Add vectors to Firestore
    firestore_store.add_texts(["test1", "test2"], ids=["1", "2"])

    # Perform similarity search
    results = firestore_store.similarity_search_by_vector(
        embeddings.embed_query("test1"), k=2
    )

    # Verify that the search results are as expected
    assert len(results) == 2


def test_firestore_max_marginal_relevance(client, embeddings: FakeEmbeddings):
    """
    An end-to-end test for max marginal relevance in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, embeddings, client=client)

    # Add vectors to Firestore
    firestore_store.add_texts(
        ["test1", "test2", "test3", "test4", "test5"], ids=["1", "2", "3", "4", "5"]
    )

    # Perform max marginal relevance
    results = firestore_store.max_marginal_relevance_search("1", k=2, fetch_k=5)

    # Verify that the search results are as expected, matching `k`
    assert len(results) == 2


def test_firestore_max_marginal_relevance_by_vector(client, embeddings: FakeEmbeddings):
    """
    An end-to-end test for max marginal relevance in FirestoreVectorStore using a vector query.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(TEST_COLLECTION, embeddings, client=client)

    # Add vectors to Firestore
    firestore_store.add_texts(
        ["test1", "test2", "test3", "test4", "test5"], ids=["1", "2", "3", "4", "5"]
    )

    # Perform max marginal relevance
    results = firestore_store.max_marginal_relevance_search_by_vector(
        embeddings.embed_query("1"), k=3
    )

    # Verify that the search results are as expected, matching `k`
    assert len(results) == 3


def test_firestore_from_texts(test_case: TestCase, client, embeddings: FakeEmbeddings):
    """
    An end-to-end test for initializing FirestoreVectorStore from texts.
    """
    texts = ["test1", "test2"]
    # Add vectors to Firestore
    firestore_store = FirestoreVectorStore.from_texts(
        texts,
        collection=TEST_COLLECTION,
        embedding=embeddings,
        ids=["1", "2"],
        client=client,
    )

    # Assert that the vectors were added to Firestore
    docs = firestore_store.collection.stream()
    for text, doc in zip(texts, docs):
        data = doc.to_dict()
        test_case.assertEqual(data["content"], text)


def test_firestore_from_documents(
    test_case: TestCase, client, embeddings: FakeEmbeddings
):
    """
    An end-to-end test for initializing FirestoreVectorStore from Documents.
    """
    documents = [Document(page_content="test1", metadata={"foo": "bar"})]

    # Add vectors to Firestore
    firestore_store = FirestoreVectorStore.from_documents(
        documents,
        collection=TEST_COLLECTION,
        embedding=embeddings,
        client=client,
        metadata_field="metadata",
        ids=["1"],
    )

    # Assert that the vectors were added to Firestore
    docs = firestore_store.collection.stream()
    for doc, document in zip(docs, documents):
        test_case.assertEqual(doc.to_dict()["content"], document.page_content)
        test_case.assertEqual(doc.to_dict()["metadata"], document.metadata)
        test_case.assertEqual(doc.id, "1")


@pytest.mark.asyncio
async def test_firestore_from_documents_async(
    test_case: TestCase, client, embeddings: FakeEmbeddings
):
    """
    An end-to-end test for initializing FirestoreVectorStore from Documents asynchronously.
    """
    documents = [Document(page_content="test1", metadata={"foo": "bar"})]

    # Add vectors to Firestore
    firestore_store = await FirestoreVectorStore.afrom_documents(
        documents,
        collection=TEST_COLLECTION,
        embedding=embeddings,
        client=client,
        metadata_field="metadata",
    )

    # Assert that the vectors were added to Firestore
    docs = firestore_store.collection.stream()
    for doc, document in zip(docs, documents):
        test_case.assertEqual(doc.to_dict()["content"], document.page_content)
        test_case.assertEqual(doc.to_dict()["metadata"], document.metadata)


@pytest.mark.asyncio
async def test_firestore_from_texts_async(client, embeddings: FakeEmbeddings):
    """
    An end-to-end test for initializing FirestoreVectorStore asynchronously.
    """
    # Add vectors to Firestore
    firestore_store = await FirestoreVectorStore.afrom_texts(
        ["test1", "test2"],
        collection=TEST_COLLECTION,
        embedding=embeddings,
        ids=["1", "2"],
        client=client,
    )

    # Perform similarity search
    results = firestore_store.similarity_search("test1", k=2)

    # Verify that the search results are as expected
    assert len(results) == 2
