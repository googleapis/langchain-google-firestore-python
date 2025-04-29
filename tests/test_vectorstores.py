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

import sys
from unittest import TestCase

import pytest
from fake import FakeImageEmbeddings
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

from langchain_google_firestore import FirestoreVectorStore
from langchain_google_firestore.document_converter import DOC_REF, VECTOR


@pytest.fixture(scope="module", autouse=True, name="test_collection")
def test_collection_id():
    # Get current Python version
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    return f"test_collection_{python_version}"


@pytest.fixture(autouse=True, name="test_case")
def init_test_case() -> TestCase:
    """Returns a TestCase instance."""
    return TestCase()


@pytest.fixture(scope="module", autouse=True, name="embeddings")
def get_embeddings():
    """Returns a FakeEmbeddings instance with a size of 100."""
    return FakeEmbeddings(size=100)


@pytest.fixture(scope="module", autouse=True, name="image_embeddings")
def get_image_embeddings():
    """Returns a FakeImageEmbeddings instance with a size of 100."""
    return FakeImageEmbeddings(size=1408)


@pytest.fixture(scope="module", autouse=True, name="client")
def firestore_client():
    """Returns a Firestore client."""
    return firestore.Client()


@pytest.fixture(autouse=True)
def cleanup_firestore(
    test_case: TestCase, test_collection: str, client: firestore.Client
):
    """Deletes all documents in the test collection. Will be run before each test."""
    collection = client.collection(test_collection)
    snapshots = collection.list_documents()

    for doc in snapshots:
        doc.delete()

    count = collection.count().get()[0][0]

    test_case.assertEqual(count.value, 0)


def test_firestore_add_vectors(
    test_case: TestCase,
    test_collection: str,
    client: firestore.Client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for adding vectors to FirestoreVectorStore.
    """
    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)

    texts = ["test_doc1", "test_doc2"]
    metadatas = [{"foo": "bar"}, {"baz": "qux"}]

    # Add vectors to Firestore
    firestore_store.add_texts(
        texts,
        metadatas=metadatas,
        ids=["1", "2"],
    )

    # Verify that the vectors were added to Firestore
    docs = firestore_store.collection.stream()
    for doc, text, metadata in zip(docs, texts, metadatas):
        data = doc.to_dict()
        test_case.assertEqual(data["content"], text)
        test_case.assertEqual(data["metadata"], metadata)


def test_firestore_add_vectors_auto_id(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for adding vectors to FirestoreVectorStore.
    """
    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)

    texts = ["test_doc_1", "test_doc_2"]

    # Add vectors to Firestore
    ids = firestore_store.add_texts(texts)

    # Verify that the first document was added to Firestore
    # with a generated ID that matches the returned ID.
    # Order is not guarnteed, so we order the documents by content to match the texts.
    docs = firestore_store.collection.order_by("content").get()

    for doc, _id, text in zip(docs, ids, texts):
        data = doc.to_dict()
        test_case.assertEqual(data["content"], text)
        test_case.assertEqual(doc.id, _id)


def test_firestore_add_vectors_assertions(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    Tests assertions in FirestoreVectorStore.add_vectors method.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)
    texts = ["test_doc1", "test_doc2"]

    # Test assertions
    test_case.assertRaises(
        ValueError,
        firestore_store.add_texts,
        texts,
        ids=["1", "2", "3"],
    )

    test_case.assertRaises(
        ValueError,
        firestore_store.add_texts,
        texts,
        metadatas=[{"foo": "bar"}],
    )

    test_case.assertRaises(
        ValueError,
        firestore_store.add_texts,
        texts,
        ids=["1"],
    )

    test_case.assertRaises(
        ValueError,
        firestore_store.add_texts,
        [],
    )


def test_firestore_add_image_vectors(
    test_case: TestCase,
    test_collection: str,
    client: firestore.Client,
    image_embeddings: FakeImageEmbeddings,
):
    """
    An end-to-end test for adding image vectors to FirestoreVectorStore.
    """
    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(
        test_collection, image_embeddings, client=client
    )

    image_uri1 = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
    image_uri2 = "gs://github-repo/img/vision/google-cloud-next.jpeg"

    image_paths = [image_uri1, image_uri2]
    ids = ["1", "2"]
    metadatas = [{"image_uri": image_uri1}, {"image_uri": image_uri2}]

    # Add vectors to Firestore without storing the base64 encoding
    firestore_store.add_images(image_paths, ids=ids)

    # Verify that the vectors were added to Firestore
    docs = firestore_store.collection.stream()
    for doc, _id, image_path, metadata in zip(docs, ids, image_paths, metadatas):
        data = doc.to_dict()
        test_case.assertEqual(doc.id, _id)
        test_case.assertEqual(data["content"], "")
        test_case.assertEqual(data["metadata"], metadata)


def test_firestore_add_image_vectors_store_encodings_true(
    test_case: TestCase,
    test_collection: str,
    client: firestore.Client,
    image_embeddings: FakeImageEmbeddings,
):
    """
    A test for adding image vectors to FirestoreVectorStore without storing base64 encoding.
    """
    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(
        test_collection, image_embeddings, client=client
    )

    image_uri = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"

    image_paths = [image_uri]
    ids = ["1"]
    metadatas = [{"image_uri": image_uri}]

    # Add vectors to Firestore with base64 encoding stored as content
    firestore_store.add_images([image_uri], ids=ids, store_encodings=True)

    # Verify that the vectors were added to Firestore and the content is empty
    docs = firestore_store.collection.stream()
    for doc, _id, image_path, metadata in zip(docs, ids, image_paths, metadatas):
        data = doc.to_dict()
        test_case.assertEqual(doc.id, _id)
        test_case.assertEqual(data["metadata"], metadata)
        test_case.assertEqual(
            data["content"], firestore_store._encode_image(image_path)
        )


def test_firestore_update_vectors(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for updating vectors in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)
    texts = ["test1", "test2"]
    ids = ["1", "2"]

    # Add vectors to Firestore
    firestore_store.add_texts(texts, ids=ids)

    # Verify that the vectors were updated in Firestore
    docs = firestore_store.collection.stream()
    for doc, text, _id in zip(docs, texts, ids):
        data = doc.to_dict()
        test_case.assertEqual(data["content"], text)
        test_case.assertEqual(doc.id, _id)


def test_firestore_delete(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for deleting vectors in FirestoreVectorStore.
    """

    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)

    texts = ["test1", "test2"]
    ids = ["1", "2"]

    firestore_store.add_texts(texts, ids=ids)

    # Verify that the vectors were added to Firestore
    docs = firestore_store.collection.get()
    test_case.assertEqual(len(docs), 2)

    # Delete vectors from Firestore
    firestore_store.delete(ids)

    # Verify that the vectors were deleted from Firestore
    docs = firestore_store.collection.get()
    test_case.assertEqual(len(docs), 0)


def test_firestore_similarity_search(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for similarity search in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)

    texts = ["test1", "test2"]
    k = 2

    # Add vectors to Firestore
    firestore_store.add_texts(texts, ids=["1", "2"])

    # Perform similarity search
    results = firestore_store.similarity_search("test1", k)

    # Verify that the search results are as expected
    test_case.assertEqual(len(results), k)


def test_firestore_similarity_search_with_filters(
    test_case: TestCase,
    test_collection: str,
    client: firestore.Client,
    embeddings: FakeEmbeddings,
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
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)

    # Add vectors to Firestore
    firestore_store.add_texts(
        ["test1", "test2"],
        ids=["1", "2"],
        metadatas=[{"foo": "bar"}, {"foo": "baz"}],
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
                "reference": {
                    "path": f"{test_collection}/1",
                    "firestore_type": DOC_REF,
                },
                "embedding": {
                    "values": list(query_vector.to_map_value()["value"]),
                    "firestore_type": VECTOR,
                },
                "metadata": {"foo": "bar"},
            },
        ),
    )


def test_firestore_similarity_search_by_vector(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for similarity search in FirestoreVectorStore using a vector query.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)

    k = 2

    # Add vectors to Firestore
    firestore_store.add_texts(["test1", "test2"], ids=["1", "2"])

    # Perform similarity search
    results = firestore_store.similarity_search_by_vector(
        embeddings.embed_query("test1"), k
    )

    # Verify that the search results are as expected
    test_case.assertEqual(len(results), k)


def test_firestore_image_similarity_search(
    test_case: TestCase,
    test_collection: str,
    client,
    image_embeddings: FakeImageEmbeddings,
):
    """
    An end-to-end test for image similarity search in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(
        test_collection, image_embeddings, client=client
    )

    k = 1

    image_name = "googlelogo_color_272x92dp.png"
    image_uri = f"https://www.google.com/images/branding/googlelogo/1x/{image_name}"

    # Add vectors to Firestore
    firestore_store.add_images([image_uri])

    # Perform image similarity search
    results = firestore_store.similarity_search_image(image_uri, k)

    # Verify that the search results are as expected
    test_case.assertEqual(len(results), k)


def test_firestore_max_marginal_relevance(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for max marginal relevance in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)

    k = 2

    # Add vectors to Firestore
    firestore_store.add_texts(
        ["test1", "test2", "test3", "test4", "test5"],
        ids=["1", "2", "3", "4", "5"],
    )

    # Perform max marginal relevance
    results = firestore_store.max_marginal_relevance_search("1", k, fetch_k=5)

    # Verify that the search results are as expected, matching `k`
    test_case.assertEqual(len(results), k)


def test_firestore_max_marginal_relevance_by_vector(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for max marginal relevance in FirestoreVectorStore using a vector query.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)

    k = 3

    # Add vectors to Firestore
    firestore_store.add_texts(
        ["test1", "test2", "test3", "test4", "test5"],
        ids=["1", "2", "3", "4", "5"],
    )

    # Perform max marginal relevance
    results = firestore_store.max_marginal_relevance_search_by_vector(
        embeddings.embed_query("1"), k
    )

    # Verify that the search results are as expected, matching `k`
    test_case.assertEqual(len(results), k)


def test_firestore_similarity_search_with_score(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for similarity search with score in FirestoreVectorStore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)

    texts = ["test1", "test2"]
    k = 2

    # Add vectors to Firestore
    firestore_store.add_texts(texts, ids=["1", "2"])

    # Perform similarity search with score
    results = firestore_store.similarity_search_with_score("test1", k)

    # Verify that the search results are as expected
    test_case.assertEqual(len(results), k)
    
    # Check that each result is a tuple with a Document and a score
    for result in results:
        test_case.assertTrue(isinstance(result, tuple))
        test_case.assertEqual(len(result), 2)
        test_case.assertTrue(isinstance(result[0], Document))
        test_case.assertTrue(isinstance(result[1], float))


def test_firestore_similarity_search_with_score_with_filters(
    test_case: TestCase,
    test_collection: str,
    client: firestore.Client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for similarity search with score in FirestoreVectorStore with filters.
    Requires an index on the filter field in Firestore.
    """

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore(test_collection, embeddings, client=client)

    # Add vectors to Firestore
    firestore_store.add_texts(
        ["test1", "test2"],
        ids=["1", "2"],
        metadatas=[{"foo": "bar"}, {"foo": "baz"}],
    )

    # Perform similarity search with score and filter
    results = firestore_store.similarity_search_with_score(
        "test1", k=2, filters=FieldFilter("metadata.foo", "==", "bar")
    )

    # Verify that the search results are as expected with the filter applied
    test_case.assertEqual(len(results), 1)
    
    # Check that the result is a tuple with a Document and a score
    doc, score = results[0]
    test_case.assertTrue(isinstance(doc, Document))
    test_case.assertTrue(isinstance(score, float))
    test_case.assertEqual(doc.page_content, "test1")
    test_case.assertEqual(doc.metadata["metadata"]["foo"], "bar")


def test_firestore_from_texts(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for initializing FirestoreVectorStore from texts.
    """
    texts = ["test1", "test2"]
    # Add vectors to Firestore
    firestore_store = FirestoreVectorStore.from_texts(
        texts,
        collection=test_collection,
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
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for initializing FirestoreVectorStore from Documents.
    """
    documents = [Document(page_content="test1", metadata={"foo": "bar"})]

    # Add vectors to Firestore
    firestore_store = FirestoreVectorStore.from_documents(
        documents,
        collection=test_collection,
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
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for initializing FirestoreVectorStore from Documents asynchronously.
    """
    documents = [Document(page_content="test1", metadata={"foo": "bar"})]

    # Add vectors to Firestore
    firestore_store = await FirestoreVectorStore.afrom_documents(
        documents,
        collection=test_collection,
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
async def test_firestore_from_texts_async(
    test_case: TestCase,
    test_collection: str,
    client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for initializing FirestoreVectorStore asynchronously.
    """
    # Add vectors to Firestore
    firestore_store = await FirestoreVectorStore.afrom_texts(
        ["test1", "test2"],
        collection=test_collection,
        embedding=embeddings,
        ids=["1", "2"],
        client=client,
    )
    k = 2
    # Perform similarity search
    results = firestore_store.similarity_search("test1", k)

    # Verify that the search results are as expected
    test_case.assertEqual(len(results), k)
