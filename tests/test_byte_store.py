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
import sys

import pytest
from google.cloud import firestore  # type: ignore
from langchain_core.documents import Document

from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_google_firestore import FirestoreStore


@pytest.fixture(autouse=True, name="test_case")
def init_test_case() -> TestCase:
    """Returns a TestCase instance."""
    return TestCase()


@pytest.fixture(scope="module", autouse=True, name="embeddings")
def get_embeddings():
    """Returns a FakeEmbeddings instance with a size of 100."""
    return FakeEmbeddings(size=100)


@pytest.fixture(scope="module", autouse=True, name="client")
def firestore_client():
    """Returns a Firestore client."""
    return firestore.Client()


def test_firestore_store_parent_document_retriever(
    test_case: TestCase,
    client: firestore.Client,
    embeddings: FakeEmbeddings,
):
    """
    An end-to-end test for adding vectors to FirestoreVectorStore.
    """
    # Create Firestore Store (byte_store)
    collection_name = "FirestoreParentDocument"
    store = FirestoreStore(
        client=client,
        collection_name=collection_name
    )

    # Create FirestoreVectorStore instance
    vector_store = InMemoryVectorStore(embeddings)

    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1, chunk_overlap=0)

    # Parent document retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
    )

    # Add vectors to Firestore
    texts = ["test_docs_1 - This is a test document to be chunked", "test_doc2 - This is a test document to be chunked"]
    metadatas = [{"foo": "bar"}, {"baz": "qux"}]
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    retriever.add_documents(
        docs,
        ids=None,
    )

    # Retrieve chunk
    retrieved_docs_chunks = vector_store.as_retriever().invoke("test_docs")

    # Retrieve parent document
    retrieved_docs_parents = retriever.invoke("test_docs")

    # Verify that the parents documents were retrieved
    test_case.assertEqual(len(retrieved_docs_chunks), 4)
    test_case.assertEqual(len(retrieved_docs_parents), 2)
    test_case.assertTrue(retrieved_docs_parents[0].page_content in texts)

    # Clean up
    keys = list(store.yield_keys())
    store.mdelete(keys)
    keys = list(store.yield_keys())
    expected_keys = []
    test_case.assertEqual(keys, expected_keys)


def test_firestore_store_workflow(
    test_case: TestCase,
    client: firestore.Client,
):
    collection_name = "FirestoreStoreTestWorkflow"
    store = FirestoreStore(
        client=client,
        collection_name=collection_name
    )

    # Set values for keys
    store.mset([("key1", b"value1"), ("key2", b"value2")])

    # Get values for keys
    values = store.mget(["key1", "key2"])
    expected_values = [b"value1", b"value2"]
    test_case.assertEqual(values, expected_values)

    # Delete a key
    store.mdelete(["key1"])
    values_after_delete = store.mget(["key1", "key2"])
    expected_values_after_delete = [None, b"value2"]
    test_case.assertEqual(values_after_delete, expected_values_after_delete)

    # Iterate over keys
    keys = list(store.yield_keys())
    expected_keys = ["key2"]
    test_case.assertEqual(keys, expected_keys)

    # Clear remaining keys
    store.mdelete(keys)
    values_after_clear = store.mget(["key1", "key2"])
    test_case.assertEqual(values_after_clear, [None, None])


def test_firestore_store_workflow_with_documents(
    test_case: TestCase,
    client: firestore.Client,
):
    collection_name = "FirestoreStoreTestWorkflow"
    store = FirestoreStore(
        client=client,
        collection_name=collection_name
    )

    # Set byte values for keys
    store.mset([("key1", b"value1"), ("key2", b"value2")])

    # Get byte values for keys
    values = store.mget(["key1", "key2"])
    expected_values = [b"value1", b"value2"]
    test_case.assertEqual(values, expected_values)

    # Add Langchain Document objects
    document1 = Document(page_content="Document 1 content", metadata={"author": "Author1"})
    document2 = Document(page_content="Document 2 content", metadata={"author": "Author2"})
    store.mset([("doc_key1", document1), ("doc_key2", document2)])

    # Retrieve Langchain Document objects
    retrieved_docs = store.mget(["doc_key1", "doc_key2"])
    test_case.assertIsInstance(retrieved_docs[0], Document, "Retrieved item is not a Document")
    test_case.assertIsInstance(retrieved_docs[1], Document, "Retrieved item is not a Document")
    test_case.assertEqual(retrieved_docs[0].page_content, "Document 1 content")
    test_case.assertEqual(retrieved_docs[1].page_content, "Document 2 content")
    test_case.assertEqual(retrieved_docs[0].metadata, {"author": "Author1"})
    test_case.assertEqual(retrieved_docs[1].metadata, {"author": "Author2"})

    # Delete a key
    store.mdelete(["key1", "doc_key1"])
    values_after_delete = store.mget(["key1", "key2", "doc_key1", "doc_key2"])
    expected_values_after_delete = [None, b"value2", None, retrieved_docs[1]]
    test_case.assertEqual(values_after_delete, expected_values_after_delete)

    # Iterate over remaining keys
    keys = list(store.yield_keys())
    expected_keys = ["key2", "doc_key2"]
    test_case.assertTrue(expected_keys[0] in keys)
    test_case.assertTrue(expected_keys[1] in keys)
    test_case.assertEqual(len(keys), len(expected_keys))

    # Clear remaining keys
    store.mdelete(keys)
    values_after_clear = store.mget(["key1", "key2", "doc_key1", "doc_key2"])
    test_case.assertEqual(values_after_clear, [None, None, None, None])


def test_firestore_store_custom_client(
    test_case: TestCase,
    client: firestore.Client,
):
    collection_name = "FirestoreStoreTestCustomClient"
    store = FirestoreStore(
        client=client,
        collection_name=collection_name
    )

    # Set a value
    store.mset([("key", b"value")])

    # Get the value
    value = store.mget(["key"])
    test_case.assertEqual(value, [b"value"])

    # Clean up
    keys = list(store.yield_keys())
    store.mdelete(keys)
    keys = list(store.yield_keys())
    expected_keys = []
    test_case.assertEqual(keys, expected_keys)

def test_firestore_store_yield_keys(
    test_case: TestCase,
    client: firestore.Client,
):
    collection_name = "FirestoreStoreTestYieldKeys"
    store = FirestoreStore(
        client=client,
        collection_name=collection_name
    )

    # Set multiple keys
    store.mset([("key1", b"value1"), ("key2", b"value2"), ("key3", b"value3")])

    # Yield all keys
    keys = list(store.yield_keys())
    expected_keys = ["key1", "key2", "key3"]
    test_case.assertCountEqual(keys, expected_keys)

    # Yield keys with prefix
    keys_with_prefix = list(store.yield_keys(prefix="key1"))
    expected_keys_with_prefix = ["key1"]
    test_case.assertEqual(keys_with_prefix, expected_keys_with_prefix)

    # Clean up
    keys = list(store.yield_keys())
    store.mdelete(keys)
    keys = list(store.yield_keys())
    expected_keys = []
    test_case.assertEqual(keys, expected_keys)


def test_firestore_store_large_number_of_keys(
    test_case: TestCase,
    client: firestore.Client,
):
    collection_name = "FirestoreStoreTestLargeKeys"
    store = FirestoreStore(
        client=client,
        collection_name=collection_name
    )

    # Set a large number of keys
    num_keys = 50
    keys = [f"key{i}" for i in range(num_keys)]
    values = [f"value{i}".encode('utf-8') for i in range(num_keys)]
    store.mset(list(zip(keys, values)))

    # Get the keys back
    retrieved_values = store.mget(keys)
    test_case.assertEqual(retrieved_values, values)

    # Clean up
    keys = list(store.yield_keys())
    store.mdelete(keys)
    keys = list(store.yield_keys())
    expected_keys = []
    test_case.assertEqual(keys, expected_keys)
    