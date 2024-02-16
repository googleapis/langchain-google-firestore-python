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

from langchain_google_firestore.vectorstores import FirestoreVectorStore


def test_firestore_vectorstore_initialization():
    """
    Tests FirestoreVectorStore initialization with mocked embeddings,
    focusing on correct attribute setting and potential errors.

    This test uses `unittest.mock` and manual patching.
    """

    # Mock Embeddings class and its attributes
    mocked_embeddings = Mock()
    mocked_embeddings.model = "models/embedding-001"
    mocked_embeddings.google_api_key = "test_api_key"

    # Create FirestoreVectorStore instance
    firestore_store = FirestoreVectorStore("my_collection", mocked_embeddings)

    # Assertions to verify attribute values and error handling
    assert firestore_store.source == "my_collection"
    assert firestore_store.embeddings == mocked_embeddings
