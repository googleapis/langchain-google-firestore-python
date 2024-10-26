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

from typing import List

from langchain_core.documents import Document

# from langchain_google_firestore import FirestoreStore
from typing import Any, Iterator, Optional, Sequence, Tuple

import json

from langchain_core.stores import ByteStore


from typing import Union




class FirestoreStore(ByteStore):
    """BaseStore implementation using Google Cloud Firestore as the underlying store.

    Examples:
        Create a FirestoreStore instance and perform operations on it:

        .. code-block:: python

            # Instantiate the FirestoreStore with Firestore client
            from your_module import FirestoreStore
            from google.cloud import firestore

            client = firestore.Client()
            firestore_store = FirestoreStore(client=client, collection_name="kv_store")

            # Set values for keys
            firestore_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = firestore_store.mget(["key1", "key2"])
            # [b"value1", b"value2"]

            # Delete keys
            firestore_store.mdelete(["key1"])

            # Iterate over keys
            for key in firestore_store.yield_keys():
                print(key)  # noqa: T201
    """

    def __init__(
        self,
        *,
        client: Any = None,
        project: Optional[str] = None,
        collection_name: str = "kv_store",
    ) -> None:
        """Initialize the FirestoreStore with a Firestore client.

        Args:
            client: An instance of `google.cloud.firestore.Client`. If not provided,
                one will be created using the provided project ID.
            project: Google Cloud project ID. Required if `client` is not provided.
            collection_name: Name of the Firestore collection to use for storage.
        """
        try:
            from google.cloud import firestore
        except ImportError as e:
            raise ImportError(
                "The FirestoreStore requires the google-cloud-firestore library to be installed. "
                "Install it with `pip install google-cloud-firestore`."
            ) from e

        if client and project:
            raise ValueError(
                "Either a Firestore client or a project ID must be provided, but not both."
            )

        if not client and not project:
            raise ValueError("Either a Firestore client or a project ID must be provided.")

        if client:
            self.client = client
        else:
            self.client = firestore.Client(project=project)

        self.collection_name = collection_name

    def _get_collection(self):
        """Get the Firestore collection for storage.

        Returns:
            CollectionReference: The Firestore collection reference.
        """
        return self.client.collection(self.collection_name)

    def _batch_keys(self, items: Sequence, batch_size: int = 30) -> Iterator[Sequence]:
        """Batch the items into groups of a given batch size."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def mget(self, keys: Sequence[str]) -> List[Optional[Union[bytes, Document]]]:
        """Get the values associated with the given keys, handling Firestore batch limits.

        Args:
            keys: A sequence of keys to retrieve.

        Returns:
            A list of values (either bytes or Document) corresponding to the provided keys.
        """
        collection = self._get_collection()
        result = []
        key_doc_map = {}

        # Batch processing for Firestore queries
        for key_batch in self._batch_keys(keys):
            docs = collection.where('key', 'in', key_batch).stream()
            for doc in docs:
                key_doc_map[doc.get('key')] = doc

        # Collect results and convert from JSON if necessary
        for key in keys:
            doc = key_doc_map.get(key)
            if doc:
                value = doc.get('value')
                if isinstance(value, str):  # Assuming JSON string represents a Document
                    try:
                        value = Document(**json.loads(value))
                    except (json.JSONDecodeError, TypeError):
                        pass  # Handle error if needed, keep original value
                result.append(value)
            else:
                result.append(None)

        return result

    def mset(self, key_value_pairs: Sequence[Tuple[str, Union[bytes, Document]]]) -> None:
        """Set the given key-value pairs in Firestore, handling Firestore batch limits.

        Args:
            key_value_pairs: A sequence of tuples containing key and value (bytes or Document).
        """
        collection = self._get_collection()

        # Firestore has a limit of 500 operations per batch
        for key_value_batch in self._batch_keys(key_value_pairs):
            batch = self.client.batch()

            for key, value in key_value_batch:
                doc_ref = collection.document(key)

                # Convert Document to JSON if needed
                if isinstance(value, Document):
                    value = value.json()  # Convert Document to JSON

                batch.set(doc_ref, {'key': key, 'value': value})

            # Commit each batch
            batch.commit()

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys.

        Args:
            keys: A sequence of keys to delete.
        """
        collection = self._get_collection()
        for key_batch in self._batch_keys(keys):
            batch = self.client.batch()
            for key in key_batch:
                doc_ref = collection.document(key)
                batch.delete(doc_ref)
            batch.commit()

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys in the store.

        Args:
            prefix: Optional prefix to filter keys.

        Yields:
            Keys stored in the Firestore collection.
        """
        collection = self._get_collection()
        if prefix:
            query = collection.where('key', '>=', prefix).where('key', '<', prefix + '\uf8ff')
        else:
            query = collection

        docs = query.stream()
        for doc in docs:
            key = doc.get('key')
            yield key