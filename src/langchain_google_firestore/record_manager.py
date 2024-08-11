import datetime
import logging
from google.cloud import firestore
from typing import List, Optional, Sequence, Dict
from langchain_core.indexing import RecordManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FirestoreRecordManager(RecordManager):
    def __init__(
        self,
        namespace: str,
        collection_name: str = "record_manager",
    ) -> None:
        super().__init__(namespace=namespace)
        self.collection_name = collection_name
        self.db = firestore.Client()
        self.collection = self.db.collection(self.collection_name)
        logger.info(f"Initialised FirestoreRecordManager with namespace: {namespace}, collection: {collection_name}")

    def create_schema(self) -> None:
        logger.info("Skipping schema creation (Firestore is schemaless)")
        pass

    async def acreate_schema(self) -> None:
        logger.info("Skipping schema creation (Firestore is schemaless)")
        pass

    def get_time(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)

    async def aget_time(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> Dict[str, int]:
        if group_ids:
            logger.info(f"Updating all {len(keys)} records")
        else:
            logger.info(f"Updating {len(keys)} records")
        if group_ids is None:
            group_ids = [None] * len(keys)

        batch = self.db.batch()
        current_time = self.get_time()
        num_updated = 0
        num_added = 0

        for key, group_id in zip(keys, group_ids):
            doc_ref = self.collection.document(key)
            doc = doc_ref.get()

            if doc.exists:
                num_updated += 1
                if group_id:
                    logger.info(f"Refreshing timestamp for record: {key}")
                else:
                    logger.info(f"Updating existing record: {key}")
            else:
                num_added += 1
                logger.info(f"Adding new record: {key}")

            batch.set(doc_ref, {
                "key": key,
                "namespace": self.namespace,
                "updated_at": current_time,
                "group_id": group_id
            }, merge=True)

        batch.commit()
        logger.info(f"Update complete. Updated: {num_updated}, Added: {num_added}")

        return {
            "num_updated": num_updated,
            "num_added": num_added
        }

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> Dict[str, int]:
        logger.info("Calling synchronous update method")
        return self.update(keys, group_ids=group_ids, time_at_least=time_at_least)

    def exists(self, keys: Sequence[str]) -> List[bool]:
        logger.info(f"Checking existence of {len(keys)} keys")
        result = [False] * len(keys)
        key_to_index = {key: i for i, key in enumerate(keys)}

        # Process keys in batches of 30 for Firestore limit
        for i in range(0, len(keys), 30):
            batch = keys[i:i+30]
            query = self.collection.where(
                filter=firestore.FieldFilter("namespace", "==", self.namespace))
            query = query.where(
                filter=firestore.FieldFilter("key", "in", batch))
            docs = query.get()

            for doc in docs:
                key = doc.get("key")
                if key in key_to_index:
                    result[key_to_index[key]] = True

        logger.info(f"Existence check complete. Found {sum(result)} records")
        return result

    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        logger.info("Calling synchronous exists method")
        return self.exists(keys)

    def list_keys(
        self,
        *,
        before: Optional[datetime.datetime] = None,
        after: Optional[datetime.datetime] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        logger.info("Listing records with filters")

        all_keys = []

        # If there are group_ids, process them in batches of 30 for Firestore limit
        if group_ids:
            for i in range(0, len(group_ids), 30):
                batch_group_ids = group_ids[i:i+30]
                keys = self._list_keys_batch(
                    before, after, batch_group_ids, limit)
                all_keys.extend(keys)
                if limit and len(all_keys) >= limit:
                    all_keys = all_keys[:limit]
                    break
        else:
            all_keys = self._list_keys_batch(before, after, None, limit)

        logger.info(f"Listed {len(all_keys)} records")
        return all_keys

    def _list_keys_batch(
        self,
        before: Optional[datetime.datetime],
        after: Optional[datetime.datetime],
        group_ids: Optional[Sequence[str]],
        limit: Optional[int]
    ) -> List[str]:
        query = self.collection.where(filter=firestore.FieldFilter("namespace", "==", self.namespace))

        if after:
            query = query.where(filter=firestore.FieldFilter("updated_at", ">", after))
            logger.debug(f"Filtering records after: {after}")
        if before:
            query = query.where(filter=firestore.FieldFilter("updated_at", "<", before))
            logger.debug(f"Filtering records before: {before}")
        if group_ids:
            query = query.where(filter=firestore.FieldFilter("group_id", "in", group_ids))
            logger.debug(f"Filtering by group_ids: {group_ids}")

        if limit:
            query = query.limit(limit)
            logger.debug(f"Limiting results to: {limit}")

        docs = query.get()
        keys = [doc.get("key") for doc in docs]
        logger.info(f"Listed {len(keys)} records")
        return keys

    async def alist_keys(
        self,
        *,
        before: Optional[datetime.datetime] = None,
        after: Optional[datetime.datetime] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        logger.info("Calling synchronous list_keys method")
        return self.list_keys(before=before, after=after, group_ids=group_ids, limit=limit)

    def delete_keys(self, keys: Sequence[str]) -> Dict[str, int]:
        logger.info(f"Deleting {len(keys)} records")
        batch = self.db.batch()
        num_deleted = 0

        for key in keys:
            doc_ref = self.collection.document(key)
            doc = doc_ref.get()

            if doc.exists:
                batch.delete(doc_ref)
                num_deleted += 1
                logger.info(f"Deleting record: {key}")

        batch.commit()
        logger.info(f"Deletion complete. Deleted {num_deleted} keys")

        return {"num_deleted": num_deleted}

    async def adelete_keys(self, keys: Sequence[str]) -> Dict[str, int]:
        logger.info("Calling synchronous delete_keys method")
        return self.delete_keys(keys)