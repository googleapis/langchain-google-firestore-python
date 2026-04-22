import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from google.cloud import firestore
from langchain_google_firestore import FirestoreRecordManager

@pytest.fixture(scope="module")
def test_collection():
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    return f"test_record_manager_{python_version}"

@pytest.fixture(scope="module")
def mock_firestore_client():
    with patch('google.cloud.firestore.Client', autospec=True) as mock_client:
        yield mock_client.return_value

@pytest.fixture(autouse=True)
def cleanup_firestore(mock_firestore_client):
    mock_firestore_client.reset_mock()

def test_firestore_record_manager_init(test_collection, mock_firestore_client):
    namespace = "test_namespace"
    record_manager = FirestoreRecordManager(namespace, test_collection)
    
    assert record_manager.namespace == namespace
    assert record_manager.collection_name == test_collection
    assert record_manager.db == mock_firestore_client

def test_firestore_record_manager_update(test_collection, mock_firestore_client):
    namespace = "test_namespace"
    record_manager = FirestoreRecordManager(namespace, test_collection)
    
    mock_doc = MagicMock()
    mock_doc.exists = False
    mock_firestore_client.collection.return_value.document.return_value.get.return_value = mock_doc
    
    keys = ["key1", "key2"]
    group_ids = ["group1", "group2"]
    
    result = record_manager.update(keys, group_ids=group_ids)
    
    assert result["num_added"] == 2
    assert result["num_updated"] == 0
    
    mock_doc.exists = True
    result = record_manager.update(keys, group_ids=group_ids)
    
    assert result["num_added"] == 0
    assert result["num_updated"] == 2

def test_firestore_record_manager_exists(test_collection, mock_firestore_client):
    namespace = "test_namespace"
    record_manager = FirestoreRecordManager(namespace, test_collection)
    
    mock_docs = [
        MagicMock(get=lambda key: "key1" if key == "key" else None),
        MagicMock(get=lambda key: "key2" if key == "key" else None)
    ]
    mock_firestore_client.collection.return_value.where.return_value.where.return_value.get.return_value = mock_docs
    
    keys = ["key1", "key2", "key3"]
    
    result = record_manager.exists(keys)
    
    assert result == [True, True, False]

def test_firestore_record_manager_list_keys(test_collection, mock_firestore_client):
    namespace = "test_namespace"
    record_manager = FirestoreRecordManager(namespace, test_collection)
    
    mock_docs = [
        MagicMock(get=lambda key: "key1" if key == "key" else None),
        MagicMock(get=lambda key: "key2" if key == "key" else None),
        MagicMock(get=lambda key: "key3" if key == "key" else None),
    ]
    
    mock_firestore_client.collection.return_value.where.return_value.get.return_value = mock_docs
    
    result = record_manager.list_keys()
    assert set(result) == {"key1", "key2", "key3"}
    
    mock_firestore_client.collection.return_value.where.return_value.where.return_value.get.return_value = mock_docs[:2]
    
    result = record_manager.list_keys(group_ids=["group1"])
    assert set(result) == {"key1", "key2"}
    
    mock_firestore_client.collection.return_value.where.return_value.limit.return_value.get.return_value = mock_docs[:2]
    
    result = record_manager.list_keys(limit=2)
    assert len(result) == 2

def test_firestore_record_manager_delete_keys(test_collection, mock_firestore_client):
    namespace = "test_namespace"
    record_manager = FirestoreRecordManager(namespace, test_collection)

    mock_doc1 = Mock(exists=True)
    mock_doc2 = Mock(exists=True)
    mock_doc3 = Mock(exists=False)

    mock_collection = mock_firestore_client.collection.return_value
    mock_document_refs = [Mock(), Mock(), Mock()]
    mock_collection.document.side_effect = mock_document_refs

    mock_document_refs[0].get.return_value = mock_doc1
    mock_document_refs[1].get.return_value = mock_doc2
    mock_document_refs[2].get.return_value = mock_doc3

    mock_batch = Mock()
    mock_firestore_client.batch.return_value = mock_batch

    keys = ["key1", "key2", "key3"]
    
    result = record_manager.delete_keys(keys)

    assert mock_batch.delete.call_count == 2
    mock_batch.delete.assert_any_call(mock_document_refs[0])
    mock_batch.delete.assert_any_call(mock_document_refs[1])
    
    assert mock_document_refs[2] not in [call[0][0] for call in mock_batch.delete.call_args_list]

    mock_batch.commit.assert_called_once()

    assert result["num_deleted"] == 2

@pytest.mark.asyncio
async def test_firestore_record_manager_async_methods(test_collection, mock_firestore_client):
    namespace = "test_namespace"
    record_manager = FirestoreRecordManager(namespace, test_collection)
    
    record_manager.aupdate = MagicMock(return_value={"num_added": 2, "num_updated": 0})
    record_manager.aexists = MagicMock(return_value=[True, True, False])
    record_manager.alist_keys = MagicMock(return_value=["key1", "key2"])
    record_manager.adelete_keys = MagicMock(return_value={"num_deleted": 2})
    
    keys = ["key1", "key2"]
    
    result = await record_manager.aupdate(keys)
    assert result["num_added"] == 2
    
    exists_result = await record_manager.aexists(keys + ["key3"])
    assert exists_result == [True, True, False]
    
    list_result = await record_manager.alist_keys()
    assert set(list_result) == set(keys)
    
    delete_result = await record_manager.adelete_keys(keys)
    assert delete_result["num_deleted"] == 2