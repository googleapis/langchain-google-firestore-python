from langchain_google_firestore.vectorstores import FirestoreVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    FirestoreVectorStore()
