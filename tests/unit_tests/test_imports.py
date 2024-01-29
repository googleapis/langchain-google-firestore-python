from langchain_google_firestore import __all__

EXPECTED_ALL = [
    "FirestoreVectorStore",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
