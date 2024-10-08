from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google.cloud import firestore
from byte_store import FirestoreStore
import uuid




loaders = [
    TextLoader("C:\\Users\\geoff\\OneDrive\\Documents\\GitHub\\langchain-google-firestore-python\\src\\langchain_google_firestore\\paul_graham_essay.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


client = firestore.Client()
store = FirestoreStore(client=client, collection_name="my_kv_store")

PROJECT_ID = "pcs-sbx-dta-ai"  # @param {type:"string"}
LOCATION = "europe-west1"  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_firestore.document_converter import DOC_REF, VECTOR

# # Initialize the a specific Embeddings Model version
# embeddings = VertexAIEmbeddings(model_name="text-embedding-004")


#     # This text splitter is used to create the child documents
# child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# # The vectorstore to use to index the child chunks
# vectorstore = Chroma(
#     collection_name="full_documents", embedding_function= VertexAIEmbeddings(model_name="text-embedding-004")
# )
# # The storage layer for the parent documents
# store = FirestoreStore(client=client, collection_name="my_kv_store")
# retriever = ParentDocumentRetriever(
#     vectorstore=vectorstore,
#     docstore=store,
#     child_splitter=child_splitter,
# )

# retriever.add_documents(docs, ids=None)

# retrieved_docs = retriever.invoke("justice breyer")

collection_name = "FirestoreStoreTestWorkflow"
namespace = f"WorkflowTest_{uuid.uuid4().hex}"
client = firestore.Client()
store = FirestoreStore(
    client=client,
    collection_name=collection_name,
    namespace=namespace
)

# Create FirestoreVectorStore instance
vector_store = InMemoryVectorStore(FakeEmbeddings(size=100))

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
    ids=["1", "2"],
)

# Retrieve parent document
retrieved_docs_notparent = vector_store.as_retriever().invoke("test_docs")
retrieved_docs = retriever.invoke("test_docs")

print(type(retrieved_docs[0]))
print(retrieved_docs)