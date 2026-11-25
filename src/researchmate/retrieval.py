
from transformers.pipelines.audio_utils import chunk_bytes_iter
from langchain_text_splitters import sentence_transformers
import chromadb
import hashlib
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.retrievers import BM25Retriever

client=chromadb.PersistentClient(
    path="./chroma_db"
)

embedding_function=SentenceTransformerEmbeddingFunction(
     model_name="all-MiniLM-L6-v2",
    device="cpu"
)


collection= client.get_or_create_collection(
    name='researchmate',
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)


def create_vectoreStore(chunks):
    ids=[]
    documents=[]
    metadatas=[]
    print('total enght of chunks',len(chunks))
    for chunk in chunks:
        unique_text=(
            chunk.metadata['source']+str(chunk.metadata['page_number'])+chunk.page_content
        )
    doc_id=hashlib.sha256(
        unique_text.encode('utf-8')
    ).hexdigest()
    print("Total chunks received:", len(chunks))

    ids.append(doc_id)
    documents.append(chunk.page_content)
    metadatas.append(chunk.metadata)
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print('documet are store in vectore store')
    print('total documents',collection.count())
    return collection

def sentiment_search(collection,query,k=4):
    result=collection.query(
        query_texts=[query],
        n_results=k
    )

    return result

def create_bm25_ret(chunks):
    bm25_retiver=BM25Retriever.from_documents(chunks)

    bm25_retiver.k=4    

    return bm25_retiver


def keyword_search(bm25_retiver,query):
    return bm25_retiver.invoke(query)

