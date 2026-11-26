
import chromadb
import hashlib
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


def RRF(densse_result,sparse_result,k=60):
    fused_result={}

    dense_docs=densse_result['documents'][0]
    dense_metadata=densse_result['metadatas'][0]

    for rank,(content,metadata) in enumerate (zip(dense_docs,dense_metadata),start=1):
        chunk_id=(
            f'{metadata.get('source','')}'
            f'_{metadata.get('page_number','')}'
            f'_{content}'
        )

        if chunk_id not in fused_result:
            fused_result[chunk_id]={
                'page_content':content,
                'metadatas':metadata,
                'score':0.0,
                'ranks':{}
        }
        fused_result[chunk_id]['score']+=1 /(k+rank)
        fused_result[chunk_id]['ranks']['dense']=rank

    # for bm25 result

    for rank,doc in enumerate(sparse_result,start=1):
        content=doc.page_content
        metadata=doc.metadata

        chunk_id=(
            f"{metadata.get('source','')}"
            f"_{metadata.get('page_number','')}"
            f"_{content}"
        )

        if chunk_id not in fused_result:
            fused_result[chunk_id]={
                'page_content':content,
                'metadatas':metadata,
                'score':0.0,
                'ranks':{}
            }
        fused_result[chunk_id]['score']+=1/ (k+rank)
        fused_result[chunk_id]['ranks']['bm25']=rank


    ranked_result=sorted(
        fused_result.values(),
        key=lambda x : x['score'],
        reverse= True
    )

    return ranked_result

