from researchmate.ingestion import load_all_papers
from researchmate.retrieval import (create_bm25_ret,sentiment_search,keyword_search, create_vectoreStore
                                        ,RRF,rerank_documets)
chunks=load_all_papers()


vectorstore=create_vectoreStore(chunks)

bm25=create_bm25_ret(chunks)

query='what is Query Rewriting?'

dense_result=sentiment_search(vectorstore,query,k=4)

spare_result=keyword_search(bm25,query)

hybrid_results=RRF(dense_result,spare_result)

final_results=rerank_documets(query,hybrid_results,top_k=5)

for i, doc in enumerate(final_results, start=1):
    print(f"\nRank: {i}")
    print(f"Score: {doc['rerank_score']:.4f}")
    print(f"Source: {doc['metadatas'].get('source')}")
    print(f"Content: {doc['page_content'][:300]}")

