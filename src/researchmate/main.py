from researchmate.ingestion import load_all_papers
from researchmate.retrieval import (create_bm25_ret,sentiment_search,keyword_search, create_vectoreStore
                                        ,RRF)
chunks=load_all_papers()

print("\n========== FIRST CHUNK ==========")
print(chunks[0].page_content[:500])
print("\nMetadata:")
print(chunks[0].metadata)


vectorstore=create_vectoreStore(chunks)

bm25=create_bm25_ret(chunks)

query='what is Generative Artificial Intelligence'

print("\n========== CHROMADB SEARCH ==========")

dense_result=sentiment_search(vectorstore,query,k=4)

for i,doc in enumerate(dense_result['documents'][0]):
    metadata=dense_result['metadatas'][0][i]

    print("\nPaper:", metadata["paper_name"])
    print("Page:", metadata["page_number"])
    print("Content:", doc[:300])


print("\n========== BM25 SEARCH ==========")

spare_result=keyword_search(bm25,query)

for doc in spare_result:

    print("\nPaper:", doc.metadata["paper_name"])
    print("Page:", doc.metadata["page_number"])
    print("Content:", doc.page_content[:300])



hybrid_results=RRF(dense_result,spare_result)

print("\n========== HYBRID SEARCH RESULTS ==========")
for index, result in enumerate(hybrid_results[:5], start=1):
    print(f"\nRank: {index}")
    print(f"Score: {result['score']:.5f}")
    print(f"Retrieval ranks: {result['ranks']}")
    print(f"Source: {result['metadatas'].get('source')}")
    print(f"Content: {result['page_content'][:300]}")