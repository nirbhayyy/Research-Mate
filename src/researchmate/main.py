from researchmate.ingestion import load_all_papers

chunks=load_all_papers()

print("\n========== FIRST CHUNK ==========")
print(chunks[0].page_content[:500])
print("\nMetadata:")
print(chunks[0].metadata)