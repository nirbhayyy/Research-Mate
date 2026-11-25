from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_and_split(pdf_path):
    pdf_path=Path(pdf_path)

    loader=PyPDFLoader(str(pdf_path))
    documents=loader.load()

    print(f'loadedd: {pdf_path.name}')
    print(f'total pages : {len(documents)}')

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks=splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata['paper_name']=pdf_path.stem
        chunk.metadata['source']=pdf_path.name

        chunk.metadata["page_number"] = (
            chunk.metadata.get("page", 0) + 1
        )

    print(f'total chuks : {len(chunks)}')

    return chunks


def load_all_papers(folder='data/papers'):
    folder=Path(folder)

    all_chunks=[]

    pdf_file=list(folder.glob('*.pdf'))

    if not pdf_file:
        raise FileNotFoundError('No pdf file in folder data/paper')

    for pdf in pdf_file:
        chunks=load_and_split(pdf)
        all_chunks.extend(chunks)

    print("\n========== INGESTION SUMMARY ==========")
    print(f"Total papers: {len(pdf_file)}")
    print(f"Total chunks: {len(all_chunks)}")

    return all_chunks