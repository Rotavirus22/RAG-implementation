from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

PDF_DIR = "data" 
CHROMA_PATH = os.path.join("data", "chroma_db")

def build_index():
    all_chunks = []
    
    for file_name in os.listdir(PDF_DIR):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, file_name)
            print(f" Processing: {file_name}")

            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(pages)

            for chunk in chunks:
                chunk.metadata["source"] = file_name

            all_chunks.extend(chunks)

    print(f" Total chunks created: {len(all_chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    vectordb.persist()

    print(f" Index built for all PDFs in {CHROMA_PATH}")

if __name__ == "__main__":
    build_index()
