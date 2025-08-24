from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
import os

CHROMA_PATH = os.path.join("data", "chroma_db")

def ask_question(query: str):
    # Reload Chroma
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # ✅ Free local LLM (Mistral via Ollama)
    llm = ChatOllama(model="mistral")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke(query)
    print("Answer:", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source"), doc.page_content[:200], "...")

if __name__ == "__main__":
    q = input("Enter your question: ")
    ask_question(q)
