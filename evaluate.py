import pandas as pd
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
import os

CHROMA_PATH = os.path.join("data", "chroma_db")
QNA_PATH = "data/eval/qna_data.csv"

def evaluate():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatOllama(model="mistral")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    df = pd.read_csv(QNA_PATH)
    correct = 0

    for _, row in df.iterrows():
        q = row["Question"]
        gold = str(row["Answer"])
        result = qa_chain.invoke(q)["result"]

        if any(tok.lower() in result.lower() for tok in gold.split()[:5]):
            correct += 1

        print(f"Q: {q}\nGold: {gold}\nPred: {result}\n")

    print(f"✅ Coverage: {correct}/{len(df)}")

if __name__ == "__main__":
    evaluate()
