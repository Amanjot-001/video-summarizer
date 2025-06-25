from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_faiss_index(chunks, save_path="faiss_index"):
    docs = [Document(page_content=chunk) for chunk in chunks]

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    vectordb.save_local(save_path)

    print(f"✅ FAISS index saved to {save_path}")
