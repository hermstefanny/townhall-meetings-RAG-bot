import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS


def chunk_splitting(doc, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(doc)

    return chunks


def embed_chunks(doc, chunk_size, chunk_overlap, doc_path):
    chunks = chunk_splitting(doc, chunk_size, chunk_overlap)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=api_key
    )
    docs_to_embed = [
        Document(page_content=chunk, metadata={"source": os.path.basename(doc_path)})
        for chunk in chunks
    ]

    vs = FAISS.from_documents(docs_to_embed, embeddings)
    vs.save_local("faiss_index")
    print("FAISS index saved locally.")
    return embeddings
