import os
from openai import OpenAI
from dotenv import load_dotenv
from text_extraction import record_text_extraction, get_pdf_paths
from embedding import chunk_splitting, embed_chunks
from llm_connection import build_retriever, prompt_with_context


if __name__ == "__main__":
    pdf_paths = get_pdf_paths("raw-pdfs")

    # Testing with One pdf
    test_doc_path = pdf_paths[3]
    print(test_doc_path)

    test_text_chunks = record_text_extraction(test_doc_path)
    embeddings = embed_chunks(test_text_chunks, 1500, 250, test_doc_path)

    retriever = build_retriever(embeddings, 5)

    load_dotenv()
    key_ct = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=key_ct)
    # model = "gpt-5-nano"
    model = "gpt-4o-mini"

    prompt = "Quien suspendio la sesion y porque?"
    chat_answer = prompt_with_context(retriever, prompt, model, client)

    print(chat_answer)
