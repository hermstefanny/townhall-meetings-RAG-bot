import os
from openai import OpenAI
from dotenv import load_dotenv
from text_extraction import record_text_extraction, get_pdf_paths
from embedding import chunk_splitting, embed_chunks
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

if __name__ == "__main__":

    load_dotenv()
    key_ct = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=key_ct)

    pdf_paths = get_pdf_paths("raw-pdfs")

    ## RAG - Tested with one document at the time
    test_doc_path = pdf_paths[3]

    test_text_chunks = record_text_extraction(test_doc_path)
    embeddings = embed_chunks(test_text_chunks, 1500, 250, test_doc_path)

    ### RETRIEVING EMBEDDINGS
    chunks_to_analize = 5
    vs = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vs.as_retriever(search_kwargs={"k": chunks_to_analize})

    ### BUILDING THE MODEL
    model = "gpt-5-nano"
    # model = "gpt-4o-mini"

    llm = ChatOpenAI(
        model=model,
        # temperature=0.5,
        # max_tokens=250,
        timeout=None,
        max_retries=2,
    )

    ### BUILDING PROMPT TEMPLATE
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Responde a la siguiente pregunta usando el contexto y la historia previa de la conversacion.
                Si no estas seguro de que el contexto responde a la pregunta, responde: La informacion
                no se encuentra en el documento
                Contexto {context}
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ],
    )

    ### CREATING CHAT MEMORY
    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="input", return_messages=True
    )

    ### CREATING CHAT OBJECT
    chat = LLMChain(llm=llm, prompt=prompt, memory=memory)

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        print("You:", user_input)
        results = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in results])

        response = chat.invoke({"input": user_input, "context": context})
        print("Bot:", response["text"])
