from langchain_community.vectorstores import FAISS
from openai import OpenAI


def build_retriever(embeddings, chunks_to_analize):
    vs = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    return vs.as_retriever(search_kwargs={"k": chunks_to_analize})


def prompt_with_context(retriever, human_prompt, model_name, client):

    results = retriever.get_relevant_documents(human_prompt)
    context = "\n\n".join([doc.page_content for doc in results])
    query = f"""Responde a la siguiente pregunta usando el contexto y la historia previa de la conversacion.
                Si no estas seguro de que el contexto responde a la pregunta, responde: La informacion
                no se encuentra en el documento
                Pregunta {human_prompt}
                Contexto {context}
                
            """

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Eres un asistente que busca informacion pertinente en actas de reuniones y responde preguntas ",
            },
            {"role": "user", "content": query},
        ],
    )

    return response.choices[0].message.content
