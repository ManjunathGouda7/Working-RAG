from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY

def build_rag_chain(llm_model: str, retriever, return_sources: bool = False):
    # Use OpenAI-compatible client pointing to LM Studio
    llm = ChatOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,          # dummy – ignored by LM Studio
        model=llm_model if llm_model else "local-model",  # can be empty or placeholder
        temperature=0.15,
        streaming=False,  # set True later if you want streaming
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful, accurate assistant. Answer using ONLY the provided context.
If the context lacks relevant information, say "I don't have enough information."

Context:
{context}

Question: {question}

Concise, clear answer:""" + 
        ("\n\nAt the end, list sources as [1], [2], etc. with file names." if return_sources else "")
    )

    def format_docs(docs):
        if return_sources:
            parts = []
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "unknown file").split("/")[-1]
                parts.append(f"[{i}] {doc.page_content.strip()} (from {src})")
            return "\n\n".join(parts)
        return "\n\n".join(doc.page_content.strip() for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if return_sources:
        return chain, retriever
    return chain