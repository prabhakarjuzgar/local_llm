import os
from functools import cache
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from src.constants import CHROMA_DB


@cache
def init_ollama(model:str="llama3") -> Ollama:
    return Ollama(model=model)


@cache
def get_embed_func() -> Embeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_file_into_db(file_name):
    name = file_name.split("/")[-1]
    name = Path(name).stem
    persist_dir = f"{CHROMA_DB}/{name}"
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=get_embed_func())

    data_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n", " ", ""]
    )
    if file_name.endswith("docx"):
        loader = Docx2txtLoader(file_path=file_name)
    else:
        loader = PyPDFLoader(file_name)

    data = loader.load()

    all_splits = data_splitter.split_documents(data)
    return Chroma.from_documents(
        all_splits,
        get_embed_func(),
        persist_directory=persist_dir
    )


def query_rag(db: Chroma, query: str) -> str:
    results = db.similarity_search_with_relevance_scores(query, k=3)
    if not results or results[0][1] < 0.6:
        # TODO log warning
        print("relevance is less than 0.6")

    response = "\n\n--\n\n".join([doc.page_content for doc, _ in results if doc])

    return response


def do_rag(query: str, context: str, db: Chroma):
    prompt = PromptTemplate.from_template(
        f"""You are a helpful assistant. The user has a question.
            Answer the user question based only on the context: {context}.
            Do not add any statements like Based on the provided context.
            The user question is {query}
        """
    )
    llm = init_ollama()
    chain = (
        {"context": db.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain.invoke(query)


def ask_llm(file_name:str, query: str) -> str:
    db = load_file_into_db(file_name)
    rag_response = query_rag(db, query)

    return do_rag(query, rag_response, db)
