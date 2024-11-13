import os

import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_retriever(invalidate: bool):
    # Path for the vectorstore
    persist_directory = "./chroma_db"

    # Check if the vectorstore already exists
    if invalidate is False and os.path.exists(persist_directory):
        # If it exists, load it from disk
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=OpenAIEmbeddings()
        )
        print("Loaded existing vectorstore from disk.")
        return vectorstore.as_retriever()

    # Custom function to find divs with class starting with "ExternalClass"
    def find_external_class_div(name, attrs):
        if name == "div" and "class" in attrs:
            classes = attrs["class"]
            if isinstance(classes, list):
                return any(cls.startswith("ExternalClass") for cls in classes)
            elif isinstance(classes, str):
                return classes.startswith("ExternalClass")
        return False

    # If it doesn't exist, create it
    loader = WebBaseLoader(
        web_paths=(
            "https://www.isbank.com.tr/blog/ters-ibraz-chargeback-nedir-ve-basvurusu-nasil-yapilir",
            "https://www.isbank.com.tr/blog/vadeli-hesap-vadeli-mevduat-hesaplari-nedir",
            "https://www.isbank.com.tr/blog/ihracat-ne-demek",
            "https://www.isbank.com.tr/blog/altin-hesabi-nedir",
            "https://www.isbank.com.tr/blog/forex-nedir",
            "https://www.isbank.com.tr/blog/kumulatif-vergi-matrahi-nedir",
            "https://www.isbank.com.tr/blog/tahvil-ve-bono-nedir",
        ),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(find_external_class_div)),  # type: ignore
        encoding="utf-8",
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )
    print("Created and saved new vectorstore to disk.")
    return vectorstore.as_retriever()


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If the answer does not exist in the context, say that you don't know. "
    "If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise. Answer in turkish, if the answer is unrelated to the context below, apologize "
    "and say that you don't know."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
llm = ChatOpenAI(model="gpt-4o-mini")
retriever = get_retriever(False)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def rag(message, history):
    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content=system_prompt))
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))
    history_langchain_format.append(HumanMessage(content=message))
    answer = ""
    context = {}
    for chunk in rag_chain.stream({"input": message}):
        ctx = chunk.get("context", None)
        if ctx is not None:
            context = ctx
        c = chunk.get("answer", None)
        if c is not None:
            answer = answer + c
            yield answer
    print(answer)

    # Add sources
    if len(context) > 0 and "bilmiyorum" not in answer.lower():
        sources = set()
        for doc in context:
            sources.add(doc.metadata["source"])
        print(f"\nKaynaklar: {', '.join(sources)}\n")
        answer += f"\nKaynaklar: {', '.join(sources)}\n"
        yield answer
