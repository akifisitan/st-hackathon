import gradio as gr
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()


llm = ChatOpenAI(temperature=1.0, model="gpt-4o-mini")


def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content


gr.ChatInterface(predict, type="messages").launch()
