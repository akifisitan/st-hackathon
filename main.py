import os

import gradio as gr
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from functions import bildirim_gonder, estimate_future_expenses

load_dotenv()

expenses_data = [
    "Ay,Market,Yeme ve İçme,Sağlık ve Kişisel Bakım,Giyim ve Aksesuar,Fatura ve Tekrar Eden Ödemeler,Benzin ve Akaryakıt,Diğer,Ev Eşyaları,Toplam Harcama",
    "1,6500,4153,1200,2500,7000,2000,1500,0,24853",
    "2,6700,3335,1100,2700,7000,2060,1550,0,24445",
    "3,6900,3263,1300,2600,7000,2120,1600,0,24783",
    "4,7100,4231,1200,2800,7000,2180,1650,0,26161",
    "5,7300,3533,1400,2500,7000,2240,1700,0,25673",
    "6,7500,2777,1300,2900,7000,2300,1750,0,25527",
    "7,7700,4278,1500,2600,7000,2360,1800,0,27238",
    "8,7900,4328,1400,2800,7000,2420,1850,0,27698",
    "9,8100,4862,1500,2600,7000,2480,1900,0,28442",
]

user_data = {"maas": 30000, "yas": 30, "meslek": "ogretmen", "cinsiyet": "kadin"}

system_prompt = """
Sen finans , ekonomi, banka, yatırım, harcama yönetimi gibi konularda bilgi veren bir botsun. 
Bu konular dışında hiç bir soruya cevap verme.

Kullanıcı verisi : 
"Aylık harcama" : 
"Ay,Market,Yeme ve İçme,Sağlık ve Kişisel Bakım,Giyim ve Aksesuar,Fatura ve Tekrar Eden Ödemeler,Benzin ve Akaryakıt,Diğer,Ev Eşyaları,Toplam Harcama",
    "1,6500,4153,1200,2500,7000,2000,1500,0,24853",
    "2,6700,3335,1100,2700,7000,2060,1550,0,24445",
    "3,6900,3263,1300,2600,7000,2120,1600,0,24783",
    "4,7100,4231,1200,2800,7000,2180,1650,0,26161",
    "5,7300,3533,1400,2500,7000,2240,1700,0,25673",
    "6,7500,2777,1300,2900,7000,2300,1750,0,25527",
    "7,7700,4278,1500,2600,7000,2360,1800,0,27238",
    "8,7900,4328,1400,2800,7000,2420,1850,0,27698",
    "9,8100,4862,1500,2600,7000,2480,1900,0,28442"

"Maaş": 30000,
"Yaş": 30,
"Meslek": "Öğretmen",
"Cinsiyet": "Kadın"

"""


llm = ChatOpenAI(temperature=0.4, model="gpt-4o-mini")


def predict(message, history):
    # Send an initial message if the chat history is empty
    if not history:
        return """Merhaba, ben senin harcama danışmanın ParaPedia. 
        Bana bu tarz soruları sorabilirsin: \n 
        1. Gelecek aylar için harcamalarımı tahmin et \n 
        2. Harcamalarıma göre bütçe planlaması yap\n 
        3. Geçmiş harcamalarımı kategorilere ayır \n 
        4. Eğitim kategorisindeki harcamalarım 10.000 TL'yi geçtiğinde bana haber ver \n 
        5. Aylık ortalama gelirim nedir?"""

    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content=system_prompt))
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm.invoke(history_langchain_format)

    # Check if the message is about saving money
    if "Gelecek aylar için harcamalarımı tahmin et" in message:
        return estimate_future_expenses(user_data, expenses_data)
    elif "bana haber ver" in message:
        return bildirim_gonder(user_data, expenses_data)
    return gpt_response.content


def main():
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        raise Exception("OPENAI_API_KEY missing")

    theme = gr.themes.Default(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="blue",
    ).set(body_background_fill="*background_fill_secondary")

    gr.ChatInterface(predict, type="messages", title="Parapedia", theme=theme).launch()


if __name__ == "__main__":
    main()
