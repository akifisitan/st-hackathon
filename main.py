import os
import random
import time
from typing import Literal

import gradio as gr
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag import rag, rag_examples

load_dotenv()


with open("data.csv", encoding="utf-8") as f:
    data = f.read()

with open("forecast.csv", encoding="utf-8") as f:
    forecast = f.read()


llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")


system_message = """Sen Türkçe konuşan, kişiselleştirilmiş finansal analiz ve tavsiyeler sunan bir finansal asistansın. 
    Aşağıdaki kullanıcı profili ve harcama verilerine göre yardımcı olacaksın:

    KULLANICI PROFİLİ:
    - Meslek: {meslek}
    - Yaş: {yas}
    - Aylık Net Gelir: ₺{maas:,}
    - Cinsiyet: {cinsiyet}

    SON 3 SENENIN HARCAMA VERISI:
    {harcama_verisi}

    Gelecek 6 ay için Holtz exponential smoothing kullanarak hesaplanan harcama tahmini
    {tahmin}

    ÖNEMLİ NOTLAR:
    1. Tüm finansal tavsiyeleri kişinin profiline ve harcama alışkanlıklarına göre özelleştir
    2. Birikim önerilerinde bulunurken mevcut harcama paternlerini göz önünde bulundur
    3. Enflasyonu ve ekonomik koşulları değerlendirmeye kat
    4. Harcama kategorilerindeki değişimleri analiz et ve trend bazlı önerilerde bulun
    5. Acil durum fonu ve uzun vadeli finansal hedefler için öneriler sun

    Kullanıcının sorularını yanıtlarken:
    - Net ve anlaşılır bir dil kullan
    - Sayısal verileri ₺ sembolü ve binlik ayraçlarla formatla
    - Gerektiğinde kategori bazında detaylı analiz sun
    - Praktik ve uygulanabilir öneriler ver
    - Finansal hedeflere ulaşmak için actionable adımlar öner

    ÖRNEK SORULAR VE ANAHTAR KELİMELER:
    - "Aylık harcama analizi" - Detaylı harcama analizi sun
    - "Birikim önerisi" - Kişiye özel birikim stratejileri öner
    - "Harcama trendi" - Kategori bazında trend analizi yap
    - "Bütçe tavsiyesi" - Kişiselleştirilmiş bütçe önerileri sun
    - "Tasarruf imkanları" - Mevcut harcamalara göre tasarruf fırsatları belirt

    - "X kategorisindeki harcamalarım y TL yi geçtiğinde beni bilgilendir" - X kategorisi için y TL bilgilendirme alarmı kuruldu şeklinde cevap ver, başka detay verme
    - "Gelecek ay için harcama tahmini yap" - Tahmin verisini kullanarak detaylı ve inandırıcı bir açıklama yap, bu bilgiye nasıl ulaştığını da basitçe açıkla
    - "Harcamalarım hakkında bilgi ver" - Verileri kullanarak detaylı bilgi ver, sene bazında değil olayın geneline odaklan
    - "Harcamalarım için bütçe planlaması yap" - Kişiselleştirilmiş bütçe önerileri sun
    """.format(
    meslek="Grafik Dizaynırı",
    yas=25,
    harcama_verisi=data,
    tahmin=forecast,
    maas=35000,
    cinsiyet="Kadın",
)

examples = [
    "Harcamalarım hakkında bilgi ver",
    "Harcamalarım için bütçe planlaması yap",
    "Gelecek ay için harcama tahmini yap",
    "Kişisel bakım harcamalarım 5.000 TL'yi geçtiğinde beni bilgilendir",
]


def predict(message, history):
    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content=system_message))
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))
    history_langchain_format.append(HumanMessage(content=message))

    stream = ""
    for chunk in llm.stream(history_langchain_format):
        stream += str(chunk.content)
        yield stream

    # Simulate streaming
    # response = str(llm.invoke(history_langchain_format).content)
    # print(response)
    # stream = ""
    # chunk_size = 50
    # chunks = [response[i : i + chunk_size] for i in range(0, len(response), chunk_size)]
    # for chunk in chunks:
    #     time.sleep(random.uniform(0.1, 0.15))
    #     stream += chunk
    #     yield stream


def main():
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        raise Exception("OPENAI_API_KEY missing")

    theme = gr.themes.Default(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="blue",
    ).set(body_background_fill="*background_fill_secondary")

    current_assistant: Literal["butce", "okuryazar"] = "butce"
    run = predict if current_assistant == "butce" else rag

    gr.ChatInterface(
        run,
        type="messages",
        title="Parapedia - Bütçe Analizi"
        if current_assistant == "butce"
        else "Parapedia - Finansal Okuryazarlık",
        examples=examples if current_assistant == "butce" else rag_examples,
        theme=theme,
        show_progress="hidden",
        fill_width=True,
    ).launch()


if __name__ == "__main__":
    main()
