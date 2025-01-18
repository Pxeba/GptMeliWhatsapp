import os
from flask import Flask, request, jsonify
import requests
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

app = Flask(__name__)

# Configurações da Z-API
ZAPI_URL = "https://api.z-api.io"
INSTANCE_TOKEN = "<SEU_INSTANCE_TOKEN>"
INSTANCE_ID = "<SEU_INSTANCE_ID>"
CLIENT_TOKEN = "<SEU_CLIENT_TOKEN>"

# Configurações da OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()

# Configurações da API do Mercado Livre
MELI_API_URL = "https://api.mercadolibre.com/"  # URL para buscar os pedidos recentes

# Configuração do Chroma
CHROMA_DB_PATH = "./chroma_db"
vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
from datetime import datetime, timedelta

def fetch_orders(access_token):
    try:
        # Busca os pedidos recentes na API do Mercado Livre com paginação
        headers = {"Authorization": f"Bearer {access_token}"}
        total_orders = 0
        offset = 0
        limit = 50
        orders = []
        url = MELI_API_URL + "orders/search/recent?seller=129223542"

        # Calcula a data de 15 dias atrás
        date_from = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        while True:
            params = {
                "offset": offset,
                "limit": limit,
                "sort": "date_desc",
                "date_from": date_from  # Data ajustada para os últimos 60 dias
            }
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                return jsonify({"error": "Failed to fetch orders", "details": response.text}), response.status_code

            data = response.json()
            results = data.get("results", [])
            orders.extend(results)

            if len(results) < limit:
                break

            offset += limit

        # Processa e salva os pedidos no banco de vetores
        for order in orders:
            order_text = f"Pedido {order['id']} - Cliente: {order['buyer']['nickname']} - Total: {order['total_amount']} - Data: {order['date_created']} - Status: {order['status']} - Itens: {', '.join([item['item']['title'] for item in order['order_items']])}"
            print(order_text)
            vector_store.add_texts([order_text])
            total_orders += 1
        vector_store.persist()
        return jsonify({"status": "Orders fetched and saved successfully", "orders_count": len(orders)}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json

    if not data:
        return jsonify({"error": "Invalid payload"}), 400

    if "text" in data and "message" in data["text"]:
        sender = data.get("phone")
        message_body = data["text"].get("message")

        try:
            # Consulta ao banco de vetores
            search_results = vector_store.similarity_search(message_body, k=10)

            if search_results:
                # Combina todas as informações relevantes encontradas
                relevant_data = " ".join([result.page_content for result in search_results])

                # Usa a IA generativa para criar uma resposta contextualizada
                chat_response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": "Você é um assistente preciso que responde com base nos dados fornecidos."},
                        {"role": "user", "content": f"Baseado nos seguintes dados: '{relevant_data}', responda à pergunta: '{message_body}'"}
                    ]
                )
                # Extrai a resposta do JSON retornado pela IA
                response_message = chat_response.choices[0].message.content
            else:
                response_message = (
                    "Desculpe, não encontrei informações relevantes para sua pergunta. "
                    "Você pode tentar reformular ou fornecer mais detalhes."
                )

        except Exception as e:
            print(f"Erro ao consultar o banco de vetores ou IA: {e}")
            response_message = "Houve um erro ao processar sua mensagem. Por favor, tente novamente."

        # Envia a resposta pelo WhatsApp usando a Z-API
        send_message(sender, response_message)

    return jsonify({"status": "success"}), 200



def send_message(to, message):
    url = f"{ZAPI_URL}/instances/{INSTANCE_ID}/token/{INSTANCE_TOKEN}/send-text"
    payload = {
        "phone": to,
        "message": message
    }
    headers = {
        "Client-Token": CLIENT_TOKEN
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Erro ao enviar mensagem: {response.text}")

if __name__ == "__main__":
    # with app.app_context():
    #     fetch_orders("<SEU_TOKEN_MELI>")
    app.run(debug=True, port=8080)