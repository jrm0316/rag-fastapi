print("INICIANDO API...")

from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import requests
import os

from llm_groq import responder

app = FastAPI()

# 🔍 DEBUG: ver arquivos disponíveis
print("Arquivos na pasta:")
print(os.listdir())

# 🔹 carregar base com segurança
try:
    index = faiss.read_index("index.faiss")
    print("index.faiss carregado")
except Exception as e:
    print("ERRO ao carregar index.faiss:", e)
    index = None

try:
    with open("textos.pkl", "rb") as f:
        textos = pickle.load(f)
    print("textos.pkl carregado")
except Exception as e:
    print("ERRO ao carregar textos.pkl:", e)
    textos = []


#  token do HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN")


# 🔹 gerar embedding via API (leve)
def gerar_embedding(texto):
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-MiniLM-L3-v2"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    response = requests.post(url, headers=headers, json={"inputs": texto})

    if response.status_code != 200:
        raise Exception(f"Erro HuggingFace: {response.text}")

    embedding = np.array(response.json()).astype("float32")

    # ajustar formato
    if len(embedding.shape) == 3:
        embedding = embedding.mean(axis=1)

    return embedding


class Pergunta(BaseModel):
    pergunta: str


# 🔹 busca FAISS
def buscar_similares(query, k=3):
    if index is None:
        return []

    query_embedding = gerar_embedding(query)

    distancias, indices = index.search(query_embedding, k)

    resultados = []
    for i, idx in enumerate(indices[0]):
        if idx < len(textos):
            resultados.append({
                "texto": textos[idx]["texto"],
                "pagina": textos[idx]["pagina"]
            })

    return resultados


# 🔹 rota teste
@app.get("/")
def home():
    return {"status": "API rodando"}


# 🔹 rota principal
@app.post("/perguntar")
def perguntar(dado: Pergunta):
    try:
        print("Pergunta:", dado.pergunta)

        resultados = buscar_similares(dado.pergunta)
        print("Resultados:", resultados)

        contexto = "\n".join([
            f"[Página {r['pagina']}] {r['texto']}"
            for r in resultados
        ])

        print("Contexto:", contexto[:200])

        resposta = f"Pergunta recebida: {dado.pergunta}"
        print("Resposta gerada")

        return {
            "pergunta": dado.pergunta,
            "resposta": resposta,
            "paginas": [r["pagina"] for r in resultados]
        }

    except Exception as e:
        print("ERRO:", e)
        return {"erro": str(e)}