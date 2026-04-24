print("VERSÃO FINAL API LEVE")

from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import requests
import os

from llm_groq import responder

app = FastAPI()

print("Carregando índice FAISS...")
index = faiss.read_index("index.faiss")

print("Carregando textos...")
with open("textos.pkl", "rb") as f:
    textos = pickle.load(f)

print("Base carregada com sucesso!")


# 🔹 pega token do ambiente
HF_TOKEN = os.getenv("HF_TOKEN")


# 🔹 função para gerar embedding via API (leve)
def gerar_embedding(texto):
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-MiniLM-L3-v2"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    response = requests.post(url, headers=headers, json={"inputs": texto})

    if response.status_code != 200:
        raise Exception(f"Erro na API HuggingFace: {response.text}")

    embedding = np.array(response.json()).astype("float32")

    # garantir formato correto (1, dim)
    if len(embedding.shape) == 3:
        embedding = embedding.mean(axis=1)

    return embedding


class Pergunta(BaseModel):
    pergunta: str


# 🔹 busca FAISS
def buscar_similares(query, k=3):
    query_embedding = gerar_embedding(query)

    distancias, indices = index.search(query_embedding, k)

    resultados = []
    for i, idx in enumerate(indices[0]):
        resultados.append({
            "texto": textos[idx]["texto"],
            "pagina": textos[idx]["pagina"]
        })

    return resultados


@app.get("/")
def home():
    return {"status": "API rodando"}


@app.post("/perguntar")
def perguntar(dado: Pergunta):
    resultados = buscar_similares(dado.pergunta)

    contexto = "\n".join([
        f"[Página {r['pagina']}] {r['texto']}"
        for r in resultados
    ])

    resposta = responder(dado.pergunta, contexto)

    return {
        "pergunta": dado.pergunta,
        "resposta": resposta,
        "paginas": [r["pagina"] for r in resultados]
    }