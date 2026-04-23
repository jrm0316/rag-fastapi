print("VERSÃO NOVA DA API")
print("API INICIANDO...")

from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from llm_groq import responder

print("Imports carregados")

app = FastAPI()

print("Carregando modelo...")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

print("Carregando índice FAISS...")
index = faiss.read_index("index.faiss")

print("Carregando textos...")
with open("textos.pkl", "rb") as f:
    textos = pickle.load(f)

print("Tudo carregado com sucesso!")


class Pergunta(BaseModel):
    pergunta: str


def buscar_similares(query, k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

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