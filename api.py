print("API INICIANDO...")

from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np

from llm_groq import responder

app = FastAPI()

# 🔹 carregar base pronta
print("Carregando index...")
index = faiss.read_index("index.faiss")

print("Carregando textos...")
with open("textos.pkl", "rb") as f:
    textos = pickle.load(f)
print("Total de textos:", len(textos))

print("Tudo carregado!")

class Pergunta(BaseModel):
    pergunta: str


#  EMBEDDING SIMPLES (sem modelo pesado)
def texto_para_vetor(texto):
    vetor = np.zeros(384, dtype="float32")

    for i, char in enumerate(texto[:384]):
        vetor[i] = ord(char) / 1000

    return vetor.reshape(1, -1)


def buscar_similares(query, k=3):
    query_embedding = texto_para_vetor(query)

    distancias, indices = index.search(query_embedding, k)

    resultados = []
    for idx in indices[0]:
        if idx < len(textos):
            resultados.append({
                "texto": textos[idx]["texto"],
                "pagina": textos[idx]["pagina"]
            })

    return resultados


@app.get("/")
def home():
    return {"status": "API rodando 🚀"}


@app.post("/perguntar")
def perguntar(dado: Pergunta):
    try:
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

    except Exception as e:
        return {"erro": str(e)}