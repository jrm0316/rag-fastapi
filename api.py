from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_loader2 import carregar_pdf
from llm_groq import responder

app = FastAPI()

# 🔹 carregar modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔹 carregar dados (uma vez só)
print("Carregando PDF...")
textos = carregar_pdf("Sistemas.pdf")
textos_str = [item["texto"] for item in textos]

print("Gerando embeddings...")
embeddings = model.encode(textos_str)
embeddings = np.array(embeddings).astype("float32")

print("Criando índice FAISS...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)


# 🔹 entrada da API
class Pergunta(BaseModel):
    pergunta: str


# 🔹 função de busca
def buscar_similares(query, k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distancias, indices = index.search(query_embedding, k)

    resultados = []
    for i, idx in enumerate(indices[0]):
        texto = textos[idx]["texto"]
        pagina = textos[idx]["pagina"]

        resultados.append({
            "texto": texto,
            "pagina": pagina
        })

    return resultados


# 🔹 endpoint principal
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