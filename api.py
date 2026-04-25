print("🚀 API INICIANDO...")

from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from llm_groq import responder

app = FastAPI()

# 🔹 carregar modelo (leve)
print("🤖 Carregando modelo de embedding...")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# 🔹 carregar base pronta
print("📂 Carregando índice FAISS...")
index = faiss.read_index("index.faiss")

print("📂 Carregando textos...")
with open("textos.pkl", "rb") as f:
    textos = pickle.load(f)

print("🔥 Total de textos:", len(textos))
print("✅ Tudo carregado com sucesso!")

# 🔹 entrada da API
class Pergunta(BaseModel):
    pergunta: str


# 🔹 gerar embedding da pergunta (leve)
def gerar_embedding(texto):
    embedding = model.encode([texto])
    return np.array(embedding).astype("float32")


# 🔹 busca no FAISS
def buscar_similares(query, k=3):
    query_embedding = gerar_embedding(query)

    distancias, indices = index.search(query_embedding, k)

    resultados = []
    for idx in indices[0]:
        if idx < len(textos):
            resultados.append({
                "texto": textos[idx]["texto"],
                "pagina": textos[idx]["pagina"]
            })

    return resultados


# 🔹 rota de teste
@app.get("/")
def home():
    return {"status": "API rodando 🚀"}


# 🔹 rota principal
@app.post("/perguntar")
def perguntar(dado: Pergunta):
    try:
        print("📥 Pergunta:", dado.pergunta)

        resultados = buscar_similares(dado.pergunta)

        contexto = "\n".join([
            f"[Página {r['pagina']}] {r['texto']}"
            for r in resultados
        ])

        print("📄 Contexto (resumo):", contexto[:200])

        resposta = responder(dado.pergunta, contexto)

        return {
            "pergunta": dado.pergunta,
            "resposta": resposta,
            "paginas": [r["pagina"] for r in resultados]
        }

    except Exception as e:
        print("❌ ERRO:", e)
        return {"erro": str(e)}