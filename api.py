print("🚀 API INICIANDO...")

from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np

from llm_groq import responder

app = FastAPI()

# 🔥 Lazy loading (resolve timeout no Render)
index = None
textos = None


def carregar_base():
    global index, textos

    if index is None or textos is None:
        print("📂 Carregando base...")

        index = faiss.read_index("index.faiss")

        with open("textos.pkl", "rb") as f:
            textos = pickle.load(f)

        print("🔥 Total de textos:", len(textos))
        print("✅ Base carregada com sucesso!")


# 🔹 modelo de entrada
class Pergunta(BaseModel):
    pergunta: str


# 🔥 EMBEDDING LEVE (igual ao gerar_base.py)
def texto_para_vetor(texto):
    vetor = np.zeros(384, dtype="float32")

    for i, char in enumerate(texto[:384]):
        vetor[i] = ord(char) / 1000

    return vetor.reshape(1, -1)


# 🔥 BUSCA MELHORADA (com filtro)
def buscar_similares(query, k=20):
    query_embedding = texto_para_vetor(query)

    distancias, indices = index.search(query_embedding, k)

    resultados = []

    for i, idx in enumerate(indices[0]):
        if idx < len(textos) and distancias[0][i] < 50:
            resultados.append({
                "texto": textos[idx]["texto"],
                "pagina": textos[idx]["pagina"]
            })

    return resultados[:5]  # só os melhores


# 🔹 rota de teste
@app.get("/")
def home():
    return {"status": "API rodando 🚀"}


# 🔹 rota principal
@app.post("/perguntar")
def perguntar(dado: Pergunta):
    try:
        carregar_base()  # 🔥 carrega só quando precisar

        print("📥 Pergunta:", dado.pergunta)

        resultados = buscar_similares(dado.pergunta)

        # 🔥 CONTEXTO MELHOR FORMATADO
        contexto = "\n\n---\n\n".join([
            f"[Página {r['pagina']}]\n{r['texto']}"
            for r in resultados
        ])

        print("📄 Contexto (resumo):", contexto[:300])

        resposta = responder(dado.pergunta, contexto)

        return {
            "pergunta": dado.pergunta,
            "resposta": resposta,
            "paginas": [r["pagina"] for r in resultados]
        }

    except Exception as e:
        print("❌ ERRO:", e)
        return {"erro": str(e)}