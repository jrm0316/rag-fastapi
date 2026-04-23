import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_loader2 import carregar_pdf
from llm_groq import responder

# modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# gerar embeddings
def gerar_embeddings(textos):
    return model.encode(textos)

# busca com FAISS
def buscar_similares(query, textos, model, index, k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distancias, indices = index.search(query_embedding, k)

    resultados = []
    for i, idx in enumerate(indices[0]):
        texto = textos[idx]["texto"]
        pagina = textos[idx]["pagina"]
        score = distancias[0][i]

        resultados.append((texto, pagina, score))

    return resultados


if __name__ == "__main__":

    print("Carregando PDF...")
    textos = carregar_pdf("Sistemas.pdf")  # muda se necessário

    print(f"Total de chunks: {len(textos)}")

    # extrair só os textos
    textos_str = [item["texto"] for item in textos]

    print("Gerando embeddings...")
    embeddings = gerar_embeddings(textos_str)
    embeddings = np.array(embeddings).astype("float32")

    print("Criando índice FAISS...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    pergunta = input("\nDigite sua busca: ")

    resultados = buscar_similares(pergunta, textos, model, index, k=15)

    print("\nResultados mais relevantes:\n")
    for texto, pagina, score in resultados:
        preview = texto.strip()

        if len(preview) > 150:
            preview = preview[:150] + "..."

        print(f"[Página {pagina}] {preview} (score: {score:.4f})")

    # montar contexto com página
    top_textos = [
        f"[Página {pagina}] {texto}"
        for texto, pagina, score in resultados
    ]

    contexto = "\n".join(top_textos)

    resposta_llm = responder(pergunta, contexto)

    print("\nResposta da IA:\n")
    print(resposta_llm)
