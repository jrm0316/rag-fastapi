import faiss
import numpy as np
import pickle

from pdf_loader2 import carregar_pdf

print("Carregando PDF...")
textos = carregar_pdf("Sistemas.pdf")

textos_str = [item["texto"] for item in textos]


# EMBEDDING LEVE (SEM MODELO)
def texto_para_vetor(texto):
    vetor = np.zeros(384, dtype="float32")

    for i, char in enumerate(texto[:384]):
        vetor[i] = ord(char) / 1000

    return vetor


print("Gerando embeddings (leve)...")
embeddings = [texto_para_vetor(t) for t in textos_str]
embeddings = np.array(embeddings).astype("float32")


print("Criando índice...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)


# 🔹 salvar tudo
faiss.write_index(index, "index.faiss")

with open("textos.pkl", "wb") as f:
    pickle.dump(textos, f)

print("Base pronta!")