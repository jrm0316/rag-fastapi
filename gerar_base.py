import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

from pdf_loader2 import carregar_pdf

print("Carregando PDF...")
textos = carregar_pdf("Sistemas.pdf")

textos_str = [item["texto"] for item in textos]

print("Gerando embeddings...")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
embeddings = model.encode(textos_str)
embeddings = np.array(embeddings).astype("float32")

print("Criando índice...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

#  salvar tudo
faiss.write_index(index, "index.faiss")

with open("textos.pkl", "wb") as f:
    pickle.dump(textos, f)

print("Base pronta!")