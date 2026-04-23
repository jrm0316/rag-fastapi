from pypdf import PdfReader
import re

def dividir_texto(texto, tamanho=800, overlap=150):
    frases = re.split(r'(?<=[.!?]) +', texto)

    chunks = []
    chunk = ""

    for frase in frases:
        if len(chunk) + len(frase) < tamanho:
            chunk += " " + frase
        else:
            chunks.append(chunk.strip())
            chunk = frase

    if chunk:
        chunks.append(chunk.strip())

    return chunks


def limpar_texto(texto):
    texto = texto.replace("\n", " ")

    # remove hifens quebrados
    texto = re.sub(r"-\s+", "", texto)

    # remove espaços duplicados
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def carregar_pdf(caminho):
    reader = PdfReader(caminho)
    chunks = []

    for i, pagina in enumerate(reader.pages[:10]):
        texto = pagina.extract_text()
        
        if texto:
            texto = limpar_texto(texto)
            partes = dividir_texto(texto)

            for parte in partes:
                chunks.append({
                    "texto": parte,
                    "pagina": i + 1
                })
    
    return chunks
    