import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def responder(pergunta, contexto):
    prompt = f"""
Use o contexto abaixo para responder a pergunta.
Se não encontrar diretamente, tente inferir com base no conteúdo.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta clara e direta:
"""
    
    resposta = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    
    return resposta.choices[0].message.content
