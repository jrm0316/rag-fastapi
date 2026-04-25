import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def responder(pergunta, contexto):
    prompt = f"""
Responda a pergunta usando o contexto abaixo.

Se não houver definição explícita, explique com base no conhecimento geral e no contexto.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta clara, objetiva e didática:
"""
    
    resposta = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    
    return resposta.choices[0].message.content
