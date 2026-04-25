import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def responder(pergunta, contexto):
    prompt = f"""
Você é um especialista em sistemas operacionais.

Responda a pergunta usando o contexto abaixo.

- Se houver definição, use ela
- Se não houver, explique com base no contexto
- Seja claro, direto e didático
- Não diga que não encontrou, tente responder

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
"""
    
    resposta = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    
    return resposta.choices[0].message.content
