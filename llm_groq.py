import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def responder(pergunta, contexto):
    prompt = f"""
Você é um assistente que responde perguntas baseado em um contexto.

Use APENAS o contexto abaixo.

Se a resposta não estiver explicitamente no contexto, mas puder ser inferida a partir dele, responda com base nas informações disponíveis.

Combine informações de diferentes partes do contexto para formar uma resposta completa.

Se realmente não houver informação suficiente, diga:
"Não encontrei essa informação no documento."

Sempre que possível, informe a(s) página(s) de onde a informação foi retirada.

Contexto:
{contexto}

Pergunta:
{pergunta}
"""
    
    resposta = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    
    return resposta.choices[0].message.content
