"""
api.py
Expõe o analisador de violência psicológica (analyzer.py) como uma API HTTP,
para que o n8n (e qualquer outro sistema) possa chamá-lo via requisição POST.

Como rodar:
    pip install -r requirements_api.txt
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints principais:
    GET  /          - Checagem de status (Health check)
    POST /analisar  - Analisa o texto enviado pelo n8n
"""

import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from analyzer import analisar_texto

# Configuração de logs para monitorar chamadas no terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TCC_API")

app = FastAPI(
    title="API - Detecção de Violência Psicológica (TCC)",
    description="Endpoint de análise NLP/RAG para integração com n8n e Twilio",
    version="1.0.0",
)


class TextoRequest(BaseModel):
    texto: str = Field(
        ..., 
        min_length=1, 
        description="Texto ou transcrição do áudio a ser analisado pela IA",
        example="Você é louca, ninguém mais vai te aguentar se me deixar."
    )


@app.get("/", status_code=status.HTTP_200_OK)
def raiz():
    """Rota de health check para validação de conexão do n8n."""
    return {"status": "ok", "servico": "API do Analisador de Violência Psicológica em execução."}


@app.post("/analisar", status_code=status.HTTP_200_OK)
def analisar(request: TextoRequest):
    """
    Recebe a mensagem de texto/transcrição e retorna a classificação de violência,
    taxa de risco, categorias identificadas e embasamento RAG.
    """
    texto_limpo = request.texto.strip()

    if not texto_limpo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="O campo 'texto' não pode conter apenas espaços em branco."
        )

    try:
        logger.info(f"Recebendo requisição para análise. Tamanho da mensagem: {len(texto_limpo)} caracteres.")
        
        # Chama o analisador NLP + RAG
        resultado = analisar_texto(texto_limpo)
        
        return resultado

    except Exception as e:
        logger.error(f"Erro interno durante a análise do texto: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao processar o texto no modelo NLP: {str(e)}"
        )