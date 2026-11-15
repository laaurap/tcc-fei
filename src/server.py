"""
server.py
Servidor backend para integração direta com a API do Twilio WhatsApp.

Fluxo:
  1. Recebe webhook do Twilio em /twilio-in (texto ou áudio).
  2. Responde imediatamente com TwiML neutro para evitar timeout HTTP do Twilio.
  3. Em segundo plano (BackgroundTasks):
     - Se áudio: realiza download autenticado e executa a transcrição via Faster-Whisper.
     - Envia o texto processado ao analyzer.py (BERT + spaCy + RAG).
     - Formata a mensagem de resposta empática com os canais oficiais de apoio (Ligue 180).
"""

import os
import logging
import tempfile
import mimetypes

import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.background import BackgroundTasks
from twilio.rest import Client
from dotenv import load_dotenv

from analyzer import analisar_texto
from transcriber import transcrever_audio

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tcc_servidor")

# ---------------------------------------------------------------------------
# Configuração do Twilio
# ---------------------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "+14155238886")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    logger.warning(
        "TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN não configurados no .env. "
        "O envio de mensagens (e o download de áudio) falhará."
    )

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None

app = FastAPI(
    title="TCC - Detecção de Violência Psicológica via WhatsApp",
    version="1.0.0"
)


def _enviar_whatsapp(to_number: str, mensagem: str) -> None:
    """Envia mensagem no WhatsApp via Twilio com limite de 1600 caracteres."""
    if twilio_client is None:
        logger.error("Twilio client não configurado — mensagem não enviada.")
        return

    # Limite de segurança para evitar erro da API do Twilio
    LIMITE_CARACTERES = 1550
    if len(mensagem) > LIMITE_CARACTERES:
        mensagem = mensagem[:LIMITE_CARACTERES] + "\n\n[Texto resumido para caber no limite do WhatsApp]"

    try:
        msg = twilio_client.messages.create(
            from_=f"whatsapp:{TWILIO_WHATSAPP_FROM}",
            to=f"whatsapp:{to_number}",
            body=mensagem,
        )
        logger.info(f"Mensagem enviada com sucesso para {to_number}. SID: {msg.sid}")
    except Exception as e:
        logger.exception(f"Erro ao enviar mensagem via Twilio: {e}")


def _baixar_audio(media_url: str, content_type: str) -> str:
    """
    Realiza o download autenticado do arquivo de áudio enviado via Twilio.
    Salva o arquivo em um diretório temporário no disco.
    """
    extensao = mimetypes.guess_extension(content_type or "") or ".ogg"

    resposta = requests.get(
        media_url,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
        timeout=30,
    )
    resposta.raise_for_status()

    arquivo_temp = tempfile.NamedTemporaryFile(delete=False, suffix=extensao)
    arquivo_temp.write(resposta.content)
    arquivo_temp.close()

    logger.info(f"Áudio baixado com sucesso: {arquivo_temp.name} ({len(resposta.content)} bytes)")
    return arquivo_temp.name


def _formatar_mensagem_resposta(resultado: dict) -> str:
    """Constrói o texto acolhedor e informativo com base no nível de risco."""
    nivel = resultado.get("nivel_risco", "DESCONHECIDO").upper()
    justificativa = resultado.get("justificativa", "")
    recomendacao = resultado.get("recomendacao", "")
    
    tipos = resultado.get("tipos_identificados", [])
    str_tipos = f"\n• " + "\n• ".join(tipos) if tipos else " Nenhum tipo específico categorizado."

    if nivel in ["ALTO", "MÉDIO", "MEDIO"]:
        mensagem = (
            f"🛡️ *Análise do Trecho de Conversa*\n\n"
            f"🚨 *Aviso:* Identificamos sinais potenciais de abuso psicológico nesta mensagem.\n"
            f"📊 *Nível de Risco:* {nivel}\n"
            f"🔍 *Categorias Detectadas:*{str_tipos}\n\n"
            f"📌 *Orientação:*\n{recomendacao}\n\n"
            f"--- \n"
            f"🆘 *Canais Gratuitos de Apoio:*\n"
            f"• *Ligue 180* (Central de Atendimento à Mulher - 24h)\n"
            f"• *Ligue 188* (Apoio Emocional)"
        )
    else:
        mensagem = (
            f"🛡️ *Análise do Trecho de Conversa*\n\n"
            f"✅ *Nível de Risco:* BAIXO / NÃO DETECTADO\n\n"
            f"💡 *Resumo da Análise:*\n{justificativa}\n\n"
            f"Se precisar analisar novos trechos, basta enviar por aqui."
        )

    # CORTE DE SEGURANÇA: Garante estritamente menos de 1500 caracteres
    LIMITE_CARACTERES = 1450
    if len(mensagem) > LIMITE_CARACTERES:
        mensagem = mensagem[:LIMITE_CARACTERES] + "\n\n[...Texto resumido devido ao limite do WhatsApp]"

    return mensagem


def _processar_mensagem(texto: str, numero_origem: str) -> None:
    """Executa o pipeline NLP/RAG e envia o resultado de volta ao usuário."""
    try:
        resultado = analisar_texto(texto)
        logger.info(f"Análise concluída para {numero_origem}: {resultado}")
        
        mensagem_final = _formatar_mensagem_resposta(resultado)
        _enviar_whatsapp(numero_origem, mensagem_final)
    except Exception as e:
        logger.exception(f"Erro durante o processamento do texto: {e}")
        _enviar_whatsapp(
            numero_origem,
            "Desculpe, ocorreu uma falha técnica ao analisar a sua mensagem. Por favor, tente novamente em alguns instantes."
        )


def _processar_audio(media_url: str, content_type: str, numero_origem: str) -> None:
    """Baixa, transcreve e executa a análise completa de um arquivo de áudio."""
    caminho_temp = None
    try:
        caminho_temp = _baixar_audio(media_url, content_type)

        logger.info(f"Iniciando transcrição com Faster-Whisper para {numero_origem}...")
        texto = transcrever_audio(caminho_temp)
        logger.info(f"Texto transcrito do áudio: {texto}")

        if not texto or not texto.strip():
            _enviar_whatsapp(
                numero_origem,
                "Não foi possível identificar o áudio. Tente gravar novamente sem ruídos de fundo e um pouco mais perto do microfone."
            )
            return

        _processar_mensagem(texto, numero_origem)

    except Exception as e:
        logger.exception(f"Erro no pipeline de processamento do áudio: {e}")
        _enviar_whatsapp(
            numero_origem,
            "Ocorreu um erro ao baixar ou transcrever o áudio enviado. Tente novamente em instantes."
        )
    finally:
        if caminho_temp and os.path.exists(caminho_temp):
            os.remove(caminho_temp)


@app.post("/twilio-in")
async def twilio_in(request: Request, background_tasks: BackgroundTasks):
    """
    Webhook chamado pelo Twilio quando o número de WhatsApp recebe uma mensagem.
    Responde em < 1 segundo com TwiML limpo e executa a tarefa em background.
    """
    form = await request.form()
    numero_origem = form.get("From", "").replace("whatsapp:", "")
    num_media = int(form.get("NumMedia", "0") or "0")

    if num_media > 0:
        media_url = form.get("MediaUrl0")
        content_type = form.get("MediaContentType0", "")
        logger.info(f"Áudio/mídia recebido de {numero_origem} | Tipo: {content_type}")

        if media_url and numero_origem:
            background_tasks.add_task(_processar_audio, media_url, content_type, numero_origem)
    else:
        texto = form.get("Body", "")
        logger.info(f"Texto recebido de {numero_origem}: {texto}")

        if texto and numero_origem:
            background_tasks.add_task(_processar_mensagem, texto, numero_origem)

    return Response(content="<Response></Response>", media_type="text/xml")


@app.get("/health")
async def health():
    return {"status": "ok", "servico": "Servidor de integração Twilio/WhatsApp em execução."}