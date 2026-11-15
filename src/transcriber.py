"""
transcriber.py
Transcreve arquivos de áudio para texto usando o Faster Whisper.
"""

from faster_whisper import WhisperModel


def transcrever_audio(caminho_audio: str, modelo: str = "small", dispositivo: str = "cpu") -> str:
    """
    Recebe o caminho de um arquivo de áudio e retorna o texto transcrito.

    Parâmetros:
        caminho_audio  : caminho para o arquivo .mp3, .wav, .m4a, etc.
        modelo         : tamanho do modelo Whisper ("tiny", "base", "small", "medium", "large-v3")
                         Para TCC em CPU, "small" é um bom equilíbrio de velocidade x qualidade.
        dispositivo    : "cpu" ou "cuda" (se tiver GPU NVIDIA disponível)

    Retorna:
        Texto transcrito como string única.
    """
    print(f"[Transcrição] Carregando modelo Whisper '{modelo}' em '{dispositivo}'...")
    model = WhisperModel(modelo, device=dispositivo, compute_type="int8")

    print(f"[Transcrição] Processando áudio: {caminho_audio}")
    segmentos, info = model.transcribe(caminho_audio, language="pt")

    print(f"[Transcrição] Idioma detectado: {info.language} (confiança: {info.language_probability:.0%})")

    texto_completo = " ".join(segmento.text.strip() for segmento in segmentos)
    return texto_completo.strip()


if __name__ == "__main__":
    # Teste rápido — troque pelo caminho de um áudio seu
    resultado = transcrever_audio("exemplo.wav")
    print("\nTexto transcrito:")
    print(resultado)