"""
main.py
Ponto de entrada do sistema de detecção de violência psicológica.

Como usar:
  # Analisar texto digitado:
  python main.py --texto "Você nunca faz nada certo, é inútil mesmo."

  # Analisar arquivo de áudio:
  python main.py --audio conversa.wav

  # Modo interativo (digita na hora):
  python main.py
"""

import argparse
import os
import sys

from rich.console import Console

from transcriber import transcrever_audio
from analyzer import analisar_texto
from report import exibir_resultado

console = Console()


def fluxo_audio(caminho: str):
    if not os.path.exists(caminho):
        console.print(f"[red]Arquivo não encontrado: {caminho}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Transcrevendo áudio:[/cyan] {caminho}")
    texto = transcrever_audio(caminho)

    console.print("\n[bold]Texto transcrito:[/bold]")
    console.print(f"[dim]{texto}[/dim]\n")

    resultado = analisar_texto(texto)
    exibir_resultado(resultado, texto_analisado=texto, origem="áudio")


def fluxo_texto(texto: str):
    if not texto.strip():
        console.print("[red]Texto vazio. Nada a analisar.[/red]")
        sys.exit(1)

    resultado = analisar_texto(texto)
    exibir_resultado(resultado, texto_analisado=texto, origem="texto")


def fluxo_interativo():
    console.print("[bold cyan]Modo interativo[/bold cyan] — Digite ou cole o texto da conversa.")
    console.print("[dim](Para finalizar, digite uma linha contendo apenas 'FIM')[/dim]\n")

    linhas = []
    while True:
        linha = input()
        if linha.strip().upper() == "FIM":
            break
        linhas.append(linha)

    texto = "\n".join(linhas)
    fluxo_texto(texto)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Detecção de Violência Psicológica em Conversas — TCC"
    )
    parser.add_argument("--audio", type=str, help="Caminho para arquivo de áudio (.wav, .mp3, .m4a...)")
    parser.add_argument("--texto", type=str, help="Texto a ser analisado (entre aspas)")
    args = parser.parse_args()

    console.print("[bold]Sistema de Detecção de Violência Psicológica[/bold]")
    console.print("[dim]TCC — Análise via Faster Whisper + spaCy[/dim]\n")

    if args.audio:
        fluxo_audio(args.audio)
    elif args.texto:
        fluxo_texto(args.texto)
    else:
        fluxo_interativo()


if __name__ == "__main__":
    main()