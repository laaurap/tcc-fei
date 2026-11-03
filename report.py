"""
report.py
Exibe o resultado da análise de forma clara no terminal usando Rich.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from datetime import datetime

console = Console()


def exibir_resultado(resultado: dict, texto_analisado: str = "", origem: str = "texto"):
    """
    Exibe o resultado da análise de forma visual no terminal.

    Parâmetros:
        resultado       : dicionário retornado por analyzer.analisar_texto()
        texto_analisado : texto original que foi analisado (opcional)
        origem          : "texto" ou "audio" — indica a fonte da entrada
    """

    # Define cor e emoji de acordo com o veredicto
    veredicto = resultado.get("veredicto", "NAO")
    nivel = resultado.get("nivel_risco", "NENHUM")

    cor_veredicto = {
        "SIM":      "bold red",
        "POSSIVEL": "bold yellow",
        "NAO":      "bold green",
    }.get(veredicto, "white")

    cor_nivel = {
        "ALTO":   "red",
        "MEDIO":  "yellow",
        "BAIXO":  "cyan",
        "NENHUM": "green",
    }.get(nivel, "white")

    icone = {
        "SIM":      "⚠️  VIOLÊNCIA PSICOLÓGICA DETECTADA",
        "POSSIVEL": "❓  POSSÍVEL VIOLÊNCIA PSICOLÓGICA",
        "NAO":      "✅  NENHUMA VIOLÊNCIA DETECTADA",
    }.get(veredicto, "")

    console.print()
    console.rule("[bold]Relatório de Análise — TCC Violência Psicológica[/bold]")
    console.print(f"[dim]Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Fonte: {origem}[/dim]")
    console.print()

    # Painel principal do veredicto
    console.print(Panel(
        f"[{cor_veredicto}]{icone}[/{cor_veredicto}]\n\n"
        f"Nível de risco: [{cor_nivel}]{nivel}[/{cor_nivel}]   "
        f"Confiança: [bold]{resultado.get('score_confianca', 0):.0%}[/bold]",
        title="Veredicto",
        border_style=cor_veredicto,
    ))

    # Tipos detectados
    tipos = resultado.get("tipos_detectados", [])
    if tipos:
        console.print("\n[bold]Tipos de violência identificados:[/bold]")
        for t in tipos:
            console.print(f"  • {t}")

    # Trechos relevantes
    trechos = resultado.get("trechos_relevantes", [])
    if trechos:
        console.print("\n[bold]Trechos de alerta:[/bold]")
        for tr in trechos:
            console.print(f'  [italic yellow]"{tr}"[/italic yellow]')

    # Justificativa
    console.print(Panel(
        resultado.get("justificativa", ""),
        title="Análise Detalhada",
        border_style="dim",
    ))

    # Recomendação
    console.print(Panel(
        resultado.get("recomendacao", ""),
        title="Recomendação",
        border_style="blue",
    ))

    console.print()