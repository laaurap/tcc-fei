"""
contexto.py
Camada de contexto do analisador.

O classificador zero-shot pontua cada frase pelo SIGNIFICADO LITERAL e confunde
expressões figuradas do dia a dia com agressão (ex.: "isso tá me matando de rir"
é marcado como ameaça). Este módulo reescreve essas expressões por equivalentes
neutros ANTES da classificação e fornece um critério contextual para mensagens
que expressam sentimentos e propõem resolver um desacordo por diálogo.

Regras de segurança (para não esconder violência real):
  - Só expressões de uma lista fechada e específica são reescritas.
  - "matar/morrer de ..." só é neutralizado com rir/risada e saudade, e
    "morrer de ..." também com fome, sono, calor, frio, cansaço, curiosidade e vergonha.
  - NÃO são tocados: "vou te matar", "matar de pancada", "me mata de fome",
    "morrendo de medo", "me mata de vergonha" e qualquer outra ameaça ou relato.
  - Risadas escritas (kkkk, haha) NÃO neutralizam nada, porque agressores usam
    "brincadeira" para disfarçar ofensas.

Para incluir uma nova expressão, adicione uma regra em _REGRAS.
"""

import re
import unicodedata

ROTULOS_INTENCAO_CONTEXTO = (
    "expressar tristeza ou outro sentimento próprio e querer conversar para resolver um desacordo",
    "atacar, humilhar, ameaçar, invalidar ou controlar a outra pessoa",
)
LIMIAR_INTENCAO_DIALOGO = 0.90

_PADRAO_ABUSO_EXPLICITO = re.compile(
    r"\b(?:"
    r"inutil|idiota|burra|burro|puta|vagabundo|vagabunda|imbecil|"
    r"estupida|estupido|louca|louco|"
    r")\b|"
    r"\b(?:matar|mato|matou|machucar|machuco|bater|espancar)\b|"
    r"\bacabar\s+com\s+(?:a\s+)?(?:sua\s+vida|voce)\b|"
    r"\b(?:proibido|proibir|controlar|controlo|controla)\b|"
    r"\bsem\s+minha\s+permissao\b|"
    r"\bnao\s+(?:pode|vai|deixa|deixarei)\s+(?:sair|falar|ver|decidir)\b|"
    r"\b(?:coisa\s+da\s+sua\s+cabeca|nunca\s+aconteceu|inventando|"
    r"faz\s+drama|esta\s+exagerando|a\s+culpa\s+e\s+sua)\b"
)

_PRONOME = r"(?:(?:me|te|nos|se|lhe)\s+)?"
_MATAR = r"(?:matar|matando|mata|matam|matou|mataram)"
_MORRER = r"(?:morrer|morrendo|morri|morre|morrem|morreu|morremos|morro|morta|morto|mortas|mortos)"
_RISO = r"(?:tanto\s+)?(?:rir|rindo|risada|risadas|gargalhada|gargalhadas)"
_SAUDADE = r"(?:saudade|saudades)"
_ESTADOS = r"(?P<estado>fome|sono|calor|frio|cansaço|curiosidade|vergonha)"

# (padrão, substituto). O substituto pode ser texto fixo ou uma função.
_REGRAS = [
    # "me matando de rir", "morri de rir", "vai me matar de risada"
    (re.compile(rf"(?<!\w){_PRONOME}(?:{_MATAR}|{_MORRER})\s+de\s+{_RISO}(?!\w)", re.IGNORECASE),
     "rindo muito"),
    # "me mata de saudade", "morrendo de saudades"
    (re.compile(rf"(?<!\w){_PRONOME}(?:{_MATAR}|{_MORRER})\s+de\s+{_SAUDADE}(?!\w)", re.IGNORECASE),
     "com bastante saudade"),
    # "morrendo de fome", "morta de cansaço", "morri de vergonha"
    (re.compile(rf"(?<!\w){_MORRER}\s+de\s+{_ESTADOS}(?!\w)", re.IGNORECASE),
     lambda m: f"com bastante {m.group('estado').lower()}"),
]


def eh_contexto_de_resolucao(
    score_dialogo: float,
    texto: str,
) -> bool:
    """Filtra resolução dialogada, preservando frases com indicadores explícitos."""
    texto_normalizado = unicodedata.normalize("NFKD", texto)
    texto_normalizado = "".join(
        caractere for caractere in texto_normalizado
        if not unicodedata.combining(caractere)
    ).lower()
    tem_indicador_explicito = bool(_PADRAO_ABUSO_EXPLICITO.search(texto_normalizado))
    return score_dialogo >= LIMIAR_INTENCAO_DIALOGO and not tem_indicador_explicito


def neutralizar_expressoes(texto: str):
    """
    Reescreve expressões figuradas por equivalentes neutros.

    Retorna (texto_reescrito, expressoes_encontradas), onde expressoes_encontradas
    é a lista dos trechos originais que foram reescritos.
    """
    encontradas = []

    def _registrar(substituto):
        def _sub(m):
            encontradas.append(m.group(0).strip())
            return substituto(m) if callable(substituto) else substituto
        return _sub

    resultado = texto
    for padrao, substituto in _REGRAS:
        resultado = padrao.sub(_registrar(substituto), resultado)
    return resultado, encontradas


if __name__ == "__main__":
    exemplos = [
        "Nossa isso ta me matando de rir",
        "Vou te matar de pancada",
        "To morrendo de saudade de você",
        "Você me mata de rir, mas é uma inútil",
    ]
    for e in exemplos:
        novo, achados = neutralizar_expressoes(e)
        print(f"{e!r}\n  -> {novo!r}  {achados}")
