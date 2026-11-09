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
    "morrendo de medo", "me mata de vergonha" e qualquer outra ameaça direta.
  - Uma ameaça citada como fala de personagem só é neutralizada com
    enquadramento explícito de filme e comentário de que a cena é engraçada.
  - Risadas escritas (kkkk, haha) NÃO neutralizam nada, porque agressores usam
    "brincadeira" para disfarçar ofensas.

Para incluir uma nova expressão, adicione uma regra em _REGRAS.
"""

import re
import unicodedata

ROTULOS_INTENCAO_CONTEXTO = (
    "pedir desculpas, expressar sentimentos, oferecer apoio ou propor diálogo respeitoso",
    "oferecer ajuda, fazer uma brincadeira ou contar um fato cotidiano sem intenção de ofender",
    "atacar, intimidar, insultar ou tentar controlar alguém",
)
LIMIAR_AGRESSAO_PARA_FILTRO = 0.20
LIMIAR_CONTEXTO_BENIGNO = 0.60
MARGEM_CONTEXTO_NAO_ABUSIVO = 0.25

_PADRAO_ABUSO_EXPLICITO = re.compile(
    r"\b(?:"
    r"inutil|idiota|burra|burro|puta|vagabundo|vagabunda|imbecil|"
    r"estupida|estupido|louca|louco"
    r")\b|"
    r"\b(?:matar|mato|matou|machucar|machuco|bater|espancar)\b|"
    r"\bacabar\s+com\s+(?:a\s+)?(?:sua\s+vida|voce)\b|"
    r"\b(?:proibido|proibir|controlar|controlo|controla)\b|"
    r"\bsem\s+minha\s+permissao\b|"
    r"\bnao\s+(?:pode|vai|deixa|deixarei)\s+(?:sair|falar|ver|decidir)\b|"
    r"\bnunca\s+(?:pode|vai)\s+(?:sair|falar|ver|decidir)\b|"
    r"\b(?:coisa\s+da\s+sua\s+cabeca|nunca\s+aconteceu|inventando|"
    r"faz\s+drama|esta\s+exagerando|a\s+culpa\s+e\s+sua)\b|"
    r"\b(?:sempre|nunca)\s+entende\s+tudo\s+errado\b|"
    r"\b(?:talvez\s+)?o\s+problema\s+(?:seja|e)\s+voce\b|"
    r"\bnao\s+tem\s+direito\s+de\s+decidir\b"
)
_PADRAO_INTENCAO_RESOLUCAO = re.compile(
    r"\b(?:quero|queria|gostaria|preciso|podemos|vamos)\b"
    r"[^.!?]{0,120}\b(?:conversar|dialogar|falar)\b"
    r"[^.!?]{0,120}\bresolver\w*\b"
)

_PRONOME = r"(?:(?:me|te|nos|se|lhe)\s+)?"
_MATAR = r"(?:matar|matando|mata|matam|matou|mataram)"
_MORRER = r"(?:morrer|morrendo|morri|morre|morrem|morreu|morremos|morro|morta|morto|mortas|mortos)"
_RISO = r"(?:tanto\s+)?(?:rir|rindo|risada|risadas|gargalhada|gargalhadas)"
_SAUDADE = r"(?:saudade|saudades)"
_ESTADOS = r"(?P<estado>fome|sono|calor|frio|cansaço|curiosidade|vergonha)"

# (padrão, substituto). O substituto pode ser texto fixo ou uma função.
_REGRAS = [
    # A citação de ameaça fica restrita a um enquadramento ficcional inequívoco.
    (
        re.compile(
            r"\bno filme\b[^.!?]{0,80}\bpersonagem\b[^.!?]{0,40}\bdiz\b"
            r"\s*(['\"]).{0,100}\b(?:matar|machucar|bater)\b.{0,100}\1"
            r"[^.!?]{0,80}\bachei\b[^.!?]{0,40}\bcena\b[^.!?]{0,20}\bengraçad\w*\b",
            re.IGNORECASE,
        ),
        "No filme, a cena foi engraçada",
    ),
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


def eh_contexto_nao_abusivo(
    score_dialogo: float,
    score_cotidiano: float,
    score_agressao: float,
    texto: str,
) -> bool:
    """Reconhece diálogo ou conversa cotidiana sem ocultar sinais de abuso."""
    texto_normalizado = unicodedata.normalize("NFKD", texto)
    texto_normalizado = "".join(
        caractere for caractere in texto_normalizado
        if not unicodedata.combining(caractere)
    ).lower()
    tem_indicador_explicito = bool(_PADRAO_ABUSO_EXPLICITO.search(texto_normalizado))
    quer_resolver_dialogando = bool(_PADRAO_INTENCAO_RESOLUCAO.search(texto_normalizado))
    if tem_indicador_explicito:
        return False

    if quer_resolver_dialogando:
        return score_agressao < LIMIAR_AGRESSAO_PARA_FILTRO

    score_nao_abusivo = score_dialogo + score_cotidiano
    return (
        score_nao_abusivo >= LIMIAR_CONTEXTO_BENIGNO
        and score_nao_abusivo - score_agressao >= MARGEM_CONTEXTO_NAO_ABUSIVO
    )


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
