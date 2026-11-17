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

_PADRAO_HIPERBOLE_AUTO = re.compile(
    r"\b(?:louca|louco|maluca|maluco|apaixonada|apaixonado)\s+por\s+(?:voce|vc|ti|ele|ela|isso|esse|essa)\b|"
    r"\b(?:quero|vou|vontade\s+de)\s+me\s+(?:matar|jogar|enterrar)\b[^.!?]{0,60}"
    r"\b(?:trabalh\w*|estud\w*|cansad\w*|sono|fome|calor|frio|tanta\s+coisa|tanto\s+boleto|vergonha|prova|reuniao|transito)\b|"
    r"\b(?:vou\s+surtar|surtando|sumir\s+do\s+mapa)\b[^.!?]{0,60}"
    r"\b(?:internet|wifi|sistema|computador|onibus|transito|calor|chuva|fila|trabalho|prova|faculdade)\b"
)

_PADRAO_ABUSO_EXPLICITO = re.compile(
    r"\b(?:"
    r"inutil|idiota|burra|burro|puta|vagabundo|vagabunda|imbecil|"
    r"estupida|estupido"
    r")\b|"
    r"\b(?:voce|vc|tu)\s+(?:e|ta|esta|ficou)\s+(?:uma?\s+)?(?:louca|louco|maluca|maluco)\b|"
    r"\bsua\s+(?:louca|maluca|desgracada)\b|"
    r"\b(?:te|lhe|voce|vc)\s+(?:matar|mato|machucar|machuco|bater|espancar|arrebentar|quebrar)\b|"
    r"\b(?:matar|mato|machucar|machuco|bater|espancar|arrebentar)\s+(?:voce|vc|ela|ele)\b|"
    r"\bse\s+(?:voce|vc)\s+(?:me\s+)?(?:deixar|largar|terminar|for\s+embora)[^.!?]{0,40}\b(?:me\s+mato|vou\s+me\s+matar|acabo\s+com)\b|"
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

# Padrão estrito de ameaça de dano físico / perseguição / chantagem suicida contra o parceiro
# Usado pelo analyzer.py para reservar o nível de risco ALTO apenas a situações de perigo físico/grave.
_PADRAO_RISCO_DANO_FISICO = re.compile(
    r"\b(?:te|lhe|voce|vc)\s+(?:matar|mato|machucar|machuco|bater|espancar|arrebentar|quebrar|esfolar|enforcar)\b|"
    r"\b(?:matar|mato|machucar|machuco|bater|espancar|arrebentar|quebrar)\s+(?:voce|vc|sua\s+cara|toda|todo)\b|"
    r"\b(?:levar|tomar)\s+uma\s+surra\b|"
    r"\bacabar\s+com\s+(?:a\s+)?(?:sua\s+vida|voce)\b|"
    r"\bdestruir\s+(?:a\s+)?sua\s+vida\b|"
    r"\b(?:sei\s+onde\s+voce\s+(?:mora|trabalha|estuda)|te\s+esperar\s+na\s+saida|te\s+pego\s+na\s+saida|apareco\s+na\s+sua\s+porta)\b|"
    r"\b(?:nao\s+sai\s+inteir[oa]|perder\s+a\s+cabeca\s+com\s+voce|fazer\s+voce\s+se\s+arrepender\s+de\s+ter\s+nascido)\b|"
    r"\b(?:te\s+perseguir|te\s+persigo|vou\s+atras\s+de\s+voce|te\s+encontrar\s+onde)[^.!?]{0,50}\b(?:deixar|largar|terminar|fugir|bloquear|trocar)\b|"
    r"\bse\s+(?:voce|vc)\s+(?:me\s+)?(?:deixar|largar|terminar|for\s+embora)[^.!?]{0,50}\b(?:me\s+mato|vou\s+me\s+matar|me\s+machucar|desisto\s+de\s+viver|faco\s+uma\s+besteira|te\s+perseguir|te\s+persigo|acabo\s+com|te\s+mato)\b"
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
    # "quero me matar depois de trabalhar tanto", "vontade de me matar de tanto estudar"
    (
        re.compile(
            r"\b(?:quero|vou|dá\s+vontade\s+de|da\s+vontade\s+de|vontade\s+de)\s+me\s+matar\s+"
            r"(?:depois\s+de\s+|de\s+tanto\s+|de\s+tanta\s+|por\s+|com\s+esse\s+|com\s+essa\s+|nesse\s+|nessa\s+)"
            r"(?:trabalh\w*|estud\w*|cansac\w*|cansaç\w*|calor|frio|transito|trânsito|fome|sono|prova|reuni\w*|rotina|semana|dia)",
            re.IGNORECASE,
        ),
        "estou muito exausto com a rotina",
    ),
    # "completamente louca por você", "não sei o que faria sem você na minha vida"
    (
        re.compile(
            r"\b(?:eu\s+)?(?:sou\s+|estou\s+|to\s+|tô\s+)?(?:completamente\s+|totalmente\s+|muito\s+|tão\s+|tao\s+)?"
            r"(?:louca|louco|maluca|maluco|apaixonada|apaixonado)\s+por\s+(?:você|voce|vc|ti)\b"
            r"(?:\s*,?\s*n[aã]o\s+sei\s+o\s+que\s+(?:eu\s+)?faria\s+sem\s+(?:você|voce|vc)(?:\s+na\s+minha\s+vida)?)?",
            re.IGNORECASE,
        ),
        "gosto muito de você",
    ),
    # "vou surtar e sumir do mapa se essa internet...", "vou surtar com esse computador"
    (
        re.compile(
            r"\b(?:eu\s+)?vou\s+surtar(?:\s+e\s+sumir\s+do\s+mapa)?\s+"
            r"(?:se\s+ess[ea]\s+|com\s+ess[ea]\s+|por\s+causa\s+dess[ea]\s+)"
            r"(?:internet|wi-?fi|computador|pc|celular|sistema|trânsito|transito|calor|chuva|fila|impressora|rede)",
            re.IGNORECASE,
        ),
        "estou frustrado com esse problema técnico",
    ),
]


def _normalizar(texto: str) -> str:
    texto_normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(
        caractere for caractere in texto_normalizado
        if not unicodedata.combining(caractere)
    ).lower()


def tem_abuso_explicito(texto: str) -> bool:
    """Retorna True se o texto contém um padrão inequívoco de abuso explícito."""
    norm = _normalizar(texto)
    if _PADRAO_HIPERBOLE_AUTO.search(norm) and not _PADRAO_RISCO_DANO_FISICO.search(norm):
        return False
    return bool(_PADRAO_ABUSO_EXPLICITO.search(norm))


def tem_risco_dano_fisico(texto: str, notas_sentenca: dict[str, float] | None = None, score_sup: float | None = None) -> bool:
    """
    Define se a mensagem representa risco ALTO (possível evolução para dano físico,
    ameaça direta, perseguição ou chantagem de autoextermínio contra o parceiro).
    Violências puramente verbais/manipulativas sem ameaça de dano físico ficam em MEDIO/BAIXO.
    """
    norm = _normalizar(texto)
    if _PADRAO_HIPERBOLE_AUTO.search(norm):
        return False
    if _PADRAO_RISCO_DANO_FISICO.search(norm):
        return True
    # Também aceita como ALTO quando a categoria 'ameaça' do zero-shot é altíssima (>= 0.85)
    # E o modelo supervisionado confirma fortemente (>= 0.85).
    if notas_sentenca and score_sup is not None:
        if notas_sentenca.get("ameaça", 0.0) >= 0.85 and score_sup >= 0.85:
            return True
    return False


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
    tem_indicador_explicito = tem_abuso_explicito(texto)
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
