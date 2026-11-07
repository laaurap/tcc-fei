"""
analyzer.py
Analisa texto em busca de violência psicológica usando
spaCy (pré-processamento) + HuggingFace XLM-RoBERTa / BERT (classificação Zero-Shot) + RAG.
"""

import re

import spacy
from transformers import pipeline
import torch

from rag import buscar_referencias, gerar_justificativa_fundamentada
from contexto import (
    ROTULOS_INTENCAO_CONTEXTO,
    eh_contexto_de_resolucao,
    neutralizar_expressoes,
)

# Categorias de violência psicológica e descrições para o modelo zero-shot
CATEGORIAS = {
    "xingamento":          "usar palavrão, xingar diretamente, insultar com palavras obscenas como puta, vagabundo, idiota, inútil",
    "humilhação":          "diminuir o valor da pessoa, dizer que não presta, que é burra, que não é capaz, rebaixar, envergonhar",
    "ameaça":              "ameaçar de agressão física, dizer que vai machucar, matar, se arrepender, dar ultimato violento",
    "gaslighting":         "negar que algo aconteceu, dizer que está inventando, que é coisa da sua cabeça, distorcer a realidade da pessoa",
    "controle":            "proibir de sair, controlar com quem fala, monitorar, não deixar tomar decisões, impor regras por dominação",
    "chantagem_emocional": "ameaçar se matar se a pessoa for embora, culpar a pessoa por tudo, usar amor ou culpa para manipular",
    "invalidação":         "dizer que é drama, frescura, exagero, que está inventando sentimentos, ignorar sofrimento alheio",
}

# Mapeamento para nomes amigáveis na resposta final do WhatsApp / API
NOME_CATEGORIAS_FORMATADO = {
    "xingamento": "Xingamento / Insulto",
    "humilhação": "Humilhação e Rebaixamento",
    "ameaça": "Ameaça Explicita ou Velada",
    "gaslighting": "Gaslighting (Manipulação da Realidade)",
    "controle": "Controle e Isolamento Social",
    "chantagem_emocional": "Chantagem Emocional",
    "invalidação": "Invalidação Emocional / Desprezo",
}

PALAVRAS_POSITIVAS = [
    "adoro", "amo", "linda", "lindo", "bonita", "bonito", "maravilhosa",
    "maravilhoso", "incrível", "feliz", "felicidade", "obrigada", "obrigado",
    "parabéns", "orgulho", "carinho", "amor", "querida", "querido",
    "saudade", "gratidão", "grato", "grata", "admiro", "especial",
]

# --- Parâmetros de decisão ---
LIMIAR = 0.65         # nota mínima para uma categoria ser considerada (ajustado p/ WhatsApp)
MARGEM_CATEGORIAS = 0.05  # só são listadas categorias até 5 pontos abaixo da maior nota
MAX_CATEGORIAS = 3        # máximo de categorias listadas na resposta


def carregar_modelos():
    """Carrega o spaCy e o pipeline de classificação HuggingFace."""
    try:
        nlp = spacy.load("pt_core_news_lg")
    except OSError:
        print("[Aviso] Modelo 'pt_core_news_lg' não encontrado. Usando 'pt_core_news_sm'.")
        nlp = spacy.load("pt_core_news_sm")

    # Detecta automaticamente GPU se disponível
    dispositivo = 0 if torch.cuda.is_available() else -1
    str_dev = "GPU (CUDA)" if dispositivo == 0 else "CPU"

    print(f"[Modelo] Carregando BERT em {str_dev}...")
    classificador = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        device=dispositivo
    )
    print("[Modelo] Modelo BERT e spaCy carregados com sucesso!")

    return nlp, classificador


_nlp, _classificador = carregar_modelos()


def _texto_e_positivo(texto: str) -> bool:
    """Valida se a mensagem tem viés predominantemente afetivo/neutro."""
    texto_lower = texto.lower()
    qtd_positivas = sum(1 for p in PALAVRAS_POSITIVAS if p in texto_lower)

    palavras_agressivas = [
        "matar", "bater", "inútil", "idiota", "burra", "puta",
        "vagabundo", "odeio", "imbecil", "estúpida", "proibido",
        "nunca", "culpa", "arrependida", "louca"
    ]
    qtd_agressivas = sum(1 for p in palavras_agressivas if p in texto_lower)

    return qtd_positivas >= 2 and qtd_agressivas == 0


def _tem_conteudo_apos_neutralizacao(texto: str, expressoes: list[str]) -> bool:
    restante = texto
    for expressao in expressoes:
        restante = re.sub(re.escape(expressao), " ", restante, flags=re.IGNORECASE)

    palavras_contextuais = {"ta", "tá", "to", "tô", "está", "estou"}
    return any(
        token.is_alpha and not token.is_stop and token.lower_ not in palavras_contextuais
        for token in _nlp(restante)
    )


def analisar_texto(texto: str) -> dict:
    """Pipeline completo de análise NLP + RAG."""
    # --- Contexto: reescreve expressões figuradas (ex.: "matando de rir") ---
    texto_original = texto
    texto, expressoes_neutralizadas = neutralizar_expressoes(texto_original)

    if _texto_e_positivo(texto):
        return {
            "veredicto": "NAO",
            "nivel_risco": "NENHUM",
            "score_confianca": 0.0,
            "tipos_identificados": [],
            "trechos_relevantes": [],
            "justificativa": "O texto apresenta contexto predominantemente positivo ou afetivo. Nenhum indicador de violência psicológica foi identificado.",
            "recomendacao": "Nenhuma ação necessária com base nesta mensagem.",
            "referencias": [],
            "expressoes_neutralizadas": expressoes_neutralizadas,
        }

    doc = _nlp(texto)
    sentencas = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentencas:
        sentencas = [texto]

    doc_original = _nlp(texto_original) if expressoes_neutralizadas else None
    sentencas_originais = (
        [sent.text.strip() for sent in doc_original.sents if sent.text.strip()]
        if doc_original is not None
        else sentencas
    )
    if not sentencas_originais:
        sentencas_originais = [texto_original]

    labels_descricoes = list(CATEGORIAS.values())
    labels_chaves = list(CATEGORIAS.keys())

    scores_por_categoria = {cat: 0.0 for cat in labels_chaves}  # maior nota de cada categoria
    scores_por_sentenca = []                                     # [(sentença, {categoria: nota})]

    for sentenca, sentenca_original in zip(sentencas, sentencas_originais):
        if expressoes_neutralizadas and not _tem_conteudo_apos_neutralizacao(
            sentenca_original, expressoes_neutralizadas
        ):
            scores_por_sentenca.append((sentenca_original, {}))
            continue

        resultado_hf = _classificador(
            sentenca,
            candidate_labels=labels_descricoes,
            hypothesis_template="Este texto contém: {}.",
            multi_label=True
        )

        notas_sentenca = {}
        for label_desc, score in zip(resultado_hf["labels"], resultado_hf["scores"]):
            chave_cat = labels_chaves[labels_descricoes.index(label_desc)]
            notas_sentenca[chave_cat] = score

        resultado_contexto = _classificador(
            sentenca_original,
            candidate_labels=list(ROTULOS_INTENCAO_CONTEXTO),
            hypothesis_template="O texto expressa: {}.",
            multi_label=False,
        )
        score_dialogo = resultado_contexto["scores"][
            resultado_contexto["labels"].index(ROTULOS_INTENCAO_CONTEXTO[0])
        ]
        if eh_contexto_de_resolucao(
            score_dialogo,
            sentenca,
        ):
            scores_por_sentenca.append((sentenca_original, {}))
            continue

        for chave_cat, score in notas_sentenca.items():
            if score > scores_por_categoria[chave_cat]:
                scores_por_categoria[chave_cat] = score
        scores_por_sentenca.append((sentenca_original, notas_sentenca))

    # --- Veredicto e nível de risco ---
    # Calculados a partir de todas as categorias acima do limiar (regra original,
    # mantida para não alterar os resultados das avaliações já feitas).
    detectadas = [c for c in labels_chaves if scores_por_categoria[c] >= LIMIAR]
    qtd = len(detectadas)
    score_geral = sum(scores_por_categoria[c] for c in detectadas) / qtd if qtd else 0.0

    # --- Categorias que serão EXIBIDAS ---
    # Em vez de listar todas as que passaram do limiar, lista só as que mais se destacam:
    # as mais próximas da maior nota (MARGEM_CATEGORIAS), no máximo MAX_CATEGORIAS.
    selecionadas = []
    if detectadas:
        ordenadas = sorted(detectadas, key=lambda c: scores_por_categoria[c], reverse=True)
        maior_nota = scores_por_categoria[ordenadas[0]]
        selecionadas = [c for c in ordenadas
                        if scores_por_categoria[c] >= maior_nota - MARGEM_CATEGORIAS][:MAX_CATEGORIAS]

    tipos_identificados = [NOME_CATEGORIAS_FORMATADO.get(c, c) for c in selecionadas]
    trechos_relevantes = [s for s, notas in scores_por_sentenca
                          if any(notas.get(c, 0.0) >= LIMIAR for c in selecionadas)]
    score_exibido = (sum(scores_por_categoria[c] for c in selecionadas) / len(selecionadas)
                     if selecionadas else 0.0)
    n = len(selecionadas)
    lista = ", ".join(tipos_identificados)

    if qtd == 0:
        veredicto = "NAO"
        nivel_risco = "NENHUM"
        justificativa = "O modelo não identificou indicadores significativos de violência psicológica na mensagem."
        if expressoes_neutralizadas:
            exemplos = ", ".join(f"'{e}'" for e in expressoes_neutralizadas)
            justificativa += f" Expressões figuradas ({exemplos}) foram tratadas como linguagem do dia a dia."
    elif qtd == 1 and score_geral < 0.65:
        veredicto = "POSSIVEL"
        nivel_risco = "BAIXO"
        justificativa = (
            f"Identificado um indicador pontual de {tipos_identificados[0]} "
            f"com nível de confiança de {score_exibido:.0%}."
        )
    elif qtd <= 2 or score_geral < 0.70:
        veredicto = "POSSIVEL"
        nivel_risco = "MEDIO"
        if n == 1:
            justificativa = (
                f"Identificado um indício de manipulação ou agressão verbal: {lista}. "
                f"Confiança do modelo: {score_exibido:.0%}."
            )
        else:
            justificativa = (
                f"Identificados {n} indícios de manipulação ou agressão verbal: {lista}. "
                f"Confiança média do modelo: {score_exibido:.0%}."
            )
    else:
        veredicto = "SIM"
        nivel_risco = "ALTO"
        if n == 1:
            justificativa = (
                f"Identificado um padrão claro de violência psicológica: {lista}. "
                f"Confiança do modelo: {score_exibido:.0%}."
            )
        else:
            justificativa = (
                f"Identificados {n} padrões claros de violência psicológica: {lista}. "
                f"Confiança média do modelo: {score_exibido:.0%}."
            )

    recomendacao = (
        "Se você se sente controlada, ameaçada ou invalidada nesta relação, procure apoio profissional "
        "ou entre em contato com os canais de apoio gratuito. Ligue 180 (Central de Atendimento à Mulher)."
        if veredicto != "NAO"
        else "Nenhuma ação imediata necessária."
    )

    # RAG: busca citações da base de conhecimento, só para as categorias exibidas
    referencias_usadas = buscar_referencias(selecionadas, texto)
    if referencias_usadas:
        justificativa = gerar_justificativa_fundamentada(justificativa, referencias_usadas)

    return {
        "veredicto": veredicto,
        "nivel_risco": nivel_risco,
        "score_confianca": round(score_exibido, 2),
        "tipos_identificados": tipos_identificados,
        "trechos_relevantes": trechos_relevantes,
        "justificativa": justificativa,
        "recomendacao": recomendacao,
        "referencias": [r["id"] for r in referencias_usadas],
        "expressoes_neutralizadas": expressoes_neutralizadas,
    }