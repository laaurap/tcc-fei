"""
analyzer.py
Analisa texto em busca de violência psicológica usando
spaCy (pré-processamento) + HuggingFace XLM-RoBERTa (Zero-Shot por categoria)
+ DistilBERT fine-tuned (HateBR + dataset próprio) + RAG.
"""

import os
import re

import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

from rag import buscar_referencias, gerar_justificativa_fundamentada
from contexto import (
    ROTULOS_INTENCAO_CONTEXTO,
    eh_contexto_nao_abusivo,
    neutralizar_expressoes,
    tem_abuso_explicito,
    tem_risco_dano_fisico,
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

# --- Parâmetros de decisão ---
LIMIAR = 0.65         # nota mínima para uma categoria ser considerada (ajustado p/ WhatsApp)
MARGEM_CATEGORIAS = 0.05  # só são listadas categorias até 5 pontos abaixo da maior nota
MAX_CATEGORIAS = 3        # máximo de categorias listadas na resposta
LIMIAR_CONTEXTO_AGRESSAO_ALTO = 0.50

# --- Parâmetros do modelo supervisionado (fine-tuning HateBR + dataset próprio) ---
_RAIZ_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CANDIDATOS_SUPERVISIONADO = (
    os.path.join(_RAIZ_PROJETO, "resultados_modelos", "combinado_v3"),
    os.path.join(_RAIZ_PROJETO, "resultados_modelos", "combinado_treinado"),
    os.path.join(_RAIZ_PROJETO, "resultados_modelos", "hatebr_treinado"),
)
LIMIAR_SUP_INOFENSIVO = 0.25   # abaixo disso, o modelo fine-tuned considera a frase benigna/hipérbole
LIMIAR_SUP_ALERTA = 0.60       # acima disso, o modelo fine-tuned confirma indício de violência


def carregar_modelos():
    """Carrega o spaCy, o pipeline Zero-Shot e o classificador supervisionado (fine-tuned)."""
    try:
        nlp = spacy.load("pt_core_news_lg")
    except OSError:
        print("[Aviso] Modelo 'pt_core_news_lg' não encontrado. Usando 'pt_core_news_sm'.")
        try:
            nlp = spacy.load("pt_core_news_sm")
        except OSError:
            print(
                "[Aviso] Modelo 'pt_core_news_sm' não encontrado. "
                "Usando tokenização e segmentação básicas em português."
            )
            nlp = spacy.blank("pt")
            nlp.add_pipe("sentencizer")

    # Detecta automaticamente GPU se disponível
    dispositivo = 0 if torch.cuda.is_available() else -1
    str_dev = "GPU (CUDA)" if dispositivo == 0 else "CPU"

    print(f"[Modelo] Carregando BERT Zero-Shot em {str_dev}...")
    classificador = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        device=dispositivo
    )

    tok_sup, mod_sup = None, None
    for caminho in _CANDIDATOS_SUPERVISIONADO:
        if os.path.isdir(caminho) and os.path.exists(os.path.join(caminho, "config.json")):
            try:
                tok_sup = AutoTokenizer.from_pretrained(caminho)
                mod_sup = AutoModelForSequenceClassification.from_pretrained(caminho)
                if dispositivo == 0:
                    mod_sup = mod_sup.to("cuda")
                mod_sup.eval()
                print(f"[Modelo] Modelo supervisionado (fine-tuned) carregado de: {os.path.basename(caminho)}")
                break
            except Exception as e:
                print(f"[Aviso] Falha ao carregar modelo supervisionado em {caminho}: {e}")

    print("[Modelo] Modelos carregados com sucesso!")

    return nlp, classificador, tok_sup, mod_sup


_nlp, _classificador, _tok_sup, _mod_sup = carregar_modelos()


def _pontuar_supervisionado(texto: str) -> float | None:
    """Retorna a probabilidade de violência/ofensa segundo o modelo fine-tuned (0.0 a 1.0)."""
    if _tok_sup is None or _mod_sup is None:
        return None
    entrada = _tok_sup(texto, return_tensors="pt", truncation=True, max_length=256)
    entrada.pop("token_type_ids", None)
    entrada = {k: v.to(_mod_sup.device) for k, v in entrada.items()}
    with torch.no_grad():
        return torch.softmax(_mod_sup(**entrada).logits, dim=-1)[0, 1].item()


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
    maior_score_agressao_contexto = 0.0
    maior_score_supervisionado = 0.0
    neutralizadas_supervisionado = False
    ha_risco_dano_fisico = False

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
            sentenca,
            candidate_labels=list(ROTULOS_INTENCAO_CONTEXTO),
            hypothesis_template="A intenção principal desta mensagem é {}.",
            multi_label=False,
        )
        scores_contexto = dict(zip(resultado_contexto["labels"], resultado_contexto["scores"]))
        score_dialogo = scores_contexto[ROTULOS_INTENCAO_CONTEXTO[0]]
        score_cotidiano = scores_contexto[ROTULOS_INTENCAO_CONTEXTO[1]]
        score_agressao = scores_contexto[ROTULOS_INTENCAO_CONTEXTO[2]]
        if eh_contexto_nao_abusivo(
            score_dialogo,
            score_cotidiano,
            score_agressao,
            sentenca,
        ):
            scores_por_sentenca.append((sentenca_original, {}))
            continue

        # --- Validação pelo modelo supervisionado (fine-tuned) ---
        score_sup = _pontuar_supervisionado(sentenca)
        if score_sup is not None:
            maior_score_supervisionado = max(maior_score_supervisionado, score_sup)
            # Se não há abuso explícito por regra e o modelo fine-tuned dá nota muito baixa
            # (< 0.25), trata como hipérbole/cotidiano não abusivo.
            if not tem_abuso_explicito(sentenca) and score_sup < LIMIAR_SUP_INOFENSIVO:
                neutralizadas_supervisionado = True
                scores_por_sentenca.append((sentenca_original, {}))
                continue

        limiar_efetivo = (
            0.55 if (score_sup is not None and score_sup >= 0.85) else LIMIAR
        )
        if any(score >= limiar_efetivo for score in notas_sentenca.values()):
            maior_score_agressao_contexto = max(
                maior_score_agressao_contexto,
                score_agressao,
            )

        for chave_cat, score in notas_sentenca.items():
            # Se o supervisionado tem alta confiança (>= 0.85) e a melhor categoria
            # zero-shot ficou entre 0.55 e 0.65, eleva ao limiar para não perder o alerta.
            if score_sup is not None and score_sup >= 0.85 and score >= 0.55 and score < LIMIAR:
                if score == max(notas_sentenca.values()):
                    score = LIMIAR
                    notas_sentenca[chave_cat] = score
            if score > scores_por_categoria[chave_cat]:
                scores_por_categoria[chave_cat] = score
        if tem_risco_dano_fisico(sentenca_original, notas_sentenca, score_sup):
            ha_risco_dano_fisico = True
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
        elif neutralizadas_supervisionado:
            justificativa += " O classificador supervisionado identificou o contexto como hipérbole ou conversa cotidiana não abusiva."
    elif (
        qtd == 1
        and (
            score_geral < 0.75
            or (_mod_sup is not None and maior_score_supervisionado < LIMIAR_SUP_ALERTA and not tem_abuso_explicito(texto))
        )
    ):
        veredicto = "POSSIVEL"
        nivel_risco = "BAIXO"
        justificativa = (
            f"Identificado um indicador pontual de {tipos_identificados[0]} "
            f"com nível de confiança de {score_exibido:.0%}."
        )
    elif (
        not ha_risco_dano_fisico
        or score_geral < 0.70
        or maior_score_agressao_contexto < LIMIAR_CONTEXTO_AGRESSAO_ALTO
        or (_mod_sup is not None and maior_score_supervisionado < LIMIAR_SUP_ALERTA and not tem_abuso_explicito(texto))
    ):
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
                f"Identificado um padrão grave de violência psicológica / ameaça à integridade: {lista}. "
                f"Confiança do modelo: {score_exibido:.0%}."
            )
        else:
            justificativa = (
                f"Identificados {n} padrões graves de violência psicológica / ameaça à integridade: {lista}. "
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
        "score_contexto_agressao": round(maior_score_agressao_contexto, 2),
        "score_supervisionado": round(maior_score_supervisionado, 2),
        "tipos_identificados": tipos_identificados,
        "trechos_relevantes": trechos_relevantes,
        "justificativa": justificativa,
        "recomendacao": recomendacao,
        "referencias": [r["id"] for r in referencias_usadas],
        "expressoes_neutralizadas": expressoes_neutralizadas,
    }