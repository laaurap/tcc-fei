"""
rag.py
Implementação de RAG (Retrieval-Augmented Generation) leve, sem depender
de LLM pago:

  1. RETRIEVAL: gera embeddings da base de referência (knowledge_base.py)
     usando sentence-transformers (modelo multilíngue, roda local/grátis) e
     busca os trechos mais similares às categorias detectadas pelo BERT.

  2. AUGMENTED JUSTIFICATION: em vez de "gerar" texto novo com um LLM,
     usamos os trechos recuperados para enriquecer a justificativa do
     analyzer com fundamentação técnica/legal — o que cumpre o papel do
     "G" (generation) de forma extractiva e sem custo de API.

Se no futuro quiserem trocar por geração real com LLM (OpenAI/Claude), basta
substituir a função `gerar_justificativa_fundamentada` para chamar a API
externa passando os `trechos_recuperados` como contexto do prompt.
"""

from sentence_transformers import SentenceTransformer, util
from knowledge_base import REFERENCIAS

_MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

print("[RAG] Carregando modelo de embeddings...")
_modelo = SentenceTransformer(_MODELO_EMBEDDING)

# Pré-calcula os embeddings da base de referência uma única vez
_textos_referencia = [r["texto"] for r in REFERENCIAS]
_embeddings_referencia = _modelo.encode(_textos_referencia, convert_to_tensor=True)
print(f"[RAG] Base de conhecimento carregada ({len(REFERENCIAS)} referências).")


def buscar_referencias(tipos_detectados: list[str], texto_original: str, top_k: int = 2) -> list[dict]:
    """
    Busca (retrieval) os trechos da base de conhecimento mais relevantes
    para os tipos de violência detectados pelo BERT.

    Estratégia:
      - Se o BERT detectou categorias específicas, prioriza referências
        daquelas categorias.
      - Complementa com busca por similaridade semântica ao texto original,
        caso as categorias não cubram tudo (ex: nenhuma categoria com
        confiança suficiente, mas o texto ainda é suspeito).
    """
    if not tipos_detectados:
        return []

    # 1. Referências que batem exatamente com as categorias detectadas
    referencias_diretas = [r for r in REFERENCIAS if r["categoria"] in tipos_detectados]

    # 2. Busca por similaridade semântica como reforço/desempate
    embedding_texto = _modelo.encode(texto_original, convert_to_tensor=True)
    scores = util.cos_sim(embedding_texto, _embeddings_referencia)[0]

    ranking = sorted(
        zip(REFERENCIAS, scores.tolist()),
        key=lambda par: par[1],
        reverse=True,
    )

    resultado = []
    ids_incluidos = set()

    # Prioriza as referências diretas das categorias detectadas
    for ref in referencias_diretas:
        if ref["id"] not in ids_incluidos:
            resultado.append(ref)
            ids_incluidos.add(ref["id"])

    # Completa com as mais similares semanticamente, até o limite top_k
    for ref, score in ranking:
        if len(resultado) >= top_k:
            break
        if ref["id"] not in ids_incluidos:
            resultado.append(ref)
            ids_incluidos.add(ref["id"])

    return resultado[:top_k]


def gerar_justificativa_fundamentada(justificativa_bert: str, trechos_recuperados: list[dict]) -> str:
    """
    Combina a justificativa original do BERT com os trechos recuperados,
    citando a fonte de cada um — cumprindo o papel de "geração aumentada
    por recuperação" sem precisar de um LLM externo pago.
    """
    if not trechos_recuperados:
        return justificativa_bert

    fundamentacao = "\n\n📚 Fundamentação:\n"
    for ref in trechos_recuperados:
        fundamentacao += f"- {ref['titulo']} ({ref.get('fonte', 'fonte não especificada')}): {ref['texto']}\n"

    return justificativa_bert + fundamentacao