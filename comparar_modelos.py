"""
comparar_modelos.py

Compara modelos de classificação zero-shot (NLI) para detectar violência
psicológica, usando a mesma amostra e as mesmas regras para todos.

Dados (iguais para todos os modelos):
  - HateBR: amostra balanceada (n neutros + n ofensivos, semente fixa).
  - Frases próprias: as 35 do avaliacao.py (15 NÃO e 20 SIM).

Para cada modelo, guarda a pontuação de cada categoria em resultados_modelos/
(pode ser interrompido e retomado depois) e então calcula:
  - AUC (área sob a curva ROC) com intervalo de confiança por bootstrap.
    A AUC não depende de limiar, então é a forma justa de comparar modelos
    com escalas de pontuação diferentes.
  - O limiar ideal de cada modelo (critério de Youden, nos dois conjuntos
    juntos) e, nesse limiar, falsos positivos e detecção em cada conjunto.
  - Tempo mediano por texto.

Duas formulações podem ser testadas:
  atual : rótulos longos, como no analyzer.py
  curta : rótulos curtos + um rótulo "neutro". Para ela também é calculada a
          pontuação "margem" = (maior categoria) - (pontuação do neutro).

Uso:
  python comparar_modelos.py                                  # 4 modelos, formulação atual
  python comparar_modelos.py --n 30                           # amostra menor (mais rápido)
  python comparar_modelos.py --formulacoes atual curta        # as duas formulações
  python comparar_modelos.py --modelos xlmr mdeberta          # só alguns modelos
  python comparar_modelos.py --so-analise                     # só tabela e gráfico
  python comparar_modelos.py --refazer                        # descarta resultados salvos

As categorias são lidas do analyzer.py (sem carregar os modelos dele).
"""

import argparse
import ast
import csv
import gc
import os
import random
import statistics
import sys
import time

import numpy as np

PASTA = "resultados_modelos"
TEMPLATE = "Este texto contém: {}."  # mesmo template do analyzer.py
NEUTRO = "elogio, agradecimento ou conversa comum"

MODELOS = {
    "xlmr": "joeddav/xlm-roberta-large-xnli",
    "mdeberta": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "bge-m3": "MoritzLaurer/bge-m3-zeroshot-v2.0",
    "mdeberta-2mil7": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
}

ROTULOS_CURTOS = {
    "xingamento": "xingamento ou insulto",
    "humilhação": "humilhação",
    "ameaça": "ameaça",
    "gaslighting": "negar a realidade da pessoa",
    "controle": "controle sobre a pessoa",
    "chantagem_emocional": "chantagem emocional",
    "invalidação": "desprezo pelos sentimentos da pessoa",
}

PROPRIO_NAO = [
    "Tenho muito orgulho de você.", "Você é uma pessoa muito especial para mim.",
    "Obrigado por estar ao meu lado.", "Eu gosto muito da sua companhia.",
    "Você fez um ótimo trabalho hoje.", "Parabéns pela sua dedicação.",
    "Fico feliz quando conversamos.", "Você é importante para mim.",
    "Pode contar comigo sempre que precisar.", "Eu admiro muito sua coragem.",
    "Você me ajudou muito hoje.", "Gosto de conversar com você.",
    "Você é uma pessoa incrível.", "Estou feliz por você.", "Você merece coisas boas.",
]
PROPRIO_SIM = [
    "Você está exagerando de novo.", "Isso é coisa da sua cabeça.",
    "Você sempre entende tudo errado.", "Não sei como você consegue ser tão difícil.",
    "Você faz drama por qualquer coisa.", "Ninguém aguenta esse seu jeito.",
    "Você deveria parar de reclamar tanto.",
    "Sempre que algo dá errado, você está envolvida.",
    "Você me deixa sem paciência.", "Talvez o problema seja você.",
    "Você é inútil e nunca faz nada certo.",
    "Sem minha permissão você não vai sair de casa.",
    "Se você me deixar, eu acabo com a sua vida.", "Ninguém vai acreditar em você.",
    "Você é louca, isso nunca aconteceu.", "Eu controlo com quem você fala.",
    "Você não presta para nada.",
    "Se você contar isso para alguém, vai se arrepender.",
    "A culpa é sua por eu agir assim.", "Você não tem direito de decidir nada sozinha.",
]


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar_categorias(caminho="analyzer.py"):
    """Lê o dicionário CATEGORIAS do analyzer.py sem executar o arquivo."""
    with open(caminho, encoding="utf-8") as f:
        arvore = ast.parse(f.read())
    for no in arvore.body:
        if isinstance(no, ast.Assign) and any(getattr(t, "id", None) == "CATEGORIAS" for t in no.targets):
            return ast.literal_eval(no.value)
    print(f"Não encontrei CATEGORIAS em {caminho}.")
    sys.exit(1)


def carregar_hatebr(caminho, n, seed=42):
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f))
    ofensivos = [(l["comentario"].strip(), 1) for l in linhas if l["label_final"].strip() == "1"]
    neutros = [(l["comentario"].strip(), 0) for l in linhas if l["label_final"].strip() == "0"]
    random.seed(seed)
    return random.sample(ofensivos, n) + random.sample(neutros, n)


def montar_itens(csv_hatebr, n):
    itens = [("hatebr", r, t) for t, r in carregar_hatebr(csv_hatebr, n)]
    itens += [("proprio", 0, t) for t in PROPRIO_NAO] + [("proprio", 1, t) for t in PROPRIO_SIM]
    return itens


def carregar_spacy():
    import spacy
    try:
        return spacy.load("pt_core_news_lg")
    except OSError:
        try:
            return spacy.load("pt_core_news_sm")
        except OSError:
            print(
                "[Aviso] Modelos spaCy de português não encontrados. "
                "Usando tokenização e segmentação básicas."
            )
            nlp = spacy.blank("pt")
            nlp.add_pipe("sentencizer")
            return nlp


def arquivo_scores(alias, formulacao):
    return os.path.join(PASTA, f"scores__{alias}__{formulacao}.csv")


# ---------------------------------------------------------------------------
# Coleta (lenta)
# ---------------------------------------------------------------------------
def listar_pendentes(alias, formulacao, itens, refazer):
    caminho = arquivo_scores(alias, formulacao)
    if refazer and os.path.exists(caminho):
        os.remove(caminho)
    feitos = set()
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8-sig", newline="") as f:
            feitos = {(l["fonte"], l["rotulo"], l["texto"]) for l in csv.DictReader(f)}
    return [i for i in itens if (i[0], str(i[1]), i[2].replace("\n", " ")) not in feitos]


def rodar(clf, alias, formulacao, pendentes, categorias, nlp):
    chaves = list(categorias)
    if formulacao == "curta":
        rotulos = [ROTULOS_CURTOS.get(c, categorias[c]) for c in chaves] + [NEUTRO]
    else:
        rotulos = [categorias[c] for c in chaves]

    caminho = arquivo_scores(alias, formulacao)
    os.makedirs(PASTA, exist_ok=True)
    novo = not os.path.exists(caminho)
    print(f"[{alias}/{formulacao}] {len(pendentes)} textos para analisar...")

    with open(caminho, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["fonte", "rotulo", "texto"] + chaves + ["neutro", "tempo_s"])
        for i, (fonte, rotulo, texto) in enumerate(pendentes, start=1):
            limpo = texto.replace("\n", " ")
            inicio = time.perf_counter()
            sentencas = [s.text.strip() for s in nlp(limpo).sents if s.text.strip()] or [limpo]
            resultado = clf(sentencas, candidate_labels=rotulos,
                            hypothesis_template=TEMPLATE, multi_label=True)
            if isinstance(resultado, dict):
                resultado = [resultado]
            melhor = {r: 0.0 for r in rotulos}
            for r in resultado:
                for rotulo_hf, score in zip(r["labels"], r["scores"]):
                    melhor[rotulo_hf] = max(melhor[rotulo_hf], score)
            tempo = time.perf_counter() - inicio

            valores = [f"{melhor[r]:.4f}" for r in rotulos[:len(chaves)]]
            neutro = f"{melhor[NEUTRO]:.4f}" if formulacao == "curta" else ""
            w.writerow([fonte, rotulo, limpo] + valores + [neutro, f"{tempo:.2f}"])
            f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(pendentes)}")


def coletar(aliases, formulacoes, itens, categorias, refazer):
    import torch
    from transformers import pipeline

    nlp = None
    for alias in aliases:
        pendentes = {f: listar_pendentes(alias, f, itens, refazer) for f in formulacoes}
        if not any(pendentes.values()):
            print(f"[{alias}] tudo já coletado, pulando.")
            continue

        if nlp is None:
            nlp = carregar_spacy()
        print(f"\n[{alias}] carregando {MODELOS[alias]} (na primeira vez baixa o modelo)...")
        dispositivo = 0 if torch.cuda.is_available() else -1
        clf = pipeline("zero-shot-classification", model=MODELOS[alias], device=dispositivo)

        for formulacao in formulacoes:
            if pendentes[formulacao]:
                rodar(clf, alias, formulacao, pendentes[formulacao], categorias, nlp)

        del clf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Análise (rápida)
# ---------------------------------------------------------------------------
def ler_scores(caminho, chaves):
    dados = []
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        for l in csv.DictReader(f):
            dados.append({
                "fonte": l["fonte"],
                "rotulo": int(l["rotulo"]),
                "max": max(float(l[c]) for c in chaves),
                "neutro": float(l["neutro"]) if l["neutro"] else None,
                "tempo": float(l["tempo_s"]),
            })
    return dados


def auc_ic(y, s, repeticoes=500, semente=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(semente)
    aucs = []
    for _ in range(repeticoes):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def avaliar(dados, tipo):
    from sklearn.metrics import roc_auc_score, roc_curve

    y, s = {}, {}
    for fonte in ("hatebr", "proprio"):
        y[fonte] = np.array([d["rotulo"] for d in dados if d["fonte"] == fonte])
        s[fonte] = np.array([d["max"] if tipo == "max" else d["max"] - d["neutro"]
                             for d in dados if d["fonte"] == fonte])
        if len(set(y[fonte])) < 2:
            return None

    yp = np.concatenate([y["hatebr"], y["proprio"]])
    sp = np.concatenate([s["hatebr"], s["proprio"]])
    fpr, tpr, limiares = roc_curve(yp, sp)
    limiar = float(limiares[int(np.argmax(tpr - fpr))])

    def taxa(fonte, rotulo):
        return float(np.mean(s[fonte][y[fonte] == rotulo] >= limiar))

    saida = {
        "limiar": limiar,
        "fp_hatebr": taxa("hatebr", 0), "rec_hatebr": taxa("hatebr", 1),
        "fp_proprio": taxa("proprio", 0), "rec_proprio": taxa("proprio", 1),
        "tempo": statistics.median(d["tempo"] for d in dados),
        "curvas": {},
    }
    for fonte in y:
        saida["auc_" + fonte] = float(roc_auc_score(y[fonte], s[fonte]))
        saida["ic_" + fonte] = auc_ic(y[fonte], s[fonte])
        f, t, _ = roc_curve(y[fonte], s[fonte])
        saida["curvas"][fonte] = (f, t)
    return saida


def analisar(categorias, aliases, formulacoes):
    chaves = list(categorias)
    linhas = []

    for alias in aliases:
        for formulacao in formulacoes:
            caminho = arquivo_scores(alias, formulacao)
            if not os.path.exists(caminho):
                print(f"(sem resultados para {alias}/{formulacao})")
                continue
            dados = ler_scores(caminho, chaves)
            for tipo in ["max"] + (["margem"] if formulacao == "curta" else []):
                r = avaliar(dados, tipo)
                if r:
                    linhas.append((alias, formulacao, tipo, r))

    if not linhas:
        print("Nada para analisar ainda.")
        return

    def fmt_auc(r, fonte):
        lo, hi = r["ic_" + fonte]
        return f"{r['auc_' + fonte]:.2f} [{lo:.2f}-{hi:.2f}]"

    print("\n" + "=" * 130)
    print(f"{'modelo':15}{'form.':7}{'score':8}{'AUC HateBR [IC95%]':22}{'AUC próprio [IC95%]':22}"
          f"{'limiar':8}{'FP Hate':9}{'Rec Hate':10}{'FP próp.':10}{'Rec próp.':10}{'s/texto':8}")
    print("-" * 130)
    for alias, formulacao, tipo, r in linhas:
        print(f"{alias:15}{formulacao:7}{tipo:8}{fmt_auc(r, 'hatebr'):22}{fmt_auc(r, 'proprio'):22}"
              f"{r['limiar']:<8.2f}{r['fp_hatebr']:<9.0%}{r['rec_hatebr']:<10.0%}"
              f"{r['fp_proprio']:<10.0%}{r['rec_proprio']:<10.0%}{r['tempo']:<8.1f}")
    print("=" * 130)
    print("AUC: 0,50 = acaso; 1,00 = separação perfeita. Se os intervalos de dois modelos se")
    print("sobrepõem, a diferença não é conclusiva (as amostras são pequenas).")
    print("Limiar: escolhido por modelo (critério de Youden, HateBR + próprio juntos).")
    print("FP Hate = % de comentários neutros do HateBR marcados como violência (quer baixo).")
    print("Rec Hate = % de ofensivos do HateBR detectados (quer alto).")
    print("FP próp. = % das 15 frases NÃO marcadas (baixo). Rec próp. = % das 20 frases SIM detectadas (alto).")
    print("As frases próprias foram escritas pela autora: tendem a ser otimistas.")

    with open("comparacao_modelos.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "formulacao", "score", "auc_hatebr", "ic_lo_hatebr", "ic_hi_hatebr",
                    "auc_proprio", "ic_lo_proprio", "ic_hi_proprio", "limiar", "fp_hatebr",
                    "rec_hatebr", "fp_proprio", "rec_proprio", "segundos_por_texto"])
        for alias, formulacao, tipo, r in linhas:
            w.writerow([alias, formulacao, tipo,
                        f"{r['auc_hatebr']:.4f}", f"{r['ic_hatebr'][0]:.4f}", f"{r['ic_hatebr'][1]:.4f}",
                        f"{r['auc_proprio']:.4f}", f"{r['ic_proprio'][0]:.4f}", f"{r['ic_proprio'][1]:.4f}",
                        f"{r['limiar']:.4f}", f"{r['fp_hatebr']:.4f}", f"{r['rec_hatebr']:.4f}",
                        f"{r['fp_proprio']:.4f}", f"{r['rec_proprio']:.4f}", f"{r['tempo']:.2f}"])
    print("\nTabela salva em comparacao_modelos.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
        for eixo, fonte, titulo in zip(eixos, ("hatebr", "proprio"),
                                       ("HateBR (neutro x ofensivo)", "Frases próprias (NÃO x SIM)")):
            for alias, formulacao, tipo, r in linhas:
                f, t = r["curvas"][fonte]
                eixo.plot(f, t, label=f"{alias}/{formulacao}/{tipo} (AUC {r['auc_' + fonte]:.2f})")
            eixo.plot([0, 1], [0, 1], "k--", linewidth=0.8)
            eixo.set_title(titulo)
            eixo.set_xlabel("Taxa de falsos positivos")
            eixo.set_ylabel("Taxa de detecção")
            eixo.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig("curvas_roc.png", dpi=150)
        print("Gráfico salvo em curvas_roc.png")
    except ImportError:
        print("matplotlib não instalado: gráfico não gerado.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/HateBR.csv")
    parser.add_argument("--n", type=int, default=50, help="comentários por classe do HateBR")
    parser.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=list(MODELOS))
    parser.add_argument("--formulacoes", nargs="+", default=["atual"], choices=["atual", "curta"])
    parser.add_argument("--so-analise", action="store_true")
    parser.add_argument("--refazer", action="store_true")
    args = parser.parse_args()

    categorias = carregar_categorias()

    if not args.so_analise:
        itens = montar_itens(args.csv, args.n)
        coletar(args.modelos, args.formulacoes, itens, categorias, args.refazer)

    analisar(categorias, args.modelos, args.formulacoes)


if __name__ == "__main__":
    main()