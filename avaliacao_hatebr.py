"""
avaliacao_hatebr.py

Avalia o analisador em uma amostra balanceada do HateBR
(comentários do Instagram anotados como ofensivos / não ofensivos).

Mapeamento:
  veredicto SIM ou POSSIVEL  -> previsto "ofensivo" (1)
  veredicto NAO              -> previsto "não ofensivo" (0)

Uso:
  python avaliacao_hatebr.py                 # 100 por classe (200 no total)
  python avaliacao_hatebr.py --n 50          # 50 por classe (mais rápido)
  python avaliacao_hatebr.py --csv dataset/HateBR.csv

Atenção: o HateBR mede linguagem ofensiva em comentários públicos, não
violência psicológica em relacionamento. É uma validação parcial.
"""

import argparse
import csv
import random
import sys

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from analyzer import analisar_texto

COLUNAS_TEXTO = ["instagram_comments", "comentario", "comentário", "comment", "texto", "text"]
COLUNAS_ROTULO = ["label_final", "offensive_language", "offensive", "ofensivo", "label", "rotulo", "rótulo"]


def _achar_coluna(cabecalho, candidatas):
    minusculas = {c.strip().lower(): c for c in cabecalho}
    for nome in candidatas:
        if nome in minusculas:
            return minusculas[nome]
    return None


def _rotulo_para_int(valor):
    v = str(valor).strip().lower()
    if v in ("1", "true", "sim", "yes", "offensive", "ofensivo"):
        return 1
    if v in ("0", "false", "nao", "não", "no", "non-offensive", "nao ofensivo"):
        return 0
    return None


def carregar_dataset(caminho):
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        leitor = csv.DictReader(f)
        cabecalho = leitor.fieldnames or []
        col_texto = _achar_coluna(cabecalho, COLUNAS_TEXTO)
        col_rotulo = _achar_coluna(cabecalho, COLUNAS_ROTULO)

        if not col_texto or not col_rotulo:
            print(f"Não identifiquei as colunas. Colunas encontradas: {cabecalho}")
            print("Ajuste COLUNAS_TEXTO / COLUNAS_ROTULO no topo do script.")
            sys.exit(1)

        print(f"Coluna de texto: {col_texto} | Coluna de rótulo: {col_rotulo}")

        itens = []
        for linha in leitor:
            rotulo = _rotulo_para_int(linha[col_rotulo])
            texto = (linha[col_texto] or "").strip()
            if rotulo is not None and texto:
                itens.append((texto, rotulo))
    return itens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/HateBR.csv")
    parser.add_argument("--n", type=int, default=100, help="amostras por classe")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    itens = carregar_dataset(args.csv)
    ofensivos = [i for i in itens if i[1] == 1]
    neutros = [i for i in itens if i[1] == 0]
    print(f"Dataset: {len(itens)} comentários ({len(ofensivos)} ofensivos, {len(neutros)} não ofensivos)")

    random.seed(args.seed)
    n = min(args.n, len(ofensivos), len(neutros))
    amostra = random.sample(ofensivos, n) + random.sample(neutros, n)
    random.shuffle(amostra)

    y_true, y_pred, erros = [], [], []
    for i, (texto, esperado) in enumerate(amostra, start=1):
        veredicto = analisar_texto(texto)["veredicto"]
        previsto = 0 if veredicto == "NAO" else 1
        y_true.append(esperado)
        y_pred.append(previsto)
        if previsto != esperado:
            erros.append((texto, esperado, previsto))
        if i % 20 == 0:
            print(f"  {i}/{len(amostra)} analisados...")

    print("\n=== RESULTADOS (HateBR, amostra balanceada) ===")
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.2%}\n")
    print(classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["Não ofensivo", "Ofensivo"], zero_division=0,
    ))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f"Verdadeiros negativos: {tn} | Falsos positivos: {fp}")
    print(f"Falsos negativos:      {fn} | Verdadeiros positivos: {tp}")

    with open("erros_hatebr.csv", "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["texto", "esperado", "previsto", "tipo_erro"])
        for texto, esperado, previsto in erros:
            tipo = "falso_positivo" if previsto == 1 else "falso_negativo"
            escritor.writerow([texto, esperado, previsto, tipo])
    print(f"\n{len(erros)} erros salvos em erros_hatebr.csv (use pra analisar os casos).")


if __name__ == "__main__":
    main()