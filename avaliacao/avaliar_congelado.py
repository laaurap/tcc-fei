"""
avaliar_congelado.py
Avalia um modelo treinado no teste congelado (dataset/teste_congelado.csv)
e lista os erros.

Uso:
  python avaliar_congelado.py --modelo resultados_modelos/combinado_treinado
  python avaliar_congelado.py --modelo resultados_modelos/hatebr_treinado --limiar 0.7
"""

import argparse

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--modelo", default="resultados_modelos/combinado_treinado")
parser.add_argument("--limiar", type=float, default=0.5)
parser.add_argument("--csv", default="dataset/teste_congelado.csv")
args = parser.parse_args()

df = pd.read_csv(args.csv, encoding="utf-8-sig")
tokenizer = AutoTokenizer.from_pretrained(args.modelo)
model = AutoModelForSequenceClassification.from_pretrained(args.modelo)
model.eval()

probs = []
for texto in df["texto"]:
    ent = tokenizer(texto, return_tensors="pt", truncation=True, max_length=256)
    ent.pop("token_type_ids", None)
    with torch.no_grad():
        probs.append(torch.softmax(model(**ent).logits, dim=-1)[0, 1].item())
df["prob"] = probs
df["previsto"] = (df["prob"] >= args.limiar).astype(int)

y, p = df["rotulo"], df["previsto"]
prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

print(f"\nModelo: {args.modelo} | limiar: {args.limiar} | frases: {len(df)}")
print(f"Acurácia: {accuracy_score(y, p):.3f} | Precisão: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
print(f"Falsos positivos (alerta sem motivo): {fp} | Falsos negativos (violência não detectada): {fn}")

print("\nAcerto por categoria:")
df["acertou"] = (df["rotulo"] == df["previsto"])
print(df.groupby("categoria")["acertou"].agg(["mean", "count"]).round(2).to_string())

erros = df[~df["acertou"]].sort_values("prob")
print(f"\nErros ({len(erros)}):")
for _, r in erros.iterrows():
    tipo = "FALSO NEGATIVO" if r["rotulo"] == 1 else "FALSO POSITIVO"
    print(f"  {tipo}  prob={r['prob']:.2f}  [{r['categoria']}]  {r['texto']}")

df.to_csv("resultados_modelos/teste_congelado_resultado.csv", index=False, encoding="utf-8-sig")
