"""
testar_modelo.py
Testa um modelo treinado (fine-tuning) com frases suas.

Uso:
  python testar_modelo.py "Vou te perseguir até o fim do mundo se não mandar a foto"
  python testar_modelo.py --modelo resultados_modelos/combinado_treinado "outra frase"
  python testar_modelo.py            (modo interativo; linha vazia para sair)
"""

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("frases", nargs="*", help="frases a classificar")
parser.add_argument("--modelo", default="resultados_modelos/hatebr_treinado")
parser.add_argument("--limiar", type=float, default=0.5, help="prob. mínima para alertar")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.modelo)
model = AutoModelForSequenceClassification.from_pretrained(args.modelo)
model.eval()


def classificar(frase: str) -> float:
    entrada = tokenizer(frase, return_tensors="pt", truncation=True, max_length=256)
    entrada.pop("token_type_ids", None)
    with torch.no_grad():
        return torch.softmax(model(**entrada).logits, dim=-1)[0, 1].item()


def mostrar(frase: str):
    p = classificar(frase)
    rotulo = "ALERTA" if p >= args.limiar else "ok    "
    print(f"{rotulo}  {p:.2f}  {frase}")


if args.frases:
    for f in args.frases:
        mostrar(f)
else:
    print(f"Modelo: {args.modelo} (linha vazia para sair)")
    while True:
        f = input("> ").strip()
        if not f:
            break
        mostrar(f)
