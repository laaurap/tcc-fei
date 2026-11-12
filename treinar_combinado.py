"""
treinar_combinado.py
Fine-tuning do DistilBERT multilíngue com HateBR + dataset próprio
(dataset/violencia_psicologica_proprio.csv, gerado por gerar_dataset_proprio.py).

O dataset próprio é pequeno e carrega os casos que o HateBR não tem
(ameaça, controle, gaslighting, hipérbole inofensiva...), então ele é
repetido SUPERAMOSTRAGEM vezes no treino. Validação e teste NÃO são repetidos.
Os testes são avaliados separadamente (HateBR / próprio) para não mascarar nada.
"""

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

MODELO = "distilbert/distilbert-base-multilingual-cased"
SAIDA = "resultados_modelos/combinado_treinado"
SUPERAMOSTRAGEM = 5

FRASES_VERIFICACAO = [
    "Vou te perseguir até o fim do mundo se você não me mandar a foto desse cachorro agora",
    "Vou te perseguir até o fim do mundo se você me deixar",
    "Isso tá me matando de rir",
    "Você é uma inútil, ninguém te aguenta",
    "Você não vai sair hoje, eu proibi",
    "Isso nunca aconteceu, você está inventando coisas",
]


def carregar_hatebr():
    df = pd.read_csv("dataset/HateBR.csv", encoding="utf-8-sig")
    df = df[["comentario", "label_final"]].dropna()
    df["comentario"] = df["comentario"].astype(str).str.strip()
    df["label_final"] = pd.to_numeric(df["label_final"], errors="coerce")
    df = df[df["label_final"].isin([0, 1]) & (df["comentario"] != "")]
    df = df.groupby("comentario", as_index=False).filter(
        lambda g: g["label_final"].nunique() == 1
    )
    df = df.drop_duplicates(subset="comentario")
    df = df.rename(columns={"comentario": "texto", "label_final": "labels"})
    df["labels"] = df["labels"].astype(int)
    return df[["texto", "labels"]]


def carregar_proprio():
    df = pd.read_csv("dataset/violencia_psicologica_proprio.csv", encoding="utf-8-sig")
    df = df[["texto", "rotulo"]].dropna()
    df["texto"] = df["texto"].astype(str).str.strip()
    df = df.drop_duplicates(subset="texto")
    df = df.rename(columns={"rotulo": "labels"})
    df["labels"] = df["labels"].astype(int)
    return df[["texto", "labels"]]


def dividir(df):
    treino, resto = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["labels"]
    )
    val, teste = train_test_split(
        resto, test_size=0.50, random_state=42, stratify=resto["labels"]
    )
    return treino, val, teste


hate = carregar_hatebr()
proprio = carregar_proprio()
print("HateBR:", len(hate), "| próprio:", len(proprio))

h_tr, h_va, h_te = dividir(hate)
p_tr, p_va, p_te = dividir(proprio)

treino = pd.concat([h_tr] + [p_tr] * SUPERAMOSTRAGEM).sample(frac=1, random_state=42)
validacao = pd.concat([h_va, p_va])

tokenizer = AutoTokenizer.from_pretrained(MODELO)


def preparar(dados):
    ds = Dataset.from_pandas(dados.reset_index(drop=True), preserve_index=False)
    return ds.map(
        lambda ex: tokenizer(ex["texto"], truncation=True, max_length=256),
        batched=True,
        remove_columns=["texto"],
    )


ds_treino, ds_val = preparar(treino), preparar(validacao)
ds_te_hate, ds_te_proprio = preparar(h_te), preparar(p_te)

model = AutoModelForSequenceClassification.from_pretrained(
    MODELO,
    num_labels=2,
    id2label={0: "NAO_OFENSIVO", 1: "OFENSIVO"},
    label2id={"NAO_OFENSIVO": 0, "OFENSIVO": 1},
)


def calcular_metricas(resultado):
    logits, rotulos = resultado
    prev = np.argmax(logits, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        rotulos, prev, average="binary", zero_division=0
    )
    return {"accuracy": accuracy_score(rotulos, prev), "precision": p, "recall": r, "f1": f1}


args = TrainingArguments(
    output_dir=SAIDA + "/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_treino,
    eval_dataset=ds_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=calcular_metricas,
)

trainer.train()

print("\nRESULTADOS FINAIS (teste HateBR):")
print(trainer.evaluate(ds_te_hate))
print("\nRESULTADOS FINAIS (teste próprio):")
print(trainer.evaluate(ds_te_proprio))

trainer.save_model(SAIDA)
tokenizer.save_pretrained(SAIDA)

# Verificação rápida com frases-chave
model.eval()
print("\nVERIFICAÇÃO COM FRASES-CHAVE (prob. de ser violência/ofensa):")
for frase in FRASES_VERIFICACAO:
    entrada = tokenizer(frase, return_tensors="pt", truncation=True, max_length=256)
    entrada = {k: v.to(model.device) for k, v in entrada.items()}
    with torch.no_grad():
        prob = torch.softmax(model(**entrada).logits, dim=-1)[0, 1].item()
    print(f"  {prob:.2f}  {frase}")

print("\nModelo salvo em:", SAIDA)
