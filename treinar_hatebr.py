
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
SAIDA = "resultados_modelos/hatebr_treinado"

# 1. Carregar os dados
df = pd.read_csv("dataset/HateBR.csv", encoding="utf-8-sig")

df = df[["comentario", "label_final"]].dropna()
df["comentario"] = df["comentario"].astype(str).str.strip()
df["label_final"] = pd.to_numeric(
    df["label_final"], errors="coerce"
)
df = df[df["label_final"].isin([0, 1])]
df = df[df["comentario"] != ""]

# Remover comentários repetidos e rótulos conflitantes
df = df.groupby("comentario", as_index=False).filter(
    lambda grupo: grupo["label_final"].nunique() == 1
)
df = df.drop_duplicates(subset="comentario")

df = df.rename(columns={
    "comentario": "texto",
    "label_final": "labels"
})
df["labels"] = df["labels"].astype(int)

print("Exemplos disponíveis:", len(df))
print(df["labels"].value_counts())

# 2. Dividir: 70% treino, 15% validação, 15% teste
treino, restante = train_test_split(
    df, test_size=0.30, random_state=42,
    stratify=df["labels"]
)

validacao, teste = train_test_split(
    restante, test_size=0.50, random_state=42,
    stratify=restante["labels"]
)

# 3. Tokenização
tokenizer = AutoTokenizer.from_pretrained(MODELO)

def tokenizar(exemplos):
    return tokenizer(
        exemplos["texto"],
        truncation=True,
        max_length=256
    )

def preparar(dados):
    dataset = Dataset.from_pandas(
        dados.reset_index(drop=True),
        preserve_index=False
    )
    return dataset.map(
        tokenizar, batched=True,
        remove_columns=["texto"]
    )

ds_treino = preparar(treino)
ds_validacao = preparar(validacao)
ds_teste = preparar(teste)

# 4. Modelo com duas classes
model = AutoModelForSequenceClassification.from_pretrained(
    MODELO,
    num_labels=2,
    id2label={0: "NAO_OFENSIVO", 1: "OFENSIVO"},
    label2id={"NAO_OFENSIVO": 0, "OFENSIVO": 1}
)

# 5. Métricas
def calcular_metricas(resultado):
    logits, rotulos = resultado
    previsoes = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        rotulos, previsoes,
        average="binary", zero_division=0
    )

    return {
        "accuracy": accuracy_score(rotulos, previsoes),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 6. Configuração de treinamento
args = TrainingArguments(
    output_dir=SAIDA + "/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
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
    seed=42
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_treino,
    eval_dataset=ds_validacao,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=calcular_metricas
)

# 7. Executar o fine-tuning
trainer.train()

# 8. Avaliar no teste separado
print("\nRESULTADOS FINAIS:")
print(trainer.evaluate(ds_teste))

# 9. Salvar modelo treinado
trainer.save_model(SAIDA)
tokenizer.save_pretrained(SAIDA)

print("\nModelo salvo em:", SAIDA)