"""
avaliacao.py

Avaliação binária do sistema:
- NAO = não há indício de violência psicológica
- SIM = há indício de violência psicológica
"""

from analyzer import analisar_texto

from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report
)

import matplotlib.pyplot as plt


# =========================================================
# BASE DE TESTE BINÁRIA
# =========================================================

textos = [
    # CLASSE: NAO
    "Tenho muito orgulho de você.",
    "Você é uma pessoa muito especial para mim.",
    "Obrigado por estar ao meu lado.",
    "Eu gosto muito da sua companhia.",
    "Você fez um ótimo trabalho hoje.",
    "Parabéns pela sua dedicação.",
    "Fico feliz quando conversamos.",
    "Você é importante para mim.",
    "Pode contar comigo sempre que precisar.",
    "Eu admiro muito sua coragem.",
    "Você me ajudou muito hoje.",
    "Gosto de conversar com você.",
    "Você é uma pessoa incrível.",
    "Estou feliz por você.",
    "Você merece coisas boas.",

    # CLASSE: SIM
    "Você está exagerando de novo.",
    "Isso é coisa da sua cabeça.",
    "Você sempre entende tudo errado.",
    "Não sei como você consegue ser tão difícil.",
    "Você faz drama por qualquer coisa.",
    "Ninguém aguenta esse seu jeito.",
    "Você deveria parar de reclamar tanto.",
    "Sempre que algo dá errado, você está envolvida.",
    "Você me deixa sem paciência.",
    "Talvez o problema seja você.",
    "Você é inútil e nunca faz nada certo.",
    "Sem minha permissão você não vai sair de casa.",
    "Se você me deixar, eu acabo com a sua vida.",
    "Ninguém vai acreditar em você.",
    "Você é louca, isso nunca aconteceu.",
    "Eu controlo com quem você fala.",
    "Você não presta para nada.",
    "Se você contar isso para alguém, vai se arrepender.",
    "A culpa é sua por eu agir assim.",
    "Você não tem direito de decidir nada sozinha."
]


# =========================================================
# RESPOSTAS CORRETAS
# =========================================================

y_true = [
    # NAO
    "NAO", "NAO", "NAO", "NAO", "NAO",
    "NAO", "NAO", "NAO", "NAO", "NAO",
    "NAO", "NAO", "NAO", "NAO", "NAO",

    # SIM
    "SIM", "SIM", "SIM", "SIM", "SIM",
    "SIM", "SIM", "SIM", "SIM", "SIM",
    "SIM", "SIM", "SIM", "SIM", "SIM",
    "SIM", "SIM", "SIM", "SIM", "SIM"
]


# =========================================================
# EXECUÇÃO DO MODELO
# =========================================================

y_pred = []

print("\n===================================================")
print("AVALIAÇÃO BINÁRIA DO MODELO")
print("===================================================\n")

for texto in textos:
    resultado = analisar_texto(texto)

    previsao = resultado["veredicto"]

    # Como agora só existem duas classes:
    # qualquer POSSIVEL será considerado SIM
    if previsao == "POSSIVEL":
        previsao = "SIM"

    y_pred.append(previsao)


# =========================================================
# RESULTADOS INDIVIDUAIS
# =========================================================

print("\n===================================================")
print("RESULTADOS INDIVIDUAIS")
print("===================================================")

for texto, esperado, previsto in zip(textos, y_true, y_pred):
    status = "✅" if esperado == previsto else "❌"

    print("\n---------------------------------------------------")
    print(f"{status} Texto:     {texto}")
    print(f"   Esperado: {esperado}")
    print(f"   Previsto: {previsto}")


# =========================================================
# ACURÁCIA
# =========================================================

acc = accuracy_score(y_true, y_pred)

print("\n===================================================")
print("ACURÁCIA")
print("===================================================\n")
print(f"Acurácia do modelo: {acc:.2%}")


# =========================================================
# RELATÓRIO DE CLASSIFICAÇÃO
# =========================================================

print("\n===================================================")
print("RELATÓRIO DE CLASSIFICAÇÃO")
print("===================================================\n")

print(classification_report(
    y_true,
    y_pred,
    labels=["NAO", "SIM"],
    target_names=["NÃO", "SIM"],
    zero_division=0
))


# =========================================================
# MATRIZ DE CONFUSÃO
# =========================================================

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=["NAO", "SIM"],
    display_labels=["NÃO", "SIM"],
    cmap="Blues"
)

plt.title("Matriz de Confusão - Violência Psicológica")
plt.tight_layout()
plt.show()