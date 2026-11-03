"""
diagnostico_erros.py

Lê o erros_hatebr.csv (gerado pelo avaliacaohatebr.py) e mostra, para cada
falso positivo, qual categoria o analisador detectou e com que confiança.
Serve para descobrir se uma categoria específica está causando a maioria
dos alarmes falsos.

Uso:
  python diagnostico_erros.py
"""

import csv
from collections import Counter

from analyzer import analisar_texto

with open("erros_hatebr.csv", encoding="utf-8-sig", newline="") as f:
    falsos_positivos = [l for l in csv.DictReader(f) if l["tipo_erro"] == "falso_positivo"]

print(f"\n{len(falsos_positivos)} falsos positivos encontrados.\n")

contagem = Counter()

for linha in falsos_positivos:
    r = analisar_texto(linha["texto"])
    tipos = r.get("tipos_detectados") or r.get("tipos_identificados") or []
    contagem.update(tipos)
    print(f"{r['nivel_risco']:7} confiança={r.get('score_confianca')}  categorias={tipos}")
    print(f"        {linha['texto'][:90]}\n")

print("=== Categorias que mais geraram falso positivo ===")
for categoria, qtd in contagem.most_common():
    print(f"{categoria}: {qtd}")