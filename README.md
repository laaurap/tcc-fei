# **Agora vai 💥💥**
Oi Lauraaaa, organizei o projeto em pastas pq tava ficando uma loucura de mexer, ali em baixo a IA resumiu o que tem em cada parte, mas as coisas que você precisa usar estão todas na `src/`. Pedi também pra ela deixar as instruções pra executar o projeto.

## Sistema de Detecção de Violência Psicológica em Conversas (TCC — FEI)

Sistema de Processamento de Linguagem Natural (NLP) para identificação de padrões de **violência psicológica** em conversas de texto e áudio (focado em WhatsApp), utilizando uma arquitetura híbrida que combina:

1. **Transcrição de áudio:** `Faster-Whisper` para converter mensagens de voz em texto.
2. **Camada de Contexto (`src/contexto.py`):** neutralização de expressões figuradas/idiomáticas (ex.: *"matando de rir"*) e detecção de intenção dialógica.
3. **Classificação Zero-Shot (`XLM-RoBERTa-large-XNLI`):** identificação granular de 7 categorias de violência psicológica (*xingamento, humilhação, ameaça, gaslighting, controle, chantagem emocional e invalidação*).
4. **Classificador Supervisionado Fine-Tuned (`DistilBERT Multilíngue`):** modelo ajustado com o dataset **HateBR** + dataset próprio de contextos de violência psicológica e hipérboles cotidianas (`combinado_v3`), atuando como validador para reduzir falsos positivos e calibrar o nível de risco.
5. **RAG (`Sentence-Transformers`):** busca semântica em base de conhecimento técnico-jurídica (Código Penal Art. 147-B, Lei Maria da Penha e literatura acadêmica) para fundamentar os alertas.

---

## 📁 Estrutura de Pastas do Projeto

```text
tcc-fei/
├── src/                          # Núcleo da aplicação (CLI, API, NLP e RAG)
│   ├── main.py                   # Ponto de entrada via linha de comando (CLI)
│   ├── analyzer.py               # Pipeline principal (spaCy + Zero-Shot + Fine-Tuned + RAG)
│   ├── contexto.py               # Neutralização de expressões figuradas e filtro contextual
│   ├── rag.py                    # Recuperação semântica de referências (Sentence-Transformers)
│   ├── knowledge_base.py         # Base de referências legais e psicológicas
│   ├── transcriber.py            # Transcrição de áudio/vídeo com Faster-Whisper
│   ├── report.py                 # Formatação visual de relatórios no terminal (Rich)
│   ├── api.py                    # API REST (FastAPI) para integração com n8n / sistemas externos
│   └── server.py                 # Webhook FastAPI para integração direta com WhatsApp (Twilio)
│
├── treinamento/                  # Scripts de construção de dados e Fine-Tuning
│   ├── treinar_hatebr.py         # Fine-tuning baseline apenas com o dataset HateBR
│   ├── treinar_combinado.py      # Fine-tuning combinado (HateBR + dataset próprio)
│   ├── gerar_dataset_proprio.py  # Gerador do dataset próprio (7 categorias + negativos difíceis)
│   ├── gerar_teste_congelado.py  # Gerador do conjunto de teste congelado (frases inéditas)
│   └── colab_treino.ipynb        # Notebook para execução rápida do treino em GPU no Google Colab
│
├── avaliacao/                    # Scripts de validação, métricas e testes de modelos
│   ├── testar_modelo.py          # Teste rápido (CLI/interativo) dos modelos fine-tuned
│   ├── avaliar_congelado.py      # Avaliação de acurácia, precisão, recall e F1 no teste congelado
│   ├── avaliacao.py              # Avaliação completa do pipeline (analyzer.py) + Matriz de Confusão
│   ├── avaliacao_hatebr.py       # Avaliação do pipeline em amostras do HateBR
│   ├── comparar_modelos.py       # Benchmark entre diferentes modelos Zero-Shot
│   └── diagnostico_erros.py      # Inspeção detalhada de falsos positivos/negativos no HateBR
│
├── dataset/                      # Bases de dados (CSV)
│   ├── HateBR.csv                               # Dataset público de discurso de ódio/ofensas em PT-BR
│   ├── violencia_psicologica_proprio.csv        # Dataset próprio de treino/validação (361 frases)
│   ├── teste_congelado.csv                      # Dataset inédito exclusivo para avaliação (69 frases)
│   └── contextos_violencia_psicologica.csv      # Recorte de validação contextual
│
├── resultados_modelos/           # Pesos dos modelos treinados e planilhas de métricas
│   ├── combinado_v3/             # Modelo final fine-tuned em GPU (HateBR + 361 frases próprias, 3 épocas)
│   ├── combinado_treinado/       # Versão intermediária (v2)
│   ├── combinado_v1/             # Primeira versão combinada (v1)
│   ├── hatebr_treinado/          # Modelo baseline treinado exclusivamente no HateBR
│   └── *.csv                     # Resultados consolidados de experimentos e comparações
│
├── tests/                        # Testes unitários automatizados
│   └── test_contexto.py          # Testes das regras de neutralização e contexto não abusivo
│
├── media/                        # Arquivos de áudio/vídeo para teste de transcrição
│   └── imatura_louca.mp4         # Exemplo de mídia para teste com --audio
│
├── requirements.txt              # Dependências completas do projeto
├── requirements_api.txt          # Dependências mínimas para execução da API
└── ngrok-config.yml              # Configuração de túneis do ngrok (API e n8n)
```

---

## 🚀 Como Executar e Testar no Terminal

Execute todos os comandos abaixo no **PowerShell**, sempre a partir da **pasta raiz do projeto (`tcc-fei`)**.

### 1. Ativar o ambiente virtual e instalar as dependências
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

### 2. Testar o Sistema Completo (`src/main.py`)

**a) Caso de hipérbole/brincadeira cotidiana (esperado: `NENHUMA VIOLÊNCIA DETECTADA`):**
```powershell
python src\main.py --texto "Vou te perseguir até o fim do mundo se você não me mandar a foto desse cachorro agora"
```

**b) Caso de ameaça real de perseguição (esperado: `VIOLÊNCIA PSICOLÓGICA DETECTADA - Nível ALTO`):**
```powershell
python src\main.py --texto "Vou te perseguir até o fim do mundo se você me deixar"
```

**c) Caso de controle e gaslighting:**
```powershell
python src\main.py --texto "Você não vai sair hoje, eu proibi. E para de inventar coisas, você está ficando louca."
```

**d) Modo interativo (digite várias linhas e finalize digitando `FIM`):**
```powershell
python src\main.py
```

**e) Testar transcrição e análise de arquivo de áudio/vídeo:**
```powershell
python src\main.py --audio media\imatura_louca.mp4
```

---

### 3. Testar Apenas o Classificador Fine-Tuned (`combinado_v3`) (IA colocou, mas não precisa)

**a) Passando frases diretamente:**
```powershell
python avaliacao\testar_modelo.py --modelo resultados_modelos/combinado_v3 "Vou te perseguir até o fim do mundo se não mandar a foto do cachorro" "Vou te perseguir se você terminar comigo" "Obrigada pela ajuda de hoje!" "Você é uma inútil"
```

**b) Modo interativo rápido (Enter em branco para sair):**
```powershell
python avaliacao\testar_modelo.py --modelo resultados_modelos/combinado_v3
```

---

### 4. Executar Avaliações e Métricas (Isso aqui pode ser legal pro TCC)

**a) Avaliar modelos no Teste Congelado (69 frases inéditas):**
```powershell
# Modelo final (combinado_v3) o que treinamos
python avaliacao\avaliar_congelado.py --modelo resultados_modelos/combinado_v3

# Modelo baseline (apenas HateBR) para comparação
python avaliacao\avaliar_congelado.py --modelo resultados_modelos/hatebr_treinado
#dai vc pode pegar as métricas e mostrar a melhora do modelo sla
```

**b) Avaliar o pipeline completo (`analyzer.py`) e gerar Matriz de Confusão:**
```powershell
python avaliacao\avaliacao.py
```

**c) Executar os testes unitários:**
```powershell
python -m unittest discover -s tests -v
```

---

### 5. Subir a API e Integrar com WhatsApp (Twilio) / n8n (eu não lembro se vc fazia assim mas o robozãoum recomendou)

#### Opção A: Integração Direta com WhatsApp via Twilio (`src/server.py`)

O servidor `src/server.py` recebe mensagens de **texto ou áudio** enviadas pelo WhatsApp, responde imediatamente ao Twilio para evitar *timeout* e processa a análise em segundo plano (`BackgroundTasks`), enviando o relatório de volta ao usuário via API do Twilio.

**1. Configurar o arquivo `.env` na raiz do projeto (`tcc-fei/.env`):**
Crie um arquivo `.env` com as credenciais da sua conta Twilio (disponíveis no [Console do Twilio](https://console.twilio.com/)):
```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_WHATSAPP_FROM=+14155238886
```
*(O número `+14155238886` é o padrão do Twilio Sandbox for WhatsApp).*

**2. Abrir o Terminal 1 — Subir o servidor FastAPI do Twilio (porta 8000):**
```powershell
.\.venv312\Scripts\Activate.ps1
python -m uvicorn server:app --app-dir src --host 0.0.0.0 --port 8000
```

**3. Abrir o Terminal 2 — Expor a porta 8000 com o ngrok:**
```powershell
ngrok start api --config ngrok-config.yml
```
O ngrok exibirá uma URL pública parecida com:
`https://xxxx-xxx-xxx-xxx.ngrok-free.app`

**4. Configurar o Webhook no Console do Twilio:**
1. Acesse **Messaging > Try it out > Send a WhatsApp message** (Sandbox) no painel do Twilio.
2. Aba **Sandbox settings**.
3. No campo **"When a message comes in"**, cole a URL do ngrok seguida de `/twilio-in`:
   ```text
   https://xxxx-xxx-xxx-xxx.ngrok-free.app/twilio-in
   ```
4. Certifique-se de que o método ao lado está como **`POST`** e clique em **Save**.
5. Envie uma mensagem de texto ou áudio para o número do Sandbox no WhatsApp para receber a análise!

**5. (Opcional) Simular requisição do Twilio localmente sem celular:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/twilio-in" -Method Post -ContentType "application/x-www-form-urlencoded" -Body "From=whatsapp:%2B5511999999999&Body=Vou te perseguir se voce me deixar&NumMedia=0"
```

---

#### Opção B: API REST Simples (`src/api.py` — Teste Local ou n8n)

Caso queira apenas consultar o endpoint JSON `/analisar` localmente ou via fluxo do **n8n**:

**Subir a API:**
```powershell
python -m uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
```
**Testar localmente em outro terminal PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/analisar" -Method Post -ContentType "application/json; charset=utf-8" -Body '{"texto": "Se você me deixar eu acabo com a sua vida"}'
```
- Documentação interativa (Swagger): `http://localhost:8000/docs`
- Caso use com n8n externo, suba também `ngrok start --all --config ngrok-config.yml` em outro terminal.

---

### 6. Retreinar os Modelos (Opcional)

Caso adicione novas frases em `treinamento/gerar_dataset_proprio.py`:
```powershell
# 1. Regenerar o CSV do dataset próprio
python treinamento\gerar_dataset_proprio.py

# 2. Executar o fine-tuning combinado
python treinamento\treinar_combinado.py --epocas 3 --max-len 256 --batch 16 --saida resultados_modelos/combinado_v3
```
*(Para treinar rapidamente com GPU gratuita no Google Colab, utilize o notebook `treinamento/colab_treino.ipynb`.)*
