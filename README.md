## Projeto: Classificação — Árvore de Decisão

Este repositório contém um projeto prático de classificação baseado em árvore de decisão. O objetivo deste README é documentar de forma clara e didática tudo o que você já fez aqui, como reproduzir os resultados, o que estudar para entender cada etapa e próximos passos recomendados — tudo alinhado ao módulo "Classificação Árvore de Decisão" da formação "Machine Learning em Inteligência Artificial" da Rocketseat.

### Conteúdo deste repositório

- `classificacao_segment_empresa.ipynb` — notebook com o fluxo principal: EDA, pré-processamento, treinamento e avaliação do modelo.
- `modelo_classficacao_decision_tree.pkl` — modelo treinado (Decision Tree) serializado (pickle) para inferência em batch.
- `predicoes.csv` — exemplo de saída de predições (arquivo gerado após inferência).
- `Pipfile` / `Pipfile.lock` — gerenciador de dependências (Pipenv).
- `datasets/novas_empresas.csv` — dados de entrada para inferência (novas amostras).
- `datasets/segmento_clientes.csv` — dataset original / base utilizada no projeto (ou similar) para treinamento e análise.

Se algum arquivo não estiver presente ou tiver outro nome, verifique o notebook `classificacao_segment_empresa.ipynb` para referências exatas.

## Resumo do fluxo realizado

1. Coleta / carregamento dos dados (`segmento_clientes.csv`).
2. Análise Exploratória de Dados (EDA): inspeção de colunas, tipos, valores ausentes, distribuição de classes e visualizações (histogramas, boxplots, tabelas de frequência).
3. Limpeza e pré-processamento: tratamento de valores ausentes, codificação de variáveis categóricas (LabelEncoder / OneHotEncoding conforme necessidade), normalização/scale quando aplicável.
4. Engenharia de features: criação/seleção de colunas relevantes com base em EDA.
5. Separação treino / teste (train_test_split) e definição de seed para reprodutibilidade.
6. Treinamento de um DecisionTreeClassifier (sklearn): escolha de critérios (gini/entropy), controle de profundidade e parâmetros de regularização.
7. Avaliação do modelo: acurácia, matriz de confusão, classification report (precision/recall/f1), e análise de overfitting/underfitting.
8. Persistência do modelo (`pickle`) em `modelo_classficacao_decision_tree.pkl`.
9. Inferência em batch usando `datasets/novas_empresas.csv` gerando `predicoes.csv`.

## Como reproduzir o ambiente e rodar o projeto

Recomendado: use Pipenv (há um `Pipfile` no repositório). No Windows (PowerShell), execute:

```powershell
pipenv install --dev
pipenv shell
pipenv run jupyter notebook
```

Abra `classificacao_segment_empresa.ipynb` e execute as células em ordem. Se preferir executar apenas inferência a partir do modelo salvo, crie um script Python simples (exemplo abaixo).

Exemplo mínimo para carregar o modelo e gerar predições (arquivo `inferencia_batch.py` — opção):

```python
import pandas as pd
import pickle

# carregar dados
df = pd.read_csv('datasets/novas_empresas.csv')

# pré-processamento idêntico ao usado no notebook (importante!)
# ...aplicar transformações: encoding, criação de features, seleção de colunas...

# carregar modelo
with open('modelo_classficacao_decision_tree.pkl', 'rb') as f:
    model = pickle.load(f)

# previsões
preds = model.predict(df)
df['predicao'] = preds
df.to_csv('predicoes.csv', index=False)


# 📊 Projeto: Classificação — Árvore de Decisão (Decision Tree)

## 📚 Sobre o projeto

Este repositório implementa um modelo de classificação usando Árvore de Decisão (sklearn). É um projeto prático alinhado ao módulo "Classificação — Árvore de Decisão" da formação "Machine Learning em Inteligência Artificial" (Rocketseat). O objetivo é mostrar todo o fluxo: EDA, pré-processamento, treino, avaliação, persistência do modelo e inferência em batch.

### 🎯 Problema a ser resolvido

• Variáveis de entrada: atributos das empresas (dados em `datasets/segmento_clientes.csv`)
• Variável alvo (y): segmento/label de classificação (ver notebook para o nome exato da coluna)
• Objetivo: treinar um modelo que classifique corretamente o segmento de uma empresa com base nas suas features.

## 🛠️ Tecnologias utilizadas

• Python 3.11+
• Pandas, NumPy — manipulação de dados
• Scikit-learn — DecisionTreeClassifier, métricas e utilitários
• Matplotlib/Seaborn — visualização (usado no notebook)
• Pipenv — gerenciamento de dependências (Pipfile presente)

> Observação: se você quiser uma API/serviço, podemos adicionar FastAPI e Uvicorn (opcional).

## 📁 Estrutura do projeto

```

ml-decision-tree-classification/
├── datasets/
│ ├── segmento_clientes.csv # dataset usado para treino/EDA
│ └── novas_empresas.csv # amostras para inferência em batch
├── classificacao_segment_empresa.ipynb # notebook: EDA → treino → avaliação
├── modelo_classficacao_decision_tree.pkl # modelo treinado serializado
├── predicoes.csv # exemplo de saída gerada em inferência
├── Pipfile
├── Pipfile.lock
└── README.md

```

## 🧠 Conceitos de Machine Learning aplicados

• Classificação: árvore de decisão — splits baseados em Gini/Entropy
• Validação: holdout (train/test) e possibilidade de k-fold estratificado
• Métricas: acurácia, precision, recall, f1-score, matriz de confusão
• Overfitting vs Underfitting: controle por `max_depth`, `min_samples_leaf`, etc.

## 📈 Processo de Machine Learning implementado

1. Carregamento dos dados e EDA (inspeção, estatísticas, gráficos)
2. Tratamento de valores faltantes e codificação de categóricas
3. Separação treino/teste (com `random_state` fixo)
4. Treinamento de `DecisionTreeClassifier`
5. Avaliação com métricas e matriz de confusão
6. Salvamento do modelo (`pickle`) em `modelo_classficacao_decision_tree.pkl`
7. Inferência em batch em `datasets/novas_empresas.csv` → `predicoes.csv`

## 📊 Dataset

Arquivo principal: `datasets/segmento_clientes.csv` — abra o notebook para ver as colunas e amostras. Se precisar, adicione uma amostra mínima abaixo para referência.

Exemplo de visualização (no notebook):
```

... mostra as primeiras linhas com .head() ...

````

## 🚀 Como executar o projeto

1. Instalar Pipenv (se não tiver):

```powershell
pip install pipenv
````

2. Instalar dependências e abrir ambiente (PowerShell):

```powershell
pipenv install --dev
pipenv shell
```

3. Rodar Jupyter Notebook e abrir o notebook de análise:

```powershell
jupyter notebook classificacao_segment_empresa.ipynb
```

### Inferência rápida a partir do modelo salvo

Se quiser apenas gerar predições no arquivo `datasets/novas_empresas.csv`, crie um script `inferencia_batch.py` (exemplo abaixo) e execute dentro do mesmo ambiente:

```python
import pandas as pd
import pickle

# carregar dados
df = pd.read_csv('datasets/novas_empresas.csv')

# ATENÇÃO: aplicar as mesmas transformações usadas no treinamento
# Exemplo mínimo: selecionar colunas esperadas pelo modelo
# df = df[ ['col1','col2', ...] ]

with open('modelo_classficacao_decision_tree.pkl','rb') as f:
    model = pickle.load(f)

preds = model.predict(df)
df['predicao'] = preds
df.to_csv('predicoes.csv', index=False)
print('Arquivo gerado: predicoes.csv')
```

## 🔁 API / Deploy (opcional)

No momento este repositório não inclui uma API pronta. Se quiser, eu posso adicionar um pequeno `api_modelo.py` com FastAPI e um `api_main.py` (similar ao padrão do seu outro repo), que:

- carrega o modelo serializado
- expõe um endpoint POST `/predict` que recebe JSON com uma amostra e retorna a predição

### Exemplos de uso da API (após adicionar FastAPI):

- `uvicorn api_modelo:app --reload --port 8000`
- Documentação automática: `http://localhost:8000/docs`

## 📝 Material de estudo recomendado (Rocketseat)

- Módulo: Classificação — Árvore de Decisão (Rocketseat): https://app.rocketseat.com.br/classroom/classificacao-arvore-de-decisao
- Scikit-Learn — Decision Trees: https://scikit-learn.org/stable/modules/tree.html
- Conceitos: Entropia, Gini, Overfitting/Pruning, Validação cruzada, Feature importance

## ✅ Resumo para estudo rápido

O que foi implementado:

- Modelo Decision Tree treinado e salvo
- Notebook com EDA e avaliação
- Exemplo de inferência em batch (arquivo `predicoes.csv`)

Como testar rapidamente:

1. `pipenv install --dev`
2. `pipenv shell`
3. `jupyter notebook classificacao_segment_empresa.ipynb` ou executar `inferencia_batch.py` para gerar `predicoes.csv`

## 🤝 Como contribuir

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nome-da-feature`)
3. Commit e push
4. Abra um Pull Request

## 📝 Licença

Este projeto pode usar a licença MIT. Se quiser, adiciono o arquivo `LICENSE`.

---

Se quiser, posso agora:

- gerar o script `inferencia_batch.py` automaticamente com base no pré-processamento do notebook;
- gerar um `requirements.txt` a partir do `Pipfile`;
- adicionar `api_modelo.py` + `api_main.py` com FastAPI seguindo o padrão do seu repo de regressão.

Diga qual ação prefere que eu faça a seguir e eu implemento.
