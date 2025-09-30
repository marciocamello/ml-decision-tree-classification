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

print('Inferência finalizada — arquivo gerado: predicoes.csv')
```

Observação: o pré-processamento deve ser idêntico ao aplicado durante o treinamento. Se você usou pipelines (sklearn Pipeline) e salvou o pipeline completo junto com o modelo, o carregamento e uso ficam mais simples.

## Mapeamento do projeto para o conteúdo do curso Rocketseat

O módulo "Classificação Árvore de Decisão" (Rocketseat) cobre teoria e prática que se alinham diretamente às etapas acima. Pontos principais do curso que você deve estudar e relacionar com o projeto:

- Conceitos de classificação: problemas binários vs multiclasse — relação com seu rótulo alvo.
- EDA focada em classificação: análise de distribuição das classes, balanceamento e técnicas de amostragem (undersampling/oversampling).
- Pré-processamento: codificação de variáveis categóricas, tratamento de valores nulos, normalização quando necessário.
- Teoria de Árvores de Decisão: entropia, ganho de informação, índice Gini, como função de divisão escolhe splits.
- Tamanho/complexidade da árvore: profundidade máxima, min_samples_split, min_samples_leaf — relação com overfitting/underfitting.
- Avaliação de modelos de classificação: acurácia, precision, recall, f1-score, matriz de confusão, curva ROC/AUC (quando aplicável).
- Validação: holdout, k-fold cross-validation, validação estratificada para classes desbalanceadas.
- Otimização de hiperparâmetros: GridSearchCV / RandomizedSearchCV e interpretação dos resultados.
- Persistência e inferência: serializar com pickle, joblib; pipeline para manter transformações.
- Entregáveis práticos vistos no curso: inferência batch, criação de API (FastAPI/Flask) e deploy simples.

Links úteis do conteúdo Rocketseat (módulos referenciados):

- Página do curso: https://app.rocketseat.com.br/journey/machine-learning-em-inteligencia-artificial/contents
- Módulo Classificação Árvore de Decisão: https://app.rocketseat.com.br/classroom/classificacao-arvore-de-decisao

> Dica prática: acompanhe cada vídeo/aula do módulo e, ao final de cada aula, abra o notebook `classificacao_segment_empresa.ipynb` e vincule o que foi visto (por exemplo, após a aula sobre Gini/Entropy, localize a célula do treinamento e experimente trocar o parâmetro `criterion`).

## O que já foi feito (checklist)

- [x] Carregamento dos dados e EDA inicial.
- [x] Pré-processamento básico aplicado.
- [x] Treinamento de Decision Tree e avaliação básica.
- [x] Salvamento do modelo em `modelo_classficacao_decision_tree.pkl`.
- [x] Geração de um arquivo exemplo `predicoes.csv` com inferência.

## Exercícios e estudos recomendados (práticos)

1. Reproduzir o experimento trocando `criterion` entre `gini` e `entropy` e comparar métricas.
2. Implementar k-fold cross-validation estratificada e comparar variações de `max_depth`, `min_samples_leaf`.
3. Construir um sklearn Pipeline que inclua pré-processamento e modelo; salvar o pipeline inteiro e usar para inferência.
4. Plotar a árvore com `sklearn.tree.plot_tree` e interpretar os splits mais importantes.
5. Calcular importance das features (`feature_importances_`) e documentar as 5 mais relevantes.
6. Testar técnicas para lidar com classes desbalanceadas (SMOTE, class_weight) e medir impacto.
7. Criar uma API simples (FastAPI) que receba JSON com uma amostra e retorne a predição.

## Próximos passos sugeridos

- Refinar features e adicionar engenharia de domínio (se houver conhecimento sobre as empresas no dataset).
- Implementar e comparar RandomForest e outros ensembles (módulo posterior do curso) para ver ganho de performance.
- Aplicar hyperparameter tuning com RandomizedSearchCV/Optuna para acelerar busca.
- Adicionar testes automáticos simples (por exemplo, teste que checa se o pipeline salva/carrega e produz mesmo shape de saída).
- Documentar decisões: um arquivo `NOTES.md` com experimentos e métricas comparadas (boa prática para portfólio).

## Boas práticas e cuidados

- Sempre versionar os pré-processamentos (pipelines) junto com o modelo.
- Fixar seeds (random_state) para reprodutibilidade.
- Salvar exemplos de entrada e saída (samples de `novas_empresas.csv` e `predicoes.csv`).
- Evitar vazamento de dados: aplicar transformações aprendidas no treino apenas com dados de treino.

## Como estudar usando este projeto e o curso da Rocketseat

1. Assista às aulas do módulo "Classificação Árvore de Decisão" seguindo a ordem sugerida pela Rocketseat.
2. Após cada aula teórica, abra o notebook e aplique a mudança correspondente (ex.: após a aula de EDA, adicione novas visualizações; após a aula de pruning, experimente parâmetros).
3. Anote insights e métricas em `NOTES.md` (experimento, parâmetros, métricas, observações).
4. Ao terminar o módulo, execute os exercícios práticos listados acima e compare resultados.

## Contato / Referências

- Rocketseat — Machine Learning em Inteligência Artificial: https://app.rocketseat.com.br/journey/machine-learning-em-inteligencia-artificial/contents
- Scikit-Learn — Decision Trees: https://scikit-learn.org/stable/modules/tree.html

---

Se quiser, eu posso:

- gerar um `inferencia_batch.py` pronto com o pré-processamento baseado no notebook;
- criar um `requirements.txt` a partir do `Pipfile`;
- adicionar um `NOTES.md` com um template para experimentos e um pequeno script de avaliação automatizada.

Diga qual dessas tarefas quer que eu faça agora e eu executo aqui no repositório.
