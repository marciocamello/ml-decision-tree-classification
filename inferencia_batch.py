"""Inferência em batch para o projeto de Árvore de Decisão.

Este script carrega o CSV `datasets/novas_empresas.csv`, aplica um pré-processamento
mínimo (defensivo) e usa o modelo serializado `modelo_classficacao_decision_tree.pkl`
para gerar `predicoes.csv`.

ATENÇÃO: este script contém pré-processamento genérico. Verifique e ajuste as
transformações (encoding, seleção de colunas, normalização) para que coincidam com
o que foi usado durante o treinamento no notebook `classificacao_segment_empresa.ipynb`.
"""

from pathlib import Path
import pandas as pd
import pickle
import sys

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / 'modelo_classficacao_decision_tree.pkl'
INPUT_FILE = ROOT / 'datasets' / 'novas_empresas.csv'
OUTPUT_FILE = ROOT / 'predicoes.csv'


def main():
    if not INPUT_FILE.exists():
        print(f"Arquivo de entrada não encontrado: {INPUT_FILE}")
        sys.exit(1)

    if not MODEL_FILE.exists():
        print(f"Modelo não encontrado: {MODEL_FILE}\nTreine o modelo ou coloque o arquivo correto na raiz do projeto.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # Exemplo de pré-processamento mínimo (adapte conforme o notebook):
    # - remover colunas extras que não participaram do treino
    # - preencher NA com valores simples
    # - codificar colunas categóricas se necessário

    # Tentativa simples: carregar as colunas que o modelo pode esperar.
    # Se o modelo for um Pipeline (scikit-learn), o carregamento e predict já
    # incluirão transformações; neste caso, basta passar o dataframe original.

    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print('Erro ao carregar o modelo:', e)
        sys.exit(1)

    # Se o modelo for um pipeline, ele aceitará o df diretamente (ou X com colunas).
    # Caso contrário, o usuário deve garantir que as colunas estejam na mesma ordem.

    # Detectar se o modelo possui atributo `feature_names_in_` (sklearn >= 1.0)
    expected_cols = None
    if hasattr(model, 'feature_names_in_'):
        expected_cols = list(model.feature_names_in_)

    # Se temos colunas esperadas, tentar selecioná-las
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print('Atenção: colunas esperadas pelo modelo não foram encontradas:', missing)
            print('Você precisa aplicar o mesmo pré-processamento usado no treinamento.')
            # Tentar preencher colunas faltantes com zeros
            for c in missing:
                df[c] = 0
        X = df[expected_cols]
    else:
        # Sem informações sobre colunas esperadas: passar o dataframe inteiro.
        X = df.copy()

    # Preenchimento simples de NA
    X = X.fillna(0)

    try:
        preds = model.predict(X)
    except Exception as e:
        print('Erro durante a predição. Provavelmente o formato das colunas não bate com o treinamento.')
        print('Detalhes:', e)
        sys.exit(1)

    # Anexar predições e salvar
    df_result = df.copy()
    df_result['predicao'] = preds
    df_result.to_csv(OUTPUT_FILE, index=False)
    print(f'Predições geradas e salvas em: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
