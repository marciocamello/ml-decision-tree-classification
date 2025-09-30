from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / 'modelo_classficacao_decision_tree.pkl'

app = FastAPI(title='API - Modelo Decision Tree')


class PredictionRequest(BaseModel):
    # Lista de instâncias, cada instância é um dicionário de feature_name: value
    inputs: List[Dict[str, Any]]


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f'Modelo não encontrado: {MODEL_FILE}')
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model


@app.get('/')
def root():
    return {'message': 'API pronta. Use POST /predict com payload {"inputs":[{...}]}'}


@app.post('/predict')
def predict(payload: PredictionRequest):
    model = None
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Erro ao carregar modelo: {e}')

    df = pd.DataFrame(payload.inputs)
    if df.empty:
        raise HTTPException(status_code=400, detail='Nenhuma instância enviada em inputs')

    # Se o modelo expõe feature_names_in_, selecione as colunas nessa ordem
    expected_cols = None
    if hasattr(model, 'feature_names_in_'):
        expected_cols = list(model.feature_names_in_)

    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            # Preencher colunas faltantes com zero e avisar
            for c in missing:
                df[c] = 0
        X = df[expected_cols]
    else:
        X = df

    X = X.fillna(0)

    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Erro na predição: {e}')

    result = {'predictions': [int(p) if hasattr(p, 'astype') or isinstance(p, (int,)) else p for p in preds.tolist() if True]}

    # se o modelo suportar probabilidades, inclua-as
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X)
            result['probabilities'] = probs.tolist()
        except Exception:
            # não crítico — seguir sem probabilidades
            pass

    return result
