
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .schemas import (
    DadosEstudante,
    ResultadoPredicao,
    InformacoesModelo,
)


RAIZ_PROJETO = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = RAIZ_PROJETO / "models" / "modelo_mlp.joblib"

artefato = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global artefato

    if not CAMINHO_MODELO.is_file():
        raise RuntimeError(
            f"Modelo não encontrado: {CAMINHO_MODELO}. "
            "Execute a exportação no Jupyter Notebook."
        )

    artefato = joblib.load(CAMINHO_MODELO)

    campos_necessarios = {
        "modelo",
        "colunas_entrada",
        "classes",
        "limites_desempenho",
        "versao_sklearn",
        "acuracia_teste",
    }

    if not isinstance(artefato, dict):
        raise RuntimeError("O artefato não é um dicionário válido.")

    faltando = campos_necessarios - artefato.keys()

    if faltando:
        raise RuntimeError(
            f"Campos ausentes no artefato: {sorted(faltando)}"
        )

    print("EduPredict: modelo MLP carregado com sucesso.")

    yield

    artefato = None


app = FastAPI(
    title="EduPredict API",
    description=(
        "API REST de classificação de desempenho acadêmico "
        "utilizando uma rede neural MLP. "
        "Projeto acadêmico de Aprendizagem de Máquina."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Sistema"])
def inicio():
    return {
        "aplicacao": "EduPredict",
        "mensagem": "API de inteligência artificial educacional",
        "documentacao": "/docs",
    }


@app.get("/health", tags=["Sistema"])
def health():
    return {
        "status": "ok",
        "modelo_carregado": artefato is not None,
    }


@app.get(
    "/model-info",
    response_model=InformacoesModelo,
    tags=["Modelo"],
)
def model_info():
    if artefato is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo indisponível.",
        )

    return {
        "nome": "EduPredict MLP",
        "algoritmo": "MLPClassifier",
        "classes": artefato["classes"],
        "atributos_entrada": artefato["colunas_entrada"],
        "acuracia_teste": artefato["acuracia_teste"],
        "limites_desempenho": artefato["limites_desempenho"],
        "versao_sklearn": artefato["versao_sklearn"],
    }


@app.post(
    "/predict",
    response_model=ResultadoPredicao,
    tags=["Predição"],
)
def predict(dados: DadosEstudante):
    if artefato is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo indisponível.",
        )

    modelo = artefato["modelo"]
    colunas = artefato["colunas_entrada"]

    entrada = pd.DataFrame([dados.model_dump()])
    entrada = entrada.reindex(columns=colunas)

    try:
        classificacao = str(modelo.predict(entrada)[0])

        probabilidades_array = modelo.predict_proba(entrada)[0]

        probabilidades = {
            str(classe): round(float(probabilidade), 6)
            for classe, probabilidade in zip(
                modelo.classes_,
                probabilidades_array,
            )
        }

    except (ValueError, TypeError) as erro:
        raise HTTPException(
            status_code=422,
            detail=f"Dados incompatíveis com o modelo: {erro}",
        ) from erro

    mensagens = {
        "BAIXO": "Desempenho previsto na faixa inferior.",
        "MEDIO": "Desempenho previsto na faixa intermediária.",
        "ALTO": "Desempenho previsto na faixa superior.",
    }

    return {
        "classificacao": classificacao,
        "confianca": probabilidades[classificacao],
        "probabilidades": probabilidades,
        "mensagem": mensagens.get(
            classificacao,
            "Classificação realizada.",
        ),
    }


@app.exception_handler(404)
async def rota_nao_encontrada(_request, _exception):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint não encontrado."},
    )
