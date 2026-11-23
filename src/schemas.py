
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DadosEstudante(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Hours_Studied": 20,
                "Attendance": 85,
                "Parental_Involvement": "Medium",
                "Access_to_Resources": "High",
                "Extracurricular_Activities": "Yes",
                "Sleep_Hours": 7,
                "Previous_Scores": 75,
                "Motivation_Level": "Medium",
                "Internet_Access": "Yes",
                "Tutoring_Sessions": 2,
                "Family_Income": "Medium",
                "Teacher_Quality": "Good",
                "School_Type": "Public",
                "Peer_Influence": "Positive",
                "Physical_Activity": 3,
                "Learning_Disabilities": "No",
                "Parental_Education_Level": "College",
                "Distance_from_Home": "Near",
                "Gender": "Female"
            }
        }
    )

    Hours_Studied: int = Field(ge=0, le=168)
    Attendance: int = Field(ge=0, le=100)

    Parental_Involvement: Literal["Low", "Medium", "High"]
    Access_to_Resources: Literal["Low", "Medium", "High"]
    Extracurricular_Activities: Literal["Yes", "No"]

    Sleep_Hours: int = Field(ge=0, le=24)
    Previous_Scores: int = Field(ge=0, le=100)

    Motivation_Level: Literal["Low", "Medium", "High"]
    Internet_Access: Literal["Yes", "No"]

    Tutoring_Sessions: int = Field(ge=0)
    Family_Income: Literal["Low", "Medium", "High"]

    Teacher_Quality: Literal[
        "Low", "Medium", "High", "Good", "Average", "Poor"
    ]

    School_Type: Literal["Public", "Private"]

    Peer_Influence: Literal[
        "Positive", "Neutral", "Negative"
    ]

    Physical_Activity: int = Field(ge=0)
    Learning_Disabilities: Literal["Yes", "No"]

    Parental_Education_Level: Literal[
        "High School", "College", "Postgraduate"
    ]

    Distance_from_Home: Literal[
        "Near", "Moderate", "Far"
    ]

    Gender: Literal["Male", "Female"]


class ResultadoPredicao(BaseModel):
    classificacao: Literal["BAIXO", "MEDIO", "ALTO"]
    confianca: float = Field(ge=0, le=1)
    probabilidades: dict[str, float]
    mensagem: str


class InformacoesModelo(BaseModel):
    nome: str
    algoritmo: str
    classes: list[str]
    atributos_entrada: list[str]
    acuracia_teste: float
    limites_desempenho: dict[str, float]
    versao_sklearn: str
