"""Etapa 5 — Serviço demonstrável: API FastAPI que recomenda o canal de contato.

Reaproveita a lógica já construída nas etapas anteriores em vez de duplicá-la:

- `features.py` (Etapa 2) — colunas de contexto do cliente, já tratadas.
- `bandit.py` (Etapa 3/4) — `ThompsonSamplingBandit.fit` ajusta a posterior
  sobre todo o histórico no startup do serviço (mesma abordagem do golden set
  da Etapa 4) e `select_arm` decide o canal para um cliente novo.

Rodar localmente:
    uv run uvicorn tech_challenge_5.api:app --reload

Depois, abra http://127.0.0.1:8000/docs (Swagger, gerado automaticamente pelo
FastAPI a partir do schema abaixo) para testar `POST /recommend` direto do
navegador.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from tech_challenge_5.bandit import ThompsonSamplingBandit, best_arm_by_segment
from tech_challenge_5.features import ARM_COL, TARGET_BIN_COL

# Segmento de contexto escolhido na Etapa 3 — ver notebooks/02_baseline_bandit.ipynb.
SEGMENT_COL = "month"

PROCESSED_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "bank_marketing_prepared.csv"
)


def fit_final_bandit(
    processed_path: Path = PROCESSED_PATH,
) -> tuple[ThompsonSamplingBandit, pd.DataFrame]:
    """Ajusta o bandit "final" sobre o histórico completo (mesma lógica da Etapa 4)."""
    df = pd.read_csv(processed_path)
    arms = sorted(df[ARM_COL].unique())
    bandit = ThompsonSamplingBandit.fit(
        df,
        arms=arms,
        segment_col=SEGMENT_COL,
        arm_col=ARM_COL,
        reward_col=TARGET_BIN_COL,
        rng=np.random.default_rng(7),
    )
    oracle = best_arm_by_segment(df, SEGMENT_COL, ARM_COL, TARGET_BIN_COL)
    return bandit, oracle


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bandit, app.state.oracle = fit_final_bandit()
    yield


app = FastAPI(
    title="Datathon — Recomendador de Canal de Contato",
    description=(
        "Etapa 5: dado o contexto de um cliente, retorna o canal de contato "
        "(cellular ou telephone) recomendado pelo Thompson Sampling contextual "
        "da Etapa 3."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Categorias observadas na base (Etapa 1/2) — restringir o schema a elas dá
# validação de graça e um dropdown pronto no Swagger (/docs).
Job = Literal[
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]
Marital = Literal["divorced", "married", "single", "unknown"]
Education = Literal[
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown",
]
YesNoUnknown = Literal["no", "yes", "unknown"]
Month = Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
Poutcome = Literal["failure", "nonexistent", "success"]
Offer = Literal["cellular", "telephone"]


class ClientContext(BaseModel):
    """Contexto do cliente, nas mesmas colunas produzidas por `features.clean_data`."""

    age: int = Field(..., ge=17, le=100, examples=[41])
    campaign: int = Field(..., ge=1, examples=[2])
    previous: int = Field(..., ge=0, examples=[0])
    emp_var_rate: float = Field(..., examples=[1.4])
    cons_price_idx: float = Field(..., examples=[93.994])
    cons_conf_idx: float = Field(..., examples=[-36.4])
    euribor3m: float = Field(..., examples=[4.857])
    nr_employed: float = Field(..., examples=[5191.0])
    job: Job = Field(..., examples=["technician"])
    marital: Marital = Field(..., examples=["married"])
    education: Education = Field(..., examples=["university.degree"])
    default: YesNoUnknown = Field(..., examples=["no"])
    housing: YesNoUnknown = Field(..., examples=["yes"])
    loan: YesNoUnknown = Field(..., examples=["no"])
    month: Month = Field(..., examples=["may"])
    poutcome: Poutcome = Field(..., examples=["nonexistent"])
    nunca_contatado: bool = Field(..., examples=[True])

    def to_context_row(self) -> pd.Series:
        """Mapeia os campos do request para as colunas cruas de `features.CONTEXT_COLS`."""
        return pd.Series(
            {
                "age": self.age,
                "campaign": self.campaign,
                "previous": self.previous,
                "emp.var.rate": self.emp_var_rate,
                "cons.price.idx": self.cons_price_idx,
                "cons.conf.idx": self.cons_conf_idx,
                "euribor3m": self.euribor3m,
                "nr.employed": self.nr_employed,
                "job": self.job,
                "marital": self.marital,
                "education": self.education,
                "default": self.default,
                "housing": self.housing,
                "loan": self.loan,
                "month": self.month,
                "poutcome": self.poutcome,
                "nunca_contatado": int(self.nunca_contatado),
            }
        )


class RecommendationResponse(BaseModel):
    recommended_offer: Offer
    estimated_conversion: dict[Offer, float]
    best_historical_offer: Offer


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(client: ClientContext, request: Request) -> RecommendationResponse:
    """Recomenda um canal de contato para o cliente, usando o bandit da Etapa 3/4."""
    bandit: ThompsonSamplingBandit = request.app.state.bandit
    oracle: pd.DataFrame = request.app.state.oracle

    context_row = client.to_context_row()
    recommended = bandit.select_arm(context_row)

    posterior = bandit.posterior_summary().rename(
        columns={"segmento": SEGMENT_COL, "braco": ARM_COL}
    )
    month_posterior = (
        posterior[posterior[SEGMENT_COL] == client.month]
        .set_index(ARM_COL)["conversao_estimada"]
        .round(4)
    )

    month_oracle = oracle[oracle[SEGMENT_COL] == client.month].set_index(ARM_COL)
    best_historical = month_oracle["conversao_historica"].idxmax()

    return RecommendationResponse(
        recommended_offer=recommended,
        estimated_conversion=month_posterior.to_dict(),
        best_historical_offer=best_historical,
    )
