"""Etapa 2 — Preparação da base (feature engineering) do Bank Marketing.

Este módulo concentra a lógica de tratamento de dados definida na EDA
(``notebooks/00_eda.ipynb``), para ser reutilizada tanto no notebook de
preparação (``notebooks/01_feature_engineering.ipynb``) quanto, mais adiante,
na avaliação do golden set (Etapa 4) e no serviço de recomendação (Etapa 5).

Duas camadas de transformação, com motivações diferentes:

- :func:`clean_data` — transformações *sem estado* (não dependem de nada
  aprendido em treino): remoção de vazamento temporal, criação do target
  binário, tratamento do código sentinela de ``pdays``. Uma função pura
  resolve.
- :class:`ContextEncoder` — transformação *com estado* (one-hot das
  categóricas de contexto). Precisa ser ajustada (``fit``) uma única vez
  sobre a base de treino e depois reaplicada (``transform``) tanto em lote
  quanto sobre um único cliente novo, para que a mesma codificação (mesmas
  colunas, mesmas categorias aprendidas) seja usada em treino e em serviço —
  evitando o clássico problema de *train/serving skew*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------
# Colunas (decisões documentadas na EDA — notebooks/00_eda.ipynb, seção 8)
# --------------------------------------------------------------------------

TARGET_COL = "y"
TARGET_BIN_COL = "y_bin"

# Vazamento temporal confirmado na EDA: só é conhecida depois da ligação,
# ou seja, depois que a decisão de oferta já foi tomada e o resultado já é
# sabido. O próprio enunciado do desafio pede para descartá-la.
LEAKAGE_COLS = ["duration"]

# Braço (ação controlável pela instituição) — decisão da Etapa 2.
# A EDA levantou `contact` e/ou `month` como candidatos (célula de
# conclusões). Optamos por `contact` (cellular vs telephone):
#   - é binário, o que mantém o espaço de ação simples;
#   - é uma decisão de canal que a instituição controla diretamente,
#     em vez de um atributo fixo do cliente;
#   - evita diluir as poucas conversões (dataset desbalanceado) em muitos
#     braços, como aconteceria com `month` (10 categorias).
# `month` e `poutcome` ficam como contexto (sinal temporal/histórico), não
# como ação testável.
ARM_COL = "contact"

# Contexto: atributos do cliente + indicadores macro do período, todos
# disponíveis antes da decisão de oferta.
NUMERIC_CONTEXT_COLS = [
    "age",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
CATEGORICAL_CONTEXT_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "month",
    "poutcome",
]

PDAYS_COL = "pdays"
PDAYS_NEVER_CONTACTED_CODE = 999
PDAYS_FLAG_COL = "nunca_contatado"

CONTEXT_COLS = NUMERIC_CONTEXT_COLS + CATEGORICAL_CONTEXT_COLS + [PDAYS_FLAG_COL]


def load_raw(path: Path) -> pd.DataFrame:
    """Carrega a base bruta do Bank Marketing (separador `;`, como no Kaggle)."""
    return pd.read_csv(path, sep=";")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as transformações sem estado definidas na EDA (Etapa 1).

    - remove colunas de vazamento temporal (`duration`);
    - cria o target binário `y_bin` a partir de `y`;
    - transforma o código sentinela de `pdays` (999 = "nunca contatado") em
      uma flag binária, em vez de deixá-lo como número contínuo;
    - mantém `"unknown"` como categoria própria nas colunas categóricas —
      nenhuma linha é descartada aqui (decisão da EDA).
    """
    df = df.drop(columns=LEAKAGE_COLS)
    df[TARGET_BIN_COL] = (df[TARGET_COL] == "yes").astype(int)
    df[PDAYS_FLAG_COL] = (df[PDAYS_COL] == PDAYS_NEVER_CONTACTED_CODE).astype(int)
    df = df.drop(columns=[PDAYS_COL])
    return df


def split_context_arm_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separa o dataframe já tratado em contexto (X), braço e alvo (y)."""
    context = df[CONTEXT_COLS].copy()
    arm = df[ARM_COL].copy()
    target = df[TARGET_BIN_COL].copy()
    return context, arm, target


@dataclass
class ContextEncoder:
    """Encapsula o encoder de contexto já ajustado (fit) em treino.

    Guarda o `ColumnTransformer` ajustado sobre a base de treino, para que a
    mesma codificação (mesmas colunas, mesmas categorias aprendidas) seja
    reaplicada tanto em lote (Etapa 3) quanto sobre um único cliente novo
    (Etapas 4 e 5) via `transform`. `handle_unknown="ignore"` no
    OneHotEncoder evita erro caso uma categoria nova apareça em produção.
    """

    transformer: ColumnTransformer

    @classmethod
    def fit(cls, context: pd.DataFrame) -> "ContextEncoder":
        transformer = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    CATEGORICAL_CONTEXT_COLS,
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        transformer.fit(context)
        return cls(transformer=transformer)

    def transform(self, context: pd.DataFrame) -> pd.DataFrame:
        encoded = self.transformer.transform(context)
        columns = self.transformer.get_feature_names_out()
        return pd.DataFrame(encoded, columns=columns, index=context.index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.transformer, path)

    @classmethod
    def load(cls, path: Path) -> "ContextEncoder":
        return cls(transformer=joblib.load(path))


def build_dataset(
    raw_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, ContextEncoder]:
    """Pipeline completo de preparação: raw -> (contexto, braço, alvo, encoder).

    `context` volta em formato "cru" (colunas originais, sem one-hot) —
    útil para leitura humana, para calcular conversão por braço (Etapa 3) e
    para descrever os clientes do golden set (Etapa 4) antes de codificar.
    O encoder ajustado é o artefato que garante a mesma codificação em
    treino e em serviço.
    """
    df = load_raw(raw_path)
    df = clean_data(df)
    context, arm, target = split_context_arm_target(df)
    encoder = ContextEncoder.fit(context)
    return context, arm, target, encoder


def main() -> None:
    """Executa o pipeline sobre a base raw e persiste os artefatos em `data/processed/`."""
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "bank-additional-full.csv"
    processed_dir = project_root / "data" / "processed"

    context, arm, target, encoder = build_dataset(raw_path)

    prepared = context.copy()
    prepared[ARM_COL] = arm
    prepared[TARGET_BIN_COL] = target
    processed_dir.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(processed_dir / "bank_marketing_prepared.csv", index=False)
    encoder.save(processed_dir / "context_encoder.joblib")

    print(f"Contexto: {context.shape}")
    print(f"Braços ({ARM_COL}): {sorted(arm.unique())}")
    print(f"Conversão média (y_bin): {target.mean():.4f}")
    print(f"Artefatos salvos em: {processed_dir}")


if __name__ == "__main__":
    main()
