"""Etapa 3 — Baseline e bandit adaptativo (Thompson Sampling) do Bank Marketing.

Contém:

- :class:`BaselinePolicy` — regra fixa: sempre recomenda o braço com melhor
  conversão histórica.
- :class:`ThompsonSamplingBandit` — Thompson Sampling Beta-Bernoulli, com um
  par (alpha, beta) por braço **dentro de cada segmento de contexto**. É
  assim que o contexto do cliente entra na decisão (exigência explícita do
  desafio) sem precisar de um bandit contextual mais pesado (regressão por
  braço, LinUCB etc.) — braços que funcionam bem para um cliente com
  `poutcome=success` podem não ser os mesmos que funcionam para
  `poutcome=nonexistent`.
- :func:`replay_evaluate` — avalia uma política sobre dados **logados**
  (a base é histórica, não um ambiente interativo), usando o método de
  replay/rejection sampling (Li et al.): só contamos uma rodada quando o
  braço escolhido pela política coincide com o braço realmente registrado
  naquela linha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


class Policy(Protocol):
    """Interface mínima que `replay_evaluate` espera de uma política."""

    def select_arm(self, context: pd.Series) -> str: ...

    def update(self, arm: str, reward: int, context: pd.Series) -> None: ...


@dataclass
class BaselinePolicy:
    """Regra fixa: sempre recomenda o braço com melhor conversão histórica.

    Não aprende — `update` é no-op. Serve como referência determinística
    contra a qual o bandit adaptativo precisa mostrar ganho.
    """

    fixed_arm: str

    @classmethod
    def from_historical_data(cls, arm: pd.Series, reward: pd.Series) -> "BaselinePolicy":
        rates = pd.DataFrame({"arm": arm, "reward": reward}).groupby("arm")["reward"].mean()
        return cls(fixed_arm=str(rates.idxmax()))

    def select_arm(self, context: pd.Series) -> str:
        return self.fixed_arm

    def update(self, arm: str, reward: int, context: pd.Series) -> None:
        pass


@dataclass
class ThompsonSamplingBandit:
    """Thompson Sampling Beta-Bernoulli, segmentado por uma coluna de contexto.

    Mantém um par (alpha, beta) por combinação (segmento, braço), com prior
    Beta(1, 1) (uniforme) por padrão. A cada decisão, amostra uma taxa de
    conversão da posterior de cada braço *dentro do segmento do cliente* e
    escolhe o braço com a maior amostra — equilíbrio natural entre
    explorar (braços com posterior ainda incerta) e explotar (braços com
    posterior concentrada em uma taxa alta).
    """

    arms: list[str]
    segment_col: str
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    _params: dict[tuple[str, str], list[float]] = field(default_factory=dict, repr=False)

    def _get_params(self, segment: str, arm: str) -> tuple[float, float]:
        key = (segment, arm)
        if key not in self._params:
            self._params[key] = [self.prior_alpha, self.prior_beta]
        alpha, beta = self._params[key]
        return alpha, beta

    def select_arm(self, context: pd.Series) -> str:
        segment = str(context[self.segment_col])
        samples = {arm: self.rng.beta(*self._get_params(segment, arm)) for arm in self.arms}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: int, context: pd.Series) -> None:
        segment = str(context[self.segment_col])
        alpha, beta = self._get_params(segment, arm)
        if reward:
            alpha += 1
        else:
            beta += 1
        self._params[(segment, arm)] = [alpha, beta]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        arms: list[str],
        segment_col: str,
        arm_col: str,
        reward_col: str,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> "ThompsonSamplingBandit":
        """Ajusta a posterior direto sobre dados logados completos, sem replay.

        `replay_evaluate` descarta rodadas em que o braço escolhido pela
        política não bate com o braço logado — isso é necessário para medir
        performance online sem viés (Etapa 3). Mas para *servir* recomendações
        (Etapa 4/5) queremos o melhor estado possível da posterior, e cada
        linha do histórico já nos diz qual braço foi mostrado e qual foi o
        resultado: não há nada a "explorar" quando os dados já existem todos
        de uma vez, então cada linha atualiza diretamente o par
        (segmento, braço) correspondente.
        """
        bandit = cls(
            arms=arms,
            segment_col=segment_col,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            rng=rng if rng is not None else np.random.default_rng(),
        )
        counts = (
            df.groupby([segment_col, arm_col])[reward_col]
            .agg(successes="sum", n="count")
            .reset_index()
        )
        for _, row in counts.iterrows():
            successes = int(row["successes"])
            n = int(row["n"])
            key = (str(row[segment_col]), str(row[arm_col]))
            bandit._params[key] = [prior_alpha + successes, prior_beta + (n - successes)]
        return bandit

    def posterior_summary(self) -> pd.DataFrame:
        """Média posterior (taxa de conversão estimada) por segmento x braço."""
        rows = [
            {
                "segmento": segment,
                "braco": arm,
                "alpha": alpha,
                "beta": beta,
                "conversao_estimada": alpha / (alpha + beta),
            }
            for (segment, arm), (alpha, beta) in self._params.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values(["segmento", "braco"])
            .reset_index(drop=True)
        )


def replay_evaluate(
    df: pd.DataFrame,
    policy: Policy,
    arm_col: str,
    reward_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Avalia `policy` via replay (Li et al.) sobre dados logados.

    A cada linha embaralhada, a política escolhe um braço olhando o
    contexto daquela linha; se o braço escolhido coincidir com o braço
    realmente registrado (`arm_col`), a rodada é "aproveitada": a política
    é atualizada com a recompensa observada (`reward_col`) e a rodada entra
    na conversão acumulada. Caso contrário, a linha é descartada (não
    sabemos o que teria acontecido com outro braço) e seguimos para a
    próxima linha embaralhada.

    É um estimador não-viesado da performance da política sob a hipótese
    de que a atribuição histórica do braço não depende fortemente do
    contexto observado — uma simplificação que vale deixar documentada
    (o `contact` real do banco não foi sorteado por um bandit).
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    played_rewards: list[int] = []
    for _, row in shuffled.iterrows():
        chosen_arm = policy.select_arm(row)
        if chosen_arm != row[arm_col]:
            continue
        reward = int(row[reward_col])
        policy.update(chosen_arm, reward, row)
        played_rewards.append(reward)

    rewards = np.asarray(played_rewards, dtype=float)
    n = len(rewards)
    return pd.DataFrame(
        {
            "rodada_aproveitada": np.arange(1, n + 1),
            "recompensa": rewards,
            "conversao_acumulada": np.cumsum(rewards) / np.arange(1, n + 1),
        }
    )


def best_arm_by_segment(
    df: pd.DataFrame, segment_col: str, arm_col: str, reward_col: str
) -> pd.DataFrame:
    """Conversão histórica observada por (segmento, braço) — referência/oráculo.

    Útil para comparar contra o que o Thompson Sampling aprendeu
    (`ThompsonSamplingBandit.posterior_summary`) e para justificar por que
    segmentar por `segment_col` traz ganho sobre uma única regra fixa.
    """
    return (
        df.groupby([segment_col, arm_col])[reward_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "conversao_historica", "count": "n"})
        .reset_index()
    )
