"""IV implicita ATM do USD/BRL, interpolada a partir da superficie da B3
ja coletada (data/iv_surface.py).

A superficie so tem pontos nos vencimentos que a B3 publica (ver
data/iv_surface.py); para comparar com uma previsao de RV de horizonte fixo
(ex.: 21 dias uteis) precisamos interpolar. Interpolamos linearmente na
VARIANCIA TOTAL (iv^2 * T), nao na vol diretamente -- e a convencao padrao de
mercado de opcoes (a variancia acumulada cresce ~linear no tempo sob a
hipotese de vol local constante por trecho; interpolar a vol direto distorce
o smile term structure). Nas pontas (vencimento alvo fora do range coletado),
extrapola de forma flat (usa o vencimento disponivel mais proximo).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.iv_surface import PROCESSED_PATH as IV_PROCESSED_PATH

ATM_DELTA = 50.0


def interpolate_atm_iv(smile: pd.DataFrame, refdate: date, target_days: int) -> float:
    """Interpola a IV ATM (delta=50) para `target_days` dias corridos apos
    `refdate`, a partir de um smile de um unico refdate (colunas maturity_date,
    iv_pct, ja filtrado para delta_pct == ATM_DELTA).
    """
    smile = smile.sort_values("maturity_date")
    t_days = np.array([(m - refdate).days for m in smile["maturity_date"]], dtype=float)
    iv = smile["iv_pct"].to_numpy(dtype=float)

    if target_days <= t_days[0]:
        return float(iv[0])
    if target_days >= t_days[-1]:
        return float(iv[-1])

    i = np.searchsorted(t_days, target_days) - 1
    t1, t2 = t_days[i], t_days[i + 1]
    iv1, iv2 = iv[i], iv[i + 1]

    var1 = (iv1 / 100) ** 2 * t1
    var2 = (iv2 / 100) ** 2 * t2
    frac = (target_days - t1) / (t2 - t1)
    var_target = var1 + frac * (var2 - var1)
    return float(np.sqrt(var_target / target_days) * 100)


def atm_iv_series(df: pd.DataFrame, target_days: int = 21) -> pd.DataFrame:
    """A partir do long format de data.iv_surface.load_iv_surface_processed,
    interpola a IV ATM para `target_days` dias corridos, uma linha por refdate.

    Colunas: refdate, target_date, iv_atm_pct.
    """
    atm = df[df["delta_pct"] == ATM_DELTA]
    records = []
    for refdate_ts, smile in atm.groupby("refdate"):
        refdate = refdate_ts.date()
        iv_atm = interpolate_atm_iv(smile, refdate, target_days)
        records.append(
            {
                "refdate": refdate_ts,
                "target_date": refdate + timedelta(days=target_days),
                "iv_atm_pct": iv_atm,
            }
        )
    return pd.DataFrame(records).sort_values("refdate").reset_index(drop=True)


def load_iv_atm_processed(target_days: int = 21) -> pd.DataFrame:
    """Le o parquet processado da superficie (ja coletado por
    data.iv_surface.load_iv_surface_processed) e monta a serie de IV ATM
    interpolada para o vencimento alvo.
    """
    if not IV_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{IV_PROCESSED_PATH} nao encontrado -- rode "
            "data.iv_surface.load_iv_surface_processed() primeiro."
        )
    df = pd.read_parquet(IV_PROCESSED_PATH)
    return atm_iv_series(df, target_days)
