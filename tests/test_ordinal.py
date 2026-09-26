"""Regresión ordinal con ridge (src/laliga/ordinal.py)."""
import numpy as np
import pandas as pd
import pytest

from laliga.ordinal import LAMBDAS, RegresionOrdinal, ajustar_ordinal


def datos_ordinales(n=3000, semilla=0):
    rng = np.random.default_rng(semilla)
    x = rng.normal(size=(n, 2))
    latente = 0.9 * x[:, 0] - 0.3 * x[:, 1] + rng.logistic(size=n)
    y = np.digitize(latente, [-0.8, 0.4])          # 0 visitante, 1 empate, 2 local
    return x, y


def test_probabilidades_validas():
    x, y = datos_ordinales()
    prob = RegresionOrdinal().fit(x, y).predict_proba(x)
    assert prob.shape == (len(y), 3)
    assert np.allclose(prob.sum(axis=1), 1)
    assert (prob > 0).all() and (prob < 1).all()


def test_coincide_con_statsmodels_sin_regularizacion():
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    x, y = datos_ordinales()
    nuestro = RegresionOrdinal(0.0).fit(x, y).predict_proba(x)
    referencia = OrderedModel(y, x, distr="logit").fit(method="bfgs", disp=False).predict(x)
    assert np.abs(nuestro - referencia).max() < 1e-4


def test_monotono_en_el_marcador_latente():
    x, y = datos_ordinales()
    m = RegresionOrdinal().fit(x, y)
    rejilla = np.column_stack([np.linspace(-3, 3, 50), np.zeros(50)])
    p_local = m.predict_proba(rejilla)[:, 2]
    assert (np.diff(p_local) > 0).all()


def test_ridge_encoge_los_coeficientes():
    x, y = datos_ordinales(n=500)
    libre = RegresionOrdinal(0.0).fit(x, y)
    penalizado = RegresionOrdinal(0.1).fit(x, y)
    assert np.abs(penalizado.beta).sum() < np.abs(libre.beta).sum()


def test_ajustar_ordinal_solo_usa_el_entrenamiento():
    x, y = datos_ordinales(n=1500)
    entreno = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y,
                            "temporada": np.repeat(["2014-15", "2015-16", "2016-17"], 500)})
    modelo, lam = ajustar_ordinal(entreno, ["a", "b"])
    assert lam in LAMBDAS
    # el modelo final se reentrena con TODO el entrenamiento y la lambda elegida
    directo = RegresionOrdinal(lam).fit(entreno[["a", "b"]], entreno.y)
    assert np.allclose(modelo.beta, directo.beta)
