"""RPS, log-loss y acierto (src/laliga/metricas.py). Convención: columnas [visitante, empate, local]; y 0/1/2."""
import numpy as np
import pytest

from laliga.metricas import evaluar, log_loss, rps

GANA_LOCAL, EMPATE, GANA_VISITANTE = 2, 1, 0


def test_rps_ejemplo_de_referencia():
    # el ejemplo con el que se eligió la métrica: gana el local; A = 45/35/20, B = 45/20/35 (local/empate/visitante)
    prob = np.array([[0.20, 0.35, 0.45], [0.35, 0.20, 0.45]])
    assert rps(prob, np.array([GANA_LOCAL, GANA_LOCAL])) == pytest.approx([0.17125, 0.21250])


def test_rps_premia_errar_por_el_resultado_vecino():
    # misma probabilidad para el resultado real; el RPS prefiere repartir el resto en el empate (vecino)
    vecino = np.array([[0.1, 0.4, 0.5]])
    lejano = np.array([[0.4, 0.1, 0.5]])
    y = np.array([GANA_LOCAL])
    assert rps(vecino, y)[0] < rps(lejano, y)[0]
    assert log_loss(vecino, y)[0] == pytest.approx(log_loss(lejano, y)[0])   # el log-loss no ve el orden


@pytest.mark.parametrize("y", [GANA_VISITANTE, EMPATE, GANA_LOCAL])
def test_rps_perfecto_es_cero(y):
    prob = np.eye(3)[[y]]
    assert rps(prob, np.array([y]))[0] == pytest.approx(0)


def test_rps_peor_caso_es_uno():
    assert rps(np.array([[1.0, 0.0, 0.0]]), np.array([GANA_LOCAL]))[0] == pytest.approx(1)


def test_rps_uniforme():
    prob = np.full((3, 3), 1 / 3)
    valores = rps(prob, np.array([0, 1, 2]))
    assert valores == pytest.approx([5 / 18, 1 / 9, 5 / 18])


def test_log_loss_y_recorte():
    assert log_loss(np.array([[0.2, 0.3, 0.5]]), np.array([GANA_LOCAL]))[0] == pytest.approx(-np.log(0.5))
    assert np.isfinite(log_loss(np.array([[1.0, 0.0, 0.0]]), np.array([GANA_LOCAL]))[0])


def test_evaluar():
    prob = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    y = np.array([GANA_LOCAL, EMPATE])
    r = evaluar(prob, y)
    assert r["acierto"] == pytest.approx(0.5)
    assert r["RPS"] == pytest.approx(rps(prob, y).mean())
