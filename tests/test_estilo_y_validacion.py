"""Transformaciones que se ajustan con el entrenamiento (sin fuga) y esquema de validación temporal."""
import numpy as np
import pandas as pd
import pytest

from laliga.estilo import COORD, INTERACCIONES, VARIABLES_ESTILO, coordenadas_estilo
from laliga.validacion import TEMPORADAS_TEST, TEMPORADAS_VALIDACION, pliegues_walk_forward


def tabla_z(n, semilla):
    rng = np.random.default_rng(semilla)
    columnas = {f"z_{v}_{s}": rng.normal(size=n) for v in [*VARIABLES_ESTILO, "elo_medio"] for s in ("local", "visit")}
    return pd.DataFrame(columnas)


# --- coordenadas de estilo -------------------------------------------------------

def test_crea_las_coordenadas_e_interacciones():
    e, v = coordenadas_estilo(tabla_z(300, 0), tabla_z(50, 1))
    for df in (e, v):
        assert set(COORD + INTERACCIONES) <= set(df.columns)


def test_la_evaluacion_no_influye_en_la_transformacion():
    """Escalado y regresión se ajustan solo con el entrenamiento: cambiar la evaluación no mueve nada más."""
    entreno, evaluacion = tabla_z(300, 0), tabla_z(50, 1)
    e1, v1 = coordenadas_estilo(entreno, evaluacion)
    otra = evaluacion.copy()
    otra.iloc[1:] = otra.iloc[1:] * 10 + 5                  # se cambian todas las filas salvo la primera
    e2, v2 = coordenadas_estilo(entreno, otra)
    pd.testing.assert_frame_equal(e1, e2)                  # el entrenamiento no cambia
    pd.testing.assert_series_equal(v1.iloc[0], v2.iloc[0])  # la fila no tocada tampoco


def test_el_entrenamiento_si_define_la_transformacion():
    evaluacion = tabla_z(50, 1)
    _, v1 = coordenadas_estilo(tabla_z(300, 0), evaluacion)
    _, v2 = coordenadas_estilo(tabla_z(300, 2), evaluacion)
    assert not np.allclose(v1[COORD], v2[COORD])


def test_residuos_estandarizados_e_independientes_del_elo_en_entrenamiento():
    e, _ = coordenadas_estilo(tabla_z(400, 0), tabla_z(10, 1))
    x = np.concatenate([e.z_elo_medio_local, e.z_elo_medio_visit])
    for nombre in ("presion", "posesion", "faltas"):
        r = np.concatenate([e[f"{nombre}_local"], e[f"{nombre}_visit"]])
        assert np.std(r) == pytest.approx(1)
        assert abs(np.corrcoef(r, x)[0, 1]) < 1e-9


def test_presion_es_menos_ppda():
    e, _ = coordenadas_estilo(tabla_z(300, 0), tabla_z(10, 1))
    assert np.corrcoef(e.presion_local, e.z_ppda_local)[0, 1] < -0.9


# --- validación temporal -------------------------------------------------------------

def test_pliegues_solo_entrenan_con_el_pasado():
    temporadas = [f"{a}-{(a + 1) % 100:02d}" for a in range(2014, 2024)]
    pliegues = pliegues_walk_forward(temporadas)
    assert [v for _, v in pliegues] == TEMPORADAS_VALIDACION
    for entreno, valid in pliegues:
        assert entreno and all(t < valid for t in entreno)
        assert entreno == sorted(entreno) and entreno[0] == "2014-15"
    tamaños = [len(e) for e, _ in pliegues]
    assert tamaños == sorted(tamaños)                      # ventana creciente


def test_test_posterior_a_la_validacion_y_disjunto():
    assert not set(TEMPORADAS_TEST) & set(TEMPORADAS_VALIDACION)
    assert min(TEMPORADAS_TEST) > max(TEMPORADAS_VALIDACION)


def test_test_no_entra_en_ningun_pliegue():
    temporadas = [f"{a}-{(a + 1) % 100:02d}" for a in range(2014, 2026)]    # incluidas las de test
    for entreno, valid in pliegues_walk_forward(temporadas):
        assert not set(entreno) & set(TEMPORADAS_TEST)
        assert valid not in TEMPORADAS_TEST


def test_test_son_las_ultimas_temporadas_descargadas():
    import download
    ultimas = [download.season_label(a) for a in list(download.SEASONS)[-len(TEMPORADAS_TEST):]]
    assert ultimas == TEMPORADAS_TEST
