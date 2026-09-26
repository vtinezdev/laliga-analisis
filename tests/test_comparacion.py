"""Comparación estadística de modelos (src/laliga/comparacion.py; notebook 04, sección 18).

Convención: diferencia = modelo nuevo − modelo base. En RPS y log-loss, negativa = mejora.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

from conftest import PROCESSED, requiere
from laliga.comparacion import (METRICAS, comparar_modelos, comprobar_alineacion, evidencia,
                                ic_acierto_exacto,
                                intervalo_percentil, mcnemar_exacto, medias_bootstrap, perdidas_por_partido,
                                pesos_bootstrap, t_pareada)
from laliga.metricas import evaluar, log_loss, rps


def probabilidades(n, semilla):
    return np.random.default_rng(semilla).dirichlet([2, 1.5, 3], size=n)


@pytest.fixture
def dos_modelos():
    """Dos modelos sobre los mismos 300 partidos; el nuevo pone algo más de probabilidad en el resultado real."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    base = probabilidades(300, 2)
    nuevo = 0.8 * base + 0.2 * np.eye(3)[y]
    ids = np.arange(1000, 1300)
    return {"base": perdidas_por_partido(ids, base, y), "nuevo": perdidas_por_partido(ids, nuevo, y)}


# --- pérdidas por partido y alineación -------------------------------------------------------

def test_perdidas_reutilizan_las_metricas_del_proyecto():
    y = np.array([2, 1, 0, 2])
    prob = probabilidades(4, 0)
    p = perdidas_por_partido([10, 11, 12, 13], prob, y)
    assert np.allclose(p.RPS, rps(prob, y)) and np.allclose(p.log_loss, log_loss(prob, y))
    medias = evaluar(prob, y)
    assert p.RPS.mean() == pytest.approx(medias["RPS"])
    assert p.log_loss.mean() == pytest.approx(medias["log_loss"])
    assert p.acierto.mean() == pytest.approx(medias["acierto"])


def test_partido_repetido_se_rechaza():
    with pytest.raises(ValueError, match="repetido"):
        perdidas_por_partido([1, 1], probabilidades(2, 0), np.array([0, 1]))


def test_alineacion_correcta(dos_modelos):
    comprobar_alineacion(dos_modelos)


def test_alineacion_detecta_otro_orden(dos_modelos):
    desordenado = {**dos_modelos, "nuevo": dos_modelos["nuevo"].iloc[::-1]}
    with pytest.raises(ValueError, match="mismo orden"):
        comprobar_alineacion(desordenado)


def test_alineacion_detecta_otros_partidos(dos_modelos):
    otros = dos_modelos["nuevo"].copy()
    otros.index = otros.index + 1
    with pytest.raises(ValueError, match="mismos partidos"):
        comprobar_alineacion({**dos_modelos, "nuevo": otros})
    with pytest.raises(ValueError, match="mismos partidos"):
        comprobar_alineacion({**dos_modelos, "nuevo": dos_modelos["nuevo"].iloc[:-1]})


def test_alineacion_detecta_otro_resultado(dos_modelos):
    cambiado = dos_modelos["nuevo"].copy()
    cambiado.iloc[0, cambiado.columns.get_loc("y")] = (cambiado.y.iloc[0] + 1) % 3
    with pytest.raises(ValueError, match="resultado real"):
        comprobar_alineacion({**dos_modelos, "nuevo": cambiado})


def test_comparar_modelos_exige_alineacion(dos_modelos):
    with pytest.raises(ValueError):
        comparar_modelos({**dos_modelos, "nuevo": dos_modelos["nuevo"].iloc[::-1]}, [("base", "nuevo", "x")],
                         n_remuestras=100)


# --- bootstrap emparejado ----------------------------------------------------------------------

def test_bootstrap_reproducible_con_semilla():
    assert np.array_equal(pesos_bootstrap(50, 200, semilla=7), pesos_bootstrap(50, 200, semilla=7))
    assert not np.array_equal(pesos_bootstrap(50, 200, semilla=7), pesos_bootstrap(50, 200, semilla=8))


def test_pesos_suman_n_y_son_remuestras_con_reemplazo():
    pesos = pesos_bootstrap(40, 500, semilla=0)
    assert pesos.shape == (500, 40)
    assert (pesos.sum(axis=1) == 40).all()
    assert (pesos >= 0).all() and np.allclose(pesos, pesos.round())
    assert pesos.mean() == pytest.approx(1, abs=0.02)              # cada partido entra una vez de media


def test_pesos_por_grupos_remuestrean_grupos_enteros():
    grupos = np.repeat(["s1", "s2", "s3", "s4"], [3, 5, 2, 4])
    pesos = pesos_bootstrap(len(grupos), 300, semilla=0, grupos=grupos)
    for g in np.unique(grupos):
        bloque = pesos[:, grupos == g]
        assert (bloque == bloque[:, [0]]).all()                      # mismo peso dentro del grupo
    with pytest.raises(ValueError):
        pesos_bootstrap(5, 10, grupos=["a", "b"])


def test_bootstrap_es_emparejado():
    # si el modelo nuevo pierde exactamente 0,01 menos en cada partido, TODAS las remuestras dan −0,01:
    # la variabilidad entre partidos se cancela porque se remuestrean los mismos partidos para los dos
    base = np.random.default_rng(0).uniform(0, 0.6, 400)
    nuevo = base - 0.01
    distribucion = medias_bootstrap(nuevo - base, pesos_bootstrap(400, 1000, semilla=0))
    assert np.allclose(distribucion, -0.01)
    # remuestrear cada modelo por separado (no emparejado) mezclaría esa variabilidad
    no_emparejada = (medias_bootstrap(nuevo, pesos_bootstrap(400, 1000, semilla=1))
                     - medias_bootstrap(base, pesos_bootstrap(400, 1000, semilla=2)))
    assert no_emparejada.std() > 0.005


def test_medias_bootstrap_coinciden_con_remuestrear_indices():
    d = np.random.default_rng(3).normal(size=30)
    pesos = pesos_bootstrap(30, 5, semilla=4)
    for fila, media in zip(pesos, medias_bootstrap(d, pesos)):
        indices = np.repeat(np.arange(30), fila.astype(int))
        assert d[indices].mean() == pytest.approx(media)


# --- intervalos de confianza -------------------------------------------------------------------

def test_intervalo_percentil_son_los_cuantiles():
    distribucion = np.arange(1, 10_001) / 10_000
    inferior, superior = intervalo_percentil(distribucion, 0.95)
    assert inferior == pytest.approx(0.025, abs=1e-3) and superior == pytest.approx(0.975, abs=1e-3)
    inf99, sup99 = intervalo_percentil(distribucion, 0.99)
    assert inf99 < inferior and sup99 > superior                   # más confianza, intervalo más ancho


def test_cobertura_del_ic_bootstrap():
    # con datos simulados de media conocida (0,1), el IC 95 % debe contenerla ≈ 95 % de las veces
    rng = np.random.default_rng(5)
    pesos = pesos_bootstrap(200, 1000, semilla=6)
    aciertos = 0
    for _ in range(300):
        d = rng.normal(0.1, 1, 200)
        inferior, superior = intervalo_percentil(medias_bootstrap(d, pesos))
        aciertos += inferior <= 0.1 <= superior
    assert 0.90 <= aciertos / 300 <= 0.99


# --- contrastes --------------------------------------------------------------------------------

def test_t_pareada_coincide_con_scipy():
    rng = np.random.default_rng(0)
    base, nuevo = rng.uniform(0, 1, 250), rng.uniform(0, 1, 250) - 0.05
    t, p = t_pareada(nuevo - base)
    referencia = stats.ttest_rel(nuevo, base)
    assert t == pytest.approx(referencia.statistic) and p == pytest.approx(referencia.pvalue)


def test_t_pareada_es_diebold_mariano_hln_con_horizonte_1():
    # DM = media / √(γ0 / n), con γ0 = varianza de denominador n; HLN multiplica por √((n + 1 − 2h + h(h − 1)/n) / n)
    d = np.random.default_rng(1).standard_t(5, 705) * 0.03 + 0.001
    n, h = len(d), 1
    dm = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / n)
    dm_hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    t, p = t_pareada(d)
    assert t == pytest.approx(dm_hln, rel=1e-12)
    assert p == pytest.approx(2 * stats.t.sf(abs(dm_hln), df=n - 1), rel=1e-9)


def test_t_pareada_casos_limite():
    assert t_pareada(np.zeros(10)) == (0.0, 1.0)                   # modelos idénticos
    t, p = t_pareada(np.full(10, -0.01))                            # siempre mejor por lo mismo
    assert p == 0.0 and t == -np.inf


def test_mcnemar_coincide_con_statsmodels():
    rng = np.random.default_rng(0)
    base, nuevo = rng.random(300) < 0.5, rng.random(300) < 0.55
    r = mcnemar_exacto(base, nuevo)
    tabla = pd.crosstab(base, nuevo).reindex(index=[True, False], columns=[True, False]).to_numpy()
    assert r["solo_base"] == tabla[0, 1] and r["solo_nuevo"] == tabla[1, 0]
    assert r["p"] == pytest.approx(mcnemar(tabla, exact=True).pvalue)


def test_mcnemar_sin_discordancias():
    acierto = np.array([True, False, True])
    assert mcnemar_exacto(acierto, acierto) == {"solo_base": 0, "solo_nuevo": 0, "p": 1.0}


def test_ic_acierto_exacto_coherente_con_mcnemar():
    # para cualquier tabla de discordantes, el IC 95 % excluye el 0 exactamente cuando McNemar da p < 0,05
    for b in range(0, 30):
        for c in range(0, 30):
            base = np.r_[np.ones(b, bool), np.zeros(c, bool), np.ones(50, bool)]
            nuevo = np.r_[np.zeros(b, bool), np.ones(c, bool), np.ones(50, bool)]
            p = mcnemar_exacto(base, nuevo)["p"]
            inferior, superior = ic_acierto_exacto(b, c, len(base))
            assert inferior <= (c - b) / len(base) <= superior
            assert (inferior > 0 or superior < 0) == (p < 0.05), (b, c)


def test_ic_acierto_exacto_es_clopper_pearson_transformado():
    from statsmodels.stats.proportion import proportion_confint
    inf_pi, sup_pi = proportion_confint(28, 40, alpha=0.05, method="beta")
    inferior, superior = ic_acierto_exacto(12, 28, 705)
    assert inferior == pytest.approx(40 / 705 * (2 * inf_pi - 1))
    assert superior == pytest.approx(40 / 705 * (2 * sup_pi - 1))
    assert ic_acierto_exacto(0, 0, 705) == (0.0, 0.0)
    assert ic_acierto_exacto(5, 9, 100, 0.99)[0] < ic_acierto_exacto(5, 9, 100, 0.95)[0]


def test_evidencia_nunca_afirma_igualdad():
    assert evidencia(-0.01, 0.001, 0.003, menor_es_mejor=True) == "Evidencia estadística de mejora del modelo nuevo"
    assert evidencia(-0.01, 0.001, 0.003, menor_es_mejor=False) == "Evidencia estadística de empeoramiento del modelo nuevo"
    assert evidencia(0.02, 0.001, 0.003, menor_es_mejor=False) == "Evidencia estadística de mejora del modelo nuevo"
    assert "multiplicidad" in evidencia(-0.01, 0.02, 0.10, menor_es_mejor=True)
    for texto in (evidencia(0.0, 0.8, 1.0, True), evidencia(-0.01, 0.02, 0.10, True)):
        assert texto.startswith("Sin evidencia") and "igual" not in texto.lower()


# --- tabla completa ----------------------------------------------------------------------------

def test_signo_de_la_diferencia(dos_modelos):
    tabla = comparar_modelos(dos_modelos, [("base", "nuevo", "base → nuevo")], n_remuestras=2000).set_index("metrica")
    # el nuevo es mejor en todo: diferencia negativa en pérdidas y positiva (o nula) en acierto
    assert tabla.loc["RPS", "diferencia"] < 0 and tabla.loc["log_loss", "diferencia"] < 0
    assert tabla.loc["RPS", "diferencia"] == pytest.approx(dos_modelos["nuevo"].RPS.mean() - dos_modelos["base"].RPS.mean())
    assert tabla.loc["RPS", "evidencia"] == "Evidencia estadística de mejora del modelo nuevo"
    assert tabla.loc["RPS", "cambio_relativo_vs_base"] == pytest.approx(
        tabla.loc["RPS", "diferencia"] / dos_modelos["base"].RPS.mean())      # cambio relativo = diferencia / base
    assert tabla.loc["acierto", "diferencia"] >= 0
    for metrica in METRICAS:
        f = tabla.loc[metrica]
        assert f.ic95_inferior <= f.diferencia <= f.ic95_superior
        assert f.ic_simultaneo_inferior <= f.ic95_inferior and f.ic_simultaneo_superior >= f.ic95_superior


def test_invertir_la_comparacion_invierte_el_signo(dos_modelos):
    ida = comparar_modelos(dos_modelos, [("base", "nuevo", "a")], n_remuestras=500).set_index("metrica")
    vuelta = comparar_modelos(dos_modelos, [("nuevo", "base", "b")], n_remuestras=500).set_index("metrica")
    assert np.allclose(ida.diferencia, -vuelta.diferencia)
    assert np.allclose(ida.ic95_inferior, -vuelta.ic95_superior)
    assert np.allclose(ida.p_valor, vuelta.p_valor)


def test_modelo_contra_si_mismo(dos_modelos):
    tabla = comparar_modelos({"a": dos_modelos["base"], "b": dos_modelos["base"].copy()}, [("a", "b", "igual")],
                             n_remuestras=200)
    assert (tabla.diferencia == 0).all() and (tabla.ic95_inferior == 0).all() and (tabla.ic95_superior == 0).all()
    assert (tabla.p_valor == 1).all() and (tabla.p_holm == 1).all()
    assert tabla.evidencia.eq("Sin evidencia suficiente para afirmar una diferencia").all()


def test_holm_sobre_toda_la_familia(dos_modelos):
    modelos = {**dos_modelos, "otro": perdidas_por_partido(dos_modelos["base"].index, probabilidades(300, 9),
                                                           dos_modelos["base"].y.to_numpy())}
    tabla = comparar_modelos(modelos, [("base", "nuevo", "1"), ("base", "otro", "2")], n_remuestras=200)
    assert len(tabla) == 6
    # Holm: el menor p se multiplica por el número de pruebas; los ajustados son monótonos y ≥ los originales
    orden = tabla.p_valor.sort_values().index
    assert tabla.p_holm[orden[0]] == pytest.approx(min(1, 6 * tabla.p_valor[orden[0]]))
    assert (tabla.p_holm >= tabla.p_valor - 1e-15).all()
    assert tabla.p_holm[orden].is_monotonic_increasing


# --- datos reales: los tres modelos del test --------------------------------------------------

MODELOS_TEST = ["Solo Elo", "Elo + forma (elegido)", "Completo con estilo (informativo)"]


@requiere("modelo_predicciones.csv")
def test_los_tres_modelos_usan_los_mismos_705_partidos_de_test():
    pred = pd.read_csv(PROCESSED / "modelo_predicciones.csv", dtype={"resultado": str})
    test = pred[(pred.conjunto == "test") & pred.modelo.isin(MODELOS_TEST)]
    por_modelo = {m: d.sort_values("id_partido").reset_index(drop=True) for m, d in test.groupby("modelo")}
    assert set(por_modelo) == set(MODELOS_TEST)
    ref = por_modelo[MODELOS_TEST[0]]
    assert len(ref) == 705 and ref.id_partido.is_unique
    assert set(ref.temporada) == {"2024-25", "2025-26"}
    for d in por_modelo.values():
        assert d.id_partido.equals(ref.id_partido) and d.resultado.equals(ref.resultado)


@requiere("modelo_predicciones.csv", "modelo_comparacion_estadistica.csv")
def test_tabla_exportada_se_reproduce_desde_las_predicciones():
    pred = pd.read_csv(PROCESSED / "modelo_predicciones.csv", dtype={"resultado": str})
    exportada = pd.read_csv(PROCESSED / "modelo_comparacion_estadistica.csv")
    y_de = {"2": 0, "X": 1, "1": 2}
    perdidas = {}
    for m in MODELOS_TEST:
        d = pred[(pred.conjunto == "test") & (pred.modelo == m)].sort_values("id_partido")
        perdidas[m] = perdidas_por_partido(d.id_partido, d[["p_visitante", "p_empate", "p_local"]].to_numpy(),
                                           d.resultado.map(y_de).to_numpy())
    comparaciones = exportada.drop_duplicates("comparacion")[["modelo_base", "modelo_nuevo", "comparacion"]]
    recalculada = comparar_modelos(perdidas, list(comparaciones.itertuples(index=False, name=None)))
    columnas = ["diferencia", "ic95_inferior", "ic95_superior", "p_valor", "p_holm"]
    assert np.allclose(recalculada[columnas], exportada[columnas], rtol=1e-6, atol=1e-9)
    assert (recalculada.evidencia == exportada.evidencia).all()
