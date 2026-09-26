"""Elo (src/laliga/elo.py): fórmulas, reglas de ascendidos y, sobre todo, ausencia de fuga de información."""
import numpy as np
import pandas as pd
import pytest

from conftest import liga_sintetica, requiere
from laliga.elo import (ELO_INICIAL, FACTOR_ASCENDIDO, H, K, N_FORMA, calcular_elo, multiplicador_margen,
                        puntuacion_esperada)


# --- fórmulas -----------------------------------------------------------------

@pytest.mark.parametrize("dif, esperado", [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5), (-2, 1.5), (3, 14 / 8), (5, 16 / 8)])
def test_multiplicador_margen(dif, esperado):
    assert multiplicador_margen(dif) == pytest.approx(esperado)


def test_puntuacion_esperada_simetrica_sin_ventaja():
    for a, b in [(1500, 1500), (1700, 1400), (1350, 1800)]:
        assert puntuacion_esperada(a, b, h=0) + puntuacion_esperada(b, a, h=0) == pytest.approx(1)


def test_ventaja_de_local_entre_iguales():
    # H = 65 implica ~0,592 de puntuación esperada para el local entre dos equipos iguales
    assert puntuacion_esperada(1500, 1500) == pytest.approx(1 / (1 + 10 ** (-H / 400)))
    assert 0.59 < puntuacion_esperada(1500, 1500) < 0.60


def test_400_puntos_de_diferencia_son_10_a_1():
    assert puntuacion_esperada(1900, 1500, h=0) == pytest.approx(10 / 11)


# --- reglas del algoritmo -------------------------------------------------------

def test_liga_sintetica_es_un_calendario_valido(liga):
    # premisa de los tests: como en los datos reales, ningún equipo juega dos veces el mismo día
    largo = pd.concat([liga[["fecha", "local"]].rename(columns={"local": "equipo"}),
                       liga[["fecha", "visitante"]].rename(columns={"visitante": "equipo"})])
    assert not largo.duplicated().any()
    assert not liga.duplicated(["temporada", "local", "visitante"]).any()


def test_primera_temporada_todos_empiezan_igual(liga):
    elo = calcular_elo(liga)
    primeros = elo[elo.temporada == "2012-13"].head(2)
    assert (primeros[["elo_local_antes", "elo_visitante_antes"]] == ELO_INICIAL).all().all()


def test_suma_cero_sin_ascendidos(liga):
    elo = calcular_elo(liga)
    normales = elo[~elo.periodo_ascenso_local & ~elo.periodo_ascenso_visitante]
    cambio = (normales.elo_local_despues - normales.elo_local_antes
              + normales.elo_visitante_despues - normales.elo_visitante_antes)
    assert np.allclose(cambio, 0)


def test_actualizacion_de_un_partido(liga):
    elo = calcular_elo(liga).set_index("id_partido")
    p = liga.iloc[0]
    fila = elo.loc[p.id_partido]
    s = {"1": 1.0, "X": 0.5, "2": 0.0}[p.resultado]
    delta = K * multiplicador_margen(p.goles_local - p.goles_visitante) * (s - puntuacion_esperada(1500, 1500))
    assert fila.elo_local_despues == pytest.approx(1500 + delta)
    assert fila.elo_visitante_despues == pytest.approx(1500 - delta)


def test_ascendido_hereda_media_de_descendidos_y_k_alto(liga):
    elo = calcular_elo(liga)
    largo = pd.concat([
        elo.merge(liga[["id_partido", "fecha", "local"]], on="id_partido").rename(
            columns={"local": "equipo", "elo_local_antes": "antes", "elo_local_despues": "despues",
                     "periodo_ascenso_local": "ascenso"}),
        elo.merge(liga[["id_partido", "fecha", "visitante"]], on="id_partido").rename(
            columns={"visitante": "equipo", "elo_visitante_antes": "antes", "elo_visitante_despues": "despues",
                     "periodo_ascenso_visitante": "ascenso"}),
    ]).sort_values(["fecha", "id_partido"])
    # 2013-14: E sustituye a D -> E empieza con el Elo final de D
    final_d = largo[(largo.equipo == "D") & (largo.temporada == "2012-13")].despues.iloc[-1]
    e = largo[(largo.equipo == "E") & (largo.temporada == "2013-14")]
    assert e.antes.iloc[0] == pytest.approx(final_d)
    # K x 1,5 exactamente en sus N_FORMA primeros partidos
    assert e.ascenso.tolist() == [True] * N_FORMA + [False] * (len(e) - N_FORMA)
    # 2014-15: D vuelve y NO recupera su Elo antiguo: hereda el de E (el descendido)
    final_e = e.despues.iloc[-1]
    d_vuelta = largo[(largo.equipo == "D") & (largo.temporada == "2014-15")]
    assert d_vuelta.antes.iloc[0] == pytest.approx(final_e)
    assert FACTOR_ASCENDIDO > 1


def test_resultado_determinista_ante_el_orden_de_los_sets():
    # la media de los descendidos no puede depender del orden de iteración de un set
    liga = liga_sintetica({"2012-13": ["A", "B", "C", "D", "E", "F"], "2013-14": ["A", "B", "C", "D", "X", "Y"]})
    assert calcular_elo(liga).equals(calcular_elo(liga.sample(frac=1, random_state=3)))


# --- fuga de información ------------------------------------------------------------

def test_elo_antes_no_depende_del_propio_partido(liga):
    original = calcular_elo(liga).set_index("id_partido")
    alterada = liga.copy()
    i = len(liga) // 2
    alterada.loc[i, ["goles_local", "goles_visitante", "resultado"]] = [5, 0, "1"]
    nuevo = calcular_elo(alterada).set_index("id_partido")
    pid = liga.loc[i, "id_partido"]
    cols = ["elo_local_antes", "elo_visitante_antes", "puntuacion_esperada_local"]
    assert np.allclose(original.loc[pid, cols].astype(float), nuevo.loc[pid, cols].astype(float))


def test_invarianza_al_futuro(liga):
    """Cambiar TODOS los resultados desde una fecha no puede alterar ninguna variable 'antes' hasta esa fecha."""
    corte = liga.fecha.sort_values().iloc[len(liga) // 2]
    rng = np.random.default_rng(1)
    futuro = liga.fecha >= corte
    alterada = liga.copy()
    alterada.loc[futuro, "goles_local"] = rng.poisson(3, futuro.sum())
    alterada.loc[futuro, "goles_visitante"] = rng.poisson(0.3, futuro.sum())
    alterada["resultado"] = np.select([alterada.goles_local > alterada.goles_visitante,
                                       alterada.goles_local == alterada.goles_visitante], ["1", "X"], "2")
    a = calcular_elo(liga).set_index("id_partido")
    b = calcular_elo(alterada).set_index("id_partido")
    hasta_corte = liga.loc[liga.fecha <= corte, "id_partido"]
    cols = ["elo_local_antes", "elo_visitante_antes", "puntuacion_esperada_local",
            "periodo_ascenso_local", "periodo_ascenso_visitante"]
    pd.testing.assert_frame_equal(a.loc[hasta_corte, cols], b.loc[hasta_corte, cols])
    # y el cambio sí se nota después (el test no es trivialmente cierto)
    despues = liga.loc[liga.fecha > corte, "id_partido"]
    assert not np.allclose(a.loc[despues, "elo_local_antes"], b.loc[despues, "elo_local_antes"])


# --- coherencia con la tabla publicada ----------------------------------------------

@requiere("partidos.csv", "forma_elo.csv")
def test_forma_elo_csv_coincide_con_calcular_elo(partidos, forma_elo):
    elo = calcular_elo(partidos)
    local = elo.merge(partidos[["id_partido", "local"]], on="id_partido")
    publicado = forma_elo.merge(local, left_on=["id_partido", "equipo"], right_on=["id_partido", "local"])
    assert len(publicado) == len(partidos)
    assert np.allclose(publicado.elo_antes, publicado.elo_local_antes)
    assert np.allclose(publicado.elo_rival_antes, publicado.elo_visitante_antes)
    assert np.allclose(publicado.puntuacion_esperada_elo, publicado.puntuacion_esperada_local)


@requiere("forma_elo.csv")
def test_forma_elo_csv_solo_tiene_columnas_previas_al_partido(forma_elo):
    prohibidas = [c for c in forma_elo.columns
                  if any(x in c for x in ("despues", "goles", "resultado", "puntos_esperados", "xg_favor"))]
    assert prohibidas == []
