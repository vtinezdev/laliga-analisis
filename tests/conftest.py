"""Utilidades comunes de los tests.

Hay dos tipos de tests:
- Unitarios, con datos sintéticos pequeños: siempre se ejecutan.
- De datos, sobre data/raw y data/processed: los de football-data siempre (sus CSV están en
  el repositorio); los que necesitan Understat o las tablas procesadas se saltan si todavía
  no se han generado (python src/download.py y los notebooks, o python src/pipeline.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def requiere(*ficheros: str):
    """Salta el test si falta alguno de los ficheros de data/processed."""
    faltan = [f for f in ficheros if not (PROCESSED / f).exists()]
    return pytest.mark.skipif(bool(faltan), reason=f"faltan {faltan}: ejecuta python src/pipeline.py")


@pytest.fixture(scope="session")
def partidos():
    return pd.read_csv(PROCESSED / "partidos.csv", parse_dates=["fecha"], dtype={"resultado": str})


@pytest.fixture(scope="session")
def partidos_equipo():
    return pd.read_csv(PROCESSED / "partidos_equipo.csv", parse_dates=["fecha"])


@pytest.fixture(scope="session")
def forma_elo():
    return pd.read_csv(PROCESSED / "forma_elo.csv", parse_dates=["fecha"])


def calendario_doble_vuelta(equipos: list) -> list:
    """Jornadas por el método del círculo: cada equipo juega una vez por jornada; la vuelta invierte los campos."""
    n, ida = len(equipos), []
    rotacion = equipos[:]
    for _ in range(n - 1):
        ida.append([(rotacion[i], rotacion[n - 1 - i]) for i in range(n // 2)])
        rotacion = [rotacion[0], rotacion[-1], *rotacion[1:-1]]
    return ida + [[(v, l) for l, v in jornada] for jornada in ida]


def liga_sintetica(equipos_por_temporada: dict, semilla: int = 0) -> pd.DataFrame:
    """Liga a doble vuelta con goles aleatorios, en el formato de partidos.csv.

    Cada jornada es un día distinto; id_partido sigue el orden (fecha, local) como en la fase 1.
    """
    rng = np.random.default_rng(semilla)
    filas, dia = [], pd.Timestamp("2012-08-18")
    for temporada, equipos in equipos_por_temporada.items():
        for local_visitante in calendario_doble_vuelta(list(equipos)):
            for local, visitante in local_visitante:
                gl, gv = rng.poisson(1.5), rng.poisson(1.1)
                filas.append({"temporada": temporada, "fecha": dia, "local": local, "visitante": visitante,
                              "goles_local": gl, "goles_visitante": gv,
                              "resultado": "1" if gl > gv else ("X" if gl == gv else "2")})
            dia += pd.Timedelta(days=7)
        dia += pd.Timedelta(days=60)                           # verano
    df = pd.DataFrame(filas).sort_values(["fecha", "local"]).reset_index(drop=True)
    df.insert(0, "id_partido", np.arange(1, len(df) + 1))
    return df


@pytest.fixture
def liga():
    """Tres temporadas: sube E (baja D) y después vuelve D (baja E)."""
    return liga_sintetica({"2012-13": ["A", "B", "C", "D"],
                           "2013-14": ["A", "B", "C", "E"],
                           "2014-15": ["A", "B", "C", "D"]})
