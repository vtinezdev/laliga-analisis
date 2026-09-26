"""Validación de datos: brutos (football-data, Understat), tablas procesadas y diccionario de datos."""
import json

import numpy as np
import pandas as pd
import pytest

import download
from conftest import PROCESSED, RAW, ROOT, requiere

FD_FICHEROS = sorted((RAW / "football_data").glob("SP1_*.csv"))
US_FICHEROS = sorted((RAW / "understat").glob("La_liga_*.json"))
EQUIPOS = pd.read_csv(ROOT / "data" / "equipos.csv")


# --- script de descarga ------------------------------------------------------------

def test_nombres_y_urls_de_temporada():
    assert download.season_label(2014) == "2014-15"
    assert download.season_label(2099) == "2099-00"
    assert download.football_data_url(2014).endswith("/1415/SP1.csv")
    assert download.understat_url(2025).endswith("/La_liga/2025")


def test_validador_football_data_rechaza_ficheros_incompletos():
    cabecera = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    fila = "SP1,23/08/14,A,B,1,0,H\n"
    download.validate_football_data((cabecera + fila * 380 + ",,,,,,\n").encode())   # filas vacías al final: se ignoran
    with pytest.raises(ValueError, match="379 partidos"):
        download.validate_football_data((cabecera + fila * 379).encode())
    with pytest.raises(ValueError, match="HomeTeam"):
        download.validate_football_data(b"a,b\n1,2\n")


def test_validador_understat():
    partido = {"isResult": True}
    download.validate_understat(json.dumps({"dates": [partido] * 380, "teams": {}}).encode())
    with pytest.raises(ValueError):
        download.validate_understat(json.dumps({"dates": [partido] * 200, "teams": {}}).encode())
    with pytest.raises(ValueError):
        download.validate_understat(json.dumps({"dates": []}).encode())


# --- datos brutos de football-data (versionados: siempre se comprueban) ------------------

def test_hay_14_temporadas_de_football_data():
    assert len(FD_FICHEROS) == 14


@pytest.mark.parametrize("fichero", FD_FICHEROS, ids=lambda f: f.stem)
def test_temporada_football_data(fichero):
    df = pd.read_csv(fichero, encoding="utf-8-sig", usecols=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df = df[df.HomeTeam.notna()]
    assert len(df) == 380
    assert df.HomeTeam.nunique() == 20 and set(df.HomeTeam) == set(df.AwayTeam)
    assert not df.duplicated(["HomeTeam", "AwayTeam"]).any()          # cada cruce, una vez en cada campo
    assert (df.HomeTeam != df.AwayTeam).all()
    assert (df[["FTHG", "FTAG"]] >= 0).all().all()
    esperado = np.select([df.FTHG > df.FTAG, df.FTHG == df.FTAG], ["H", "D"], "A")
    assert (df.FTR == esperado).all()                                 # resultado coherente con los goles
    assert set(df.HomeTeam) <= set(EQUIPOS.nombre_football_data)       # todos los nombres tienen equivalencia


def test_tabla_de_equipos():
    assert EQUIPOS.nombre.is_unique and EQUIPOS.nombre_football_data.is_unique
    assert EQUIPOS.nombre_understat.dropna().is_unique


@pytest.mark.skipif(not US_FICHEROS, reason="Understat no está descargado: python src/download.py")
@pytest.mark.parametrize("fichero", US_FICHEROS, ids=lambda f: f.stem)
def test_temporada_understat(fichero):
    datos = json.loads(fichero.read_text(encoding="utf-8"))
    partidos = [m for m in datos["dates"] if m["isResult"]]
    assert len(partidos) == 380
    equipos = {m["h"]["title"] for m in partidos}
    assert len(equipos) == 20 and equipos <= set(EQUIPOS.nombre_understat.dropna())
    assert all(float(m["xG"]["h"]) >= 0 and float(m["xG"]["a"]) >= 0 for m in partidos)


# --- tablas procesadas ------------------------------------------------------------------

@requiere("partidos.csv")
def test_partidos(partidos):
    assert len(partidos) == 5320 and partidos.id_partido.is_unique
    assert (partidos.groupby("temporada").size() == 380).all()
    assert not partidos.duplicated(["temporada", "local", "visitante"]).any()
    ordenado = partidos.sort_values("id_partido")
    assert ordenado.fecha.is_monotonic_increasing                    # id_partido sigue el orden cronológico
    esperado = np.select([partidos.goles_local > partidos.goles_visitante,
                          partidos.goles_local == partidos.goles_visitante], ["1", "X"], "2")
    assert (partidos.resultado == esperado).all()
    con_xg = partidos.temporada >= "2014-15"
    assert partidos.loc[con_xg, "xg_local"].notna().all() and partidos.loc[~con_xg, "xg_local"].isna().all()
    assert (partidos.loc[con_xg, "npxg_local"] <= partidos.loc[con_xg, "xg_local"] + 1e-6).all()


@requiere("partidos.csv")
def test_temporadas_no_se_solapan(partidos):
    # el Elo procesa las temporadas en bloque: ninguna puede tener partidos después del inicio de la siguiente
    limites = partidos.groupby("temporada").fecha.agg(["min", "max"]).sort_index()
    assert (limites["max"].iloc[:-1].to_numpy() < limites["min"].iloc[1:].to_numpy()).all()


@requiere("partidos.csv", "partidos_equipo.csv")
def test_partidos_equipo(partidos, partidos_equipo):
    assert len(partidos_equipo) == 2 * len(partidos)
    assert not partidos_equipo.duplicated(["id_partido", "equipo"]).any()
    assert partidos_equipo.groupby("id_partido").puntos.sum().isin([2, 3]).all()
    assert (partidos_equipo.groupby("id_partido").condicion.nunique() == 2).all()
    # ningún equipo juega dos partidos el mismo día (el orden de la forma y del Elo es inequívoco)
    assert not partidos_equipo.duplicated(["equipo", "fecha"]).any()


@requiere("diccionario_datos.csv")
def test_diccionario_documenta_todas_las_columnas():
    diccionario = pd.read_csv(PROCESSED / "diccionario_datos.csv")
    for fichero in diccionario.fichero.unique():
        ruta = PROCESSED / fichero
        if ruta.exists():
            columnas = list(pd.read_csv(ruta, nrows=1).columns)
            assert set(columnas) == set(diccionario.loc[diccionario.fichero == fichero, "columna"]), fichero
    assert set(diccionario.momento.unique()) >= {"se conoce antes del partido", "se conoce después del partido"}
