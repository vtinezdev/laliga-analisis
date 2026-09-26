"""Forma reciente (forma_elo.csv, calculada con SQL en el notebook 02): solo puede usar partidos ANTERIORES.

Se recalcula con pandas, de forma independiente, a partir de partidos_equipo.csv.
"""
import numpy as np
import pandas as pd

from conftest import requiere

VENTANA = 5


def forma_pandas(pe: pd.DataFrame, fe: pd.DataFrame) -> pd.DataFrame:
    d = (pe[["id_partido", "equipo", "fecha", "puntos", "xg_favor", "xg_contra"]]
         .merge(fe[["id_partido", "equipo", "etapa"]], on=["id_partido", "equipo"])
         .sort_values(["equipo", "etapa", "fecha", "id_partido"]).reset_index(drop=True))
    d["xg_dif"] = d.xg_favor - d.xg_contra
    g = d.groupby(["equipo", "etapa"])
    previos = lambda s: s.shift(1).rolling(VENTANA, min_periods=1)
    d["forma_puntos"] = g.puntos.transform(lambda s: previos(s).mean())
    d["n_forma"] = g.puntos.transform(lambda s: previos(s).count()).fillna(0)
    d["forma_xg_dif"] = g.xg_dif.transform(lambda s: previos(s).mean())
    d["n_forma_xg"] = g.xg_dif.transform(lambda s: previos(s).count()).fillna(0)
    return d


@requiere("partidos_equipo.csv", "forma_elo.csv")
def test_forma_coincide_con_recalculo_independiente(partidos_equipo, forma_elo):
    d = forma_pandas(partidos_equipo, forma_elo).merge(
        forma_elo, on=["id_partido", "equipo"], suffixes=("_pd", ""))
    assert len(d) == len(forma_elo)
    for col in ("forma_puntos", "forma_xg_dif"):
        assert np.allclose(d[col], d[f"{col}_pd"], equal_nan=True), col
    for col in ("n_forma", "n_forma_xg"):
        assert (d[col] == d[f"{col}_pd"]).all(), col


@requiere("partidos_equipo.csv", "forma_elo.csv")
def test_forma_no_incluye_el_propio_partido(partidos_equipo, forma_elo):
    """Si la forma incluyera el partido actual, se correlacionaría con su resultado mucho más que con el anterior."""
    d = forma_pandas(partidos_equipo, forma_elo)
    publicado = forma_elo.set_index(["id_partido", "equipo"]).forma_puntos
    d["publicado"] = publicado.loc[list(zip(d.id_partido, d.equipo))].to_numpy()
    completo = d[d.n_forma == VENTANA]
    # una forma "con trampa" (ventana que incluye el partido) difiere de la publicada
    trampa = completo.groupby(["equipo", "etapa"]).puntos.transform(lambda s: s.rolling(VENTANA).mean())
    assert not np.allclose(trampa.dropna(), completo.loc[trampa.notna(), "publicado"])
    assert np.allclose(completo.forma_puntos, completo.publicado)


@requiere("partidos_equipo.csv", "forma_elo.csv")
def test_etapas_se_reinician_solo_al_volver_a_primera(forma_elo):
    temporadas = sorted(forma_elo.temporada.unique())
    orden = {t: i for i, t in enumerate(temporadas)}
    et = forma_elo[["equipo", "temporada", "etapa"]].drop_duplicates().sort_values(["equipo", "temporada"])
    et["n"] = et.temporada.map(orden)
    et["salto"] = et.groupby("equipo").n.diff()
    et["cambio_etapa"] = et.groupby("equipo").etapa.diff()
    siguientes = et[et.salto.notna()]
    assert ((siguientes.salto > 1) == (siguientes.cambio_etapa == 1)).all()
    assert (siguientes.cambio_etapa.isin([0, 1])).all()


@requiere("forma_elo.csv")
def test_ascendidos_en_periodo_de_k_alto_sin_forma_previa(forma_elo):
    # los primeros partidos de una etapa (ascensos) empiezan sin forma: nada se hereda de etapas anteriores
    primeros = forma_elo.sort_values(["equipo", "fecha"]).groupby(["equipo", "etapa"]).head(1)
    assert (primeros.n_forma == 0).all() and primeros.forma_puntos.isna().all()
