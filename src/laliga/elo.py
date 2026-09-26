"""Elo propio de LaLiga (fase 2, notebooks/02_forma_y_fuerza.ipynb).

Parámetros fijados a priori, sin ajustarlos a los datos (evita data snooping):
K = 20, ventaja de local H = 65, multiplicador por margen de goles de eloratings.net,
1500 para todos en 2012/13, sin regresión a la media en verano. Los ascendidos heredan
el Elo medio de los descendidos de la temporada anterior y usan K x 1,5 en sus primeros
5 partidos.

Garantía contra la fuga de información: para cada partido se guarda el Elo ANTES de
actualizarlo con su resultado; solo esas columnas (`*_antes`) se usan como variables.
"""
import numpy as np
import pandas as pd

K, H, FACTOR_ASCENDIDO, N_FORMA, ELO_INICIAL = 20, 65, 1.5, 5, 1500
PUNTUACION = {"1": 1.0, "X": 0.5, "2": 0.0}


def multiplicador_margen(diferencia_goles: int) -> float:
    """World Football Elo Ratings: x1 por 1 gol (o empate), x1,5 por 2, x(11+N)/8 por N >= 3."""
    n = abs(diferencia_goles)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11 + n) / 8


def puntuacion_esperada(elo_local: float, elo_visitante: float, h: float = H) -> float:
    """Puntuación esperada del local (victoria = 1, empate = 0,5), no su probabilidad de ganar."""
    return 1 / (1 + 10 ** (-(elo_local + h - elo_visitante) / 400))


def calcular_elo(partidos: pd.DataFrame) -> pd.DataFrame:
    """Recorre los partidos en orden cronológico y devuelve el Elo antes y después de cada uno.

    `partidos` necesita: id_partido, temporada, fecha, local, visitante, goles_local,
    goles_visitante y resultado ('1', 'X', '2'). Devuelve una fila por partido.
    """
    partidos = partidos.sort_values(["fecha", "id_partido"])
    elo, k_alto_restante = {}, {}
    equipos_anteriores = None
    filas = []

    for temporada, partidos_temporada in partidos.groupby("temporada", sort=True):
        equipos = set(partidos_temporada.local) | set(partidos_temporada.visitante)
        if equipos_anteriores is None:                     # 2012/13: todos empiezan igual
            elo.update({equipo: ELO_INICIAL for equipo in equipos})
        else:
            ascendidos = equipos - equipos_anteriores
            descendidos = equipos_anteriores - equipos
            # sorted(): el orden de un set de textos cambia entre ejecuciones y alteraría la suma en ~1e-13
            elo_heredado = np.mean([elo[equipo] for equipo in sorted(descendidos)])
            for equipo in ascendidos:
                elo[equipo] = elo_heredado
                k_alto_restante[equipo] = N_FORMA

        for p in partidos_temporada.itertuples():
            antes_local, antes_visitante = elo[p.local], elo[p.visitante]
            asc_local = k_alto_restante.get(p.local, 0) > 0
            asc_visitante = k_alto_restante.get(p.visitante, 0) > 0

            esperada = puntuacion_esperada(antes_local, antes_visitante)
            sorpresa = PUNTUACION[p.resultado] - esperada
            margen = multiplicador_margen(p.goles_local - p.goles_visitante)
            elo[p.local] += K * (FACTOR_ASCENDIDO if asc_local else 1) * margen * sorpresa
            elo[p.visitante] -= K * (FACTOR_ASCENDIDO if asc_visitante else 1) * margen * sorpresa

            for equipo, ascendido in ((p.local, asc_local), (p.visitante, asc_visitante)):
                if ascendido:
                    k_alto_restante[equipo] -= 1

            filas.append({
                "id_partido": p.id_partido, "temporada": temporada,
                "elo_local_antes": antes_local, "elo_visitante_antes": antes_visitante,
                "puntuacion_esperada_local": esperada,
                "elo_local_despues": elo[p.local], "elo_visitante_despues": elo[p.visitante],
                "periodo_ascenso_local": asc_local, "periodo_ascenso_visitante": asc_visitante,
            })
        equipos_anteriores = equipos

    return pd.DataFrame(filas)
