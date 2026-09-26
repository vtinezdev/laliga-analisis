"""Coordenadas de estilo sin fuga de información (fase 4, notebooks/04_modelo.ipynb).

Cada variable de estilo llega ya estandarizada (`z_<variable>_local` / `_visit`) con la
referencia de la temporada anterior. Aquí se descuenta la calidad (regresión sobre el Elo
medio de la ventana) con parámetros ajustados SOLO con `entreno`, y se aplican tal cual a
`evaluacion`: los datos de validación o test nunca influyen en la transformación.
"""
import numpy as np
import pandas as pd

VARIABLES_ESTILO = ["ppda", "ppda_rival", "profundos_pp", "xg_por_tiro", "xg_por_tiro_contra", "faltas_pp"]
NOMBRES_COORD = {"ppda": "presion", "ppda_rival": "posesion", "profundos_pp": "profundos",
                 "xg_por_tiro": "xg_por_tiro", "xg_por_tiro_contra": "xg_por_tiro_contra", "faltas_pp": "faltas"}
SIGNO = {"ppda": -1}                                   # presión = −PPDA
COORD = [f"{n}_{s}" for s in ("local", "visit") for n in NOMBRES_COORD.values()]
INTERACCIONES = ["I1_presion_local_x_posesion_visit", "I2_posesion_local_x_presion_visit"]


def coordenadas_estilo(entreno: pd.DataFrame, evaluacion: pd.DataFrame):
    """Devuelve copias con las 12 coordenadas de estilo y las 2 interacciones, ajustando todo con `entreno`."""
    entreno, evaluacion = entreno.copy(), evaluacion.copy()
    # la regresión usa los dos lados (local y visitante) del entrenamiento como observaciones
    for v in VARIABLES_ESTILO:
        x = np.concatenate([entreno.z_elo_medio_local, entreno.z_elo_medio_visit])
        y = np.concatenate([entreno[f"z_{v}_local"], entreno[f"z_{v}_visit"]])
        b, a = np.polyfit(x, y, deg=1)
        escala = np.std(y - (a + b * x))
        for df in (entreno, evaluacion):
            for s in ("local", "visit"):
                residuo = (df[f"z_{v}_{s}"] - (a + b * df[f"z_elo_medio_{s}"])) / escala
                df[f"{NOMBRES_COORD[v]}_{s}"] = SIGNO.get(v, 1) * residuo
    for df in (entreno, evaluacion):
        df["I1_presion_local_x_posesion_visit"] = df.presion_local * df.posesion_visit
        df["I2_posesion_local_x_presion_visit"] = df.posesion_local * df.presion_visit
    return entreno, evaluacion
