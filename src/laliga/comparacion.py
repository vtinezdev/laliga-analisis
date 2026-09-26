"""Comparación estadística de modelos evaluados sobre los mismos partidos (notebook 04, sección 18).

Es un análisis de incertidumbre añadido a la evaluación existente: no cambia modelos, variables,
exclusiones ni métricas, que se reutilizan de `laliga.metricas`.

Convención de signo: diferencia = modelo nuevo − modelo base, partido a partido.
- RPS y log-loss (menor es mejor): una diferencia NEGATIVA es una mejora del modelo nuevo.
- Acierto (mayor es mejor): una diferencia POSITIVA es una mejora del modelo nuevo.

Métodos (justificación en el notebook 04, sección 18):
- RPS y log-loss: bootstrap emparejado por partido (IC percentil) + t pareada sobre las diferencias
  de pérdida. Para predicciones a horizonte 1, el contraste de Diebold-Mariano con la corrección de
  Harvey-Leybourne-Newbold es algebraicamente idéntico a esta t, así que no se implementa aparte.
- Acierto: McNemar exacto (binomial sobre los partidos en que solo acierta uno de los dos modelos)
  + IC exacto condicionado (Clopper-Pearson sobre esos partidos), coherente con McNemar. Con pocas
  discordancias el bootstrap de una diferencia de proporciones es discreto y anticonservador.
- Multiplicidad: Holm sobre toda la familia de contrastes (controla la probabilidad de al menos
  un falso positivo con cualquier dependencia entre contrastes).
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .metricas import log_loss, rps

N_REMUESTRAS = 10_000
SEMILLA = 42
ALFA = 0.05
# nombre de la métrica -> True si menor es mejor
METRICAS = {"RPS": True, "log_loss": True, "acierto": False}


def perdidas_por_partido(id_partido, prob: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """RPS, log-loss y acierto de cada partido, con las funciones de `laliga.metricas`.

    El acierto se define como en `evaluar`: el resultado más probable es el real.
    """
    id_partido = np.asarray(id_partido)
    if len(set(id_partido)) != len(id_partido):
        raise ValueError("id_partido repetido: cada partido debe aparecer una sola vez")
    return pd.DataFrame({"y": y, "RPS": rps(prob, y), "log_loss": log_loss(prob, y),
                         "acierto": (prob.argmax(1) == y).astype(float)},
                        index=pd.Index(id_partido, name="id_partido"))


def comprobar_alineacion(perdidas: dict) -> None:
    """Exige los mismos partidos, en el mismo orden y con el mismo resultado real en todos los modelos."""
    nombres = list(perdidas)
    ref = perdidas[nombres[0]]
    for nombre in nombres[1:]:
        otro = perdidas[nombre]
        if len(otro) != len(ref) or not (otro.index == ref.index).all():
            raise ValueError(f"'{nombre}' no tiene los mismos partidos en el mismo orden que '{nombres[0]}'")
        if not (otro.y.to_numpy() == ref.y.to_numpy()).all():
            raise ValueError(f"'{nombre}' y '{nombres[0]}' no tienen el mismo resultado real")


def pesos_bootstrap(n: int, n_remuestras: int = N_REMUESTRAS, semilla: int = SEMILLA, grupos=None) -> np.ndarray:
    """Matriz (remuestras × partidos) con cuántas veces entra cada partido en cada remuestra.

    Sin `grupos`, remuestrea partidos con reemplazo. Con `grupos` (p. ej. la semana), remuestrea
    grupos enteros y cada partido hereda el peso de su grupo (bootstrap por conglomerados).
    La misma matriz se aplica a todos los modelos: eso es lo que hace el bootstrap EMPAREJADO.
    """
    codigos = np.arange(n) if grupos is None else pd.factorize(np.asarray(grupos))[0]
    if len(codigos) != n:
        raise ValueError("grupos debe tener un valor por partido")
    g = codigos.max() + 1
    rng = np.random.default_rng(semilla)
    elegidos = rng.integers(0, g, size=(n_remuestras, g))
    desplazados = elegidos + g * np.arange(n_remuestras)[:, None]
    conteos = np.bincount(desplazados.ravel(), minlength=n_remuestras * g).reshape(n_remuestras, g)
    return conteos[:, codigos].astype(float)


def medias_bootstrap(d: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """Media de `d` en cada remuestra (media ponderada por los pesos del bootstrap)."""
    return pesos @ d / pesos.sum(axis=1)


def intervalo_percentil(distribucion: np.ndarray, nivel: float = 0.95) -> tuple:
    """IC percentil: cuantiles (1 − nivel)/2 y (1 + nivel)/2 de la distribución bootstrap."""
    cola = (1 - nivel) / 2
    return tuple(np.quantile(distribucion, [cola, 1 - cola]))


def t_pareada(d: np.ndarray) -> tuple:
    """t pareada sobre las diferencias de pérdida d = pérdida_nuevo − pérdida_base, por partido.

    H0: la pérdida media esperada de los dos modelos es la misma (E[d] = 0); bilateral, t con n − 1 g.l.
    Supone diferencias aproximadamente independientes y n suficiente para que la media sea
    aproximadamente normal. Equivale exactamente al contraste de Diebold-Mariano a horizonte 1 con
    la corrección de Harvey-Leybourne-Newbold: DM · √((n − 1)/n), con DM calculado con la varianza
    de denominador n, es igual a esta t (comprobado en tests/test_comparacion.py).
    Devuelve (estadístico, p-valor).
    """
    d = np.asarray(d, float)
    if np.ptp(d) == 0:                              # diferencia constante: sin variabilidad
        return (0.0, 1.0) if d[0] == 0 else (np.sign(d[0]) * np.inf, 0.0)
    n, media, desv = len(d), d.mean(), d.std(ddof=1)
    t = media / (desv / np.sqrt(n))
    return t, 2 * stats.t.sf(abs(t), df=n - 1)


def mcnemar_exacto(acierto_base, acierto_nuevo) -> dict:
    """McNemar exacto: binomial(0,5) sobre los partidos en que solo acierta uno de los dos modelos.

    b = acierta solo el base; c = acierta solo el nuevo. Los partidos en que aciertan o fallan
    los dos no informan sobre la diferencia de acierto.
    """
    base, nuevo = np.asarray(acierto_base, bool), np.asarray(acierto_nuevo, bool)
    b, c = int((base & ~nuevo).sum()), int((~base & nuevo).sum())
    p = 1.0 if b + c == 0 else stats.binomtest(c, b + c, 0.5).pvalue
    return {"solo_base": b, "solo_nuevo": c, "p": p}


def ic_acierto_exacto(solo_base: int, solo_nuevo: int, n: int, nivel: float = 0.95) -> tuple:
    """IC de la diferencia de acierto (nuevo − base) condicionado a los partidos discordantes.

    Con m = solo_base + solo_nuevo, la diferencia es m/n · (2π − 1), donde π = proporción de
    discordantes que acierta el nuevo. Se toma el IC exacto de Clopper-Pearson para π: excluye
    π = 0,5 (diferencia 0) exactamente cuando McNemar exacto da p < 1 − nivel.
    """
    m = solo_base + solo_nuevo
    if m == 0:
        return 0.0, 0.0
    ic = stats.binomtest(solo_nuevo, m, 0.5).proportion_ci(confidence_level=nivel, method="exact")
    return m / n * (2 * ic.low - 1), m / n * (2 * ic.high - 1)


def evidencia(diferencia: float, p: float, p_ajustado: float, menor_es_mejor: bool, alfa: float = ALFA) -> str:
    """Lectura de un contraste. Nunca afirma igualdad: la falta de evidencia no demuestra que sean iguales."""
    if p_ajustado < alfa:
        mejora = diferencia < 0 if menor_es_mejor else diferencia > 0
        return ("Evidencia estadística de mejora del modelo nuevo" if mejora
                else "Evidencia estadística de empeoramiento del modelo nuevo")
    if p < alfa:
        return "Sin evidencia suficiente tras corregir por multiplicidad (p sin ajustar < 0,05)"
    return "Sin evidencia suficiente para afirmar una diferencia"


def comparar_modelos(perdidas: dict, comparaciones: list, n_remuestras: int = N_REMUESTRAS,
                     semilla: int = SEMILLA, grupos=None, alfa: float = ALFA) -> pd.DataFrame:
    """Compara pares de modelos en RPS, log-loss y acierto sobre los mismos partidos.

    perdidas: {modelo: salida de perdidas_por_partido}; comparaciones: [(base, nuevo, etiqueta)].
    Todas las comparaciones usan la MISMA matriz de remuestras. La corrección de Holm se aplica a
    la familia completa (comparaciones × métricas) y el IC simultáneo es Bonferroni con ese tamaño.
    """
    comprobar_alineacion(perdidas)
    n = len(next(iter(perdidas.values())))
    pesos = pesos_bootstrap(n, n_remuestras, semilla, grupos)
    n_pruebas = len(comparaciones) * len(METRICAS)
    filas = []
    for base, nuevo, etiqueta in comparaciones:
        for metrica, menor_es_mejor in METRICAS.items():
            valor_base, valor_nuevo = perdidas[base][metrica].to_numpy(), perdidas[nuevo][metrica].to_numpy()
            d = valor_nuevo - valor_base
            fila = {"comparacion": etiqueta, "modelo_base": base, "modelo_nuevo": nuevo, "metrica": metrica,
                    "partidos": n, "valor_base": valor_base.mean(), "valor_nuevo": valor_nuevo.mean(),
                    "diferencia": d.mean(), "cambio_relativo_vs_base": d.mean() / valor_base.mean()}
            if metrica == "acierto":
                mc = mcnemar_exacto(valor_base, valor_nuevo)
                intervalo = lambda nivel: ic_acierto_exacto(mc["solo_base"], mc["solo_nuevo"], n, nivel)
                fila.update(contraste="McNemar exacto", estadistico=np.nan, p_valor=mc["p"])
            else:
                distribucion = medias_bootstrap(d, pesos)
                intervalo = lambda nivel: intervalo_percentil(distribucion, nivel)
                t, p = t_pareada(d)
                fila.update(contraste="t pareada de diferencias de pérdida (= DM con h=1)", estadistico=t, p_valor=p)
                mc = {"solo_base": np.nan, "solo_nuevo": np.nan}
            fila["aciertos_solo_base"], fila["aciertos_solo_nuevo"] = mc["solo_base"], mc["solo_nuevo"]
            fila["ic95_inferior"], fila["ic95_superior"] = intervalo(1 - alfa)
            fila["ic_simultaneo_inferior"], fila["ic_simultaneo_superior"] = intervalo(1 - alfa / n_pruebas)
            filas.append(fila)
    tabla = pd.DataFrame(filas)[[
        "comparacion", "modelo_base", "modelo_nuevo", "metrica", "partidos", "valor_base", "valor_nuevo",
        "diferencia", "cambio_relativo_vs_base", "ic95_inferior", "ic95_superior", "ic_simultaneo_inferior",
        "ic_simultaneo_superior", "contraste", "estadistico", "p_valor", "aciertos_solo_base", "aciertos_solo_nuevo"]]
    tabla["p_holm"] = multipletests(tabla.p_valor, alpha=alfa, method="holm")[1]
    tabla["evidencia"] = [evidencia(f.diferencia, f.p_valor, f.p_holm, METRICAS[f.metrica], alfa)
                          for f in tabla.itertuples()]
    return tabla
