"""Métricas de evaluación de probabilidades 1X2 (fase 4).

Convención de todo el proyecto: `prob` tiene columnas [visitante, empate, local] e
`y` vale 0 (gana el visitante), 1 (empate) o 2 (gana el local).
"""
import numpy as np


def rps(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked Probability Score de cada partido (0 = perfecto; menor es mejor).

    Compara las probabilidades acumuladas en el orden local -> empate -> visitante con las
    del resultado real. Penaliza menos equivocarse por un resultado "vecino" (dar el
    empate como probable cuando gana el local) que por el opuesto.
    """
    real = np.eye(3)[y]
    acum_p = np.cumsum(prob[:, ::-1], axis=1)[:, :2]       # orden local → empate → visitante
    acum_o = np.cumsum(real[:, ::-1], axis=1)[:, :2]
    return 0.5 * ((acum_p - acum_o) ** 2).sum(axis=1)


def log_loss(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """-log de la probabilidad asignada al resultado real, por partido."""
    return -np.log(np.clip(prob[np.arange(len(y)), y], 1e-15, 1))


def evaluar(prob: np.ndarray, y: np.ndarray) -> dict:
    """RPS medio, log-loss medio y acierto (resultado más probable = resultado real)."""
    return {"RPS": rps(prob, y).mean(), "log_loss": log_loss(prob, y).mean(),
            "acierto": (prob.argmax(1) == y).mean()}
