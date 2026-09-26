"""Regresión logística ordinal con regularización ridge (fase 4).

Implementación propia porque OrderedModel de statsmodels no admite penalización. Sin
regularización coincide con OrderedModel (comprobado en el notebook 04 y en tests/).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

LAMBDAS = [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]


class RegresionOrdinal:
    """Marcador latente eta = X·beta y dos umbrales t1 < t2.

    P(visitante) = sigma(t1 - eta); P(visitante o empate) = sigma(t2 - eta).
    Las variables se estandarizan con la media y la desviación típica del entrenamiento.
    """

    def __init__(self, lam: float = 0.0):
        self.lam = lam

    def _probs(self, Xs, beta, t1, t2):
        eta = Xs @ beta
        c1, c2 = expit(t1 - eta), expit(t2 - eta)
        return np.column_stack([c1, c2 - c1, 1 - c2])

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.media, self.desv = X.mean(0), X.std(0)
        Xs = (X - self.media) / self.desv
        k = Xs.shape[1]

        def objetivo(par):
            beta, t1, t2 = par[:k], par[k], par[k] + np.exp(par[k + 1])     # garantiza t1 < t2
            prob = np.clip(self._probs(Xs, beta, t1, t2), 1e-12, 1)
            return -np.log(prob[np.arange(len(y)), y]).mean() + self.lam * (beta ** 2).sum()

        inicio = np.r_[np.zeros(k), -0.5, np.log(1.0)]
        sol = minimize(objetivo, inicio, method="L-BFGS-B")
        self.beta, self.t1, self.t2 = sol.x[:k], sol.x[k], sol.x[k] + np.exp(sol.x[k + 1])
        return self

    def predict_proba(self, X):
        """Probabilidades [visitante, empate, local]."""
        Xs = (np.asarray(X, float) - self.media) / self.desv
        return self._probs(Xs, self.beta, self.t1, self.t2)


def ajustar_ordinal(entreno: pd.DataFrame, variables: list, transformar=None, lambdas=LAMBDAS):
    """Elige lambda con la última temporada del entrenamiento y reentrena con todo el entrenamiento.

    Nunca ve los datos de validación ni de test: solo recibe `entreno`. Si `transformar`
    no es None (p. ej. las coordenadas de estilo), se reajusta dentro del split interno
    para que la validación interna tampoco tenga fuga.
    """
    from .metricas import rps

    ultima = entreno.temporada.max()
    interno_e, interno_v = entreno[entreno.temporada < ultima], entreno[entreno.temporada == ultima]
    if transformar is not None:
        interno_e, interno_v = transformar(interno_e, interno_v)
    puntuacion = {lam: rps(RegresionOrdinal(lam).fit(interno_e[variables], interno_e.y)
                           .predict_proba(interno_v[variables]), interno_v.y.to_numpy()).mean()
                  for lam in lambdas}
    mejor = min(puntuacion, key=puntuacion.get)
    return RegresionOrdinal(mejor).fit(entreno[variables], entreno.y), mejor
