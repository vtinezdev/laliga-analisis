"""Esquema de validación temporal de la fase 4 (decisión 24, fijada antes de modelar).

- Validación progresiva (walk-forward) con ventana creciente desde 2014/15: cada temporada
  de validación se predice con un modelo entrenado solo con las temporadas anteriores.
- Test: las dos últimas temporadas, apartadas hasta la evaluación final (plan fijado de antemano en el notebook 04, sección 12).
"""
TEMPORADAS_VALIDACION = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
TEMPORADAS_TEST = ["2024-25", "2025-26"]


def pliegues_walk_forward(temporadas, temporadas_validacion=TEMPORADAS_VALIDACION):
    """Lista de (temporadas de entrenamiento, temporada de validación).

    El entrenamiento de cada pliegue son todas las temporadas estrictamente anteriores
    a la de validación ("2019-20" < "2020-21" en orden de texto = orden cronológico).
    """
    return [(sorted(t for t in set(temporadas) if t < v), v) for v in temporadas_validacion]
