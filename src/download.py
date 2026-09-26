"""Descarga los datos brutos de LaLiga de football-data.co.uk y Understat.

Los ficheros se guardan tal cual los sirve cada fuente en data/raw/ y no se
modifican nunca: cualquier corrección se hace en el paso de limpieza.
Cada descarga queda anotada en data/raw/descargas.csv (URL y fecha), porque
las fuentes corrigen sus datos con el tiempo.

Uso:  python src/download.py
"""
import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LOG = RAW / "descargas.csv"

SEASONS = range(2014, 2026)          # año de inicio: 2014 -> temporada 2014/15
# Temporadas previas solo para calentar el Elo (no se analizan). Understat no tiene datos antes de 2014.
BURN_IN_SEASONS = range(2012, 2014)
MATCHES_PER_SEASON = 380             # 20 equipos x 19 rivales x 2 (ida y vuelta)
PAUSE_SECONDS = 2                    # cortesía con servidores mantenidos por voluntarios

BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto educativo de análisis de LaLiga)"}


def season_label(start_year):
    """2014 -> '2014-15'"""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


# --- football-data.co.uk ----------------------------------------------------

def football_data_url(start_year):
    code = f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"      # 2014 -> '1415'
    return f"https://www.football-data.co.uk/mmz4281/{code}/SP1.csv"


def validate_football_data(content):
    text = content.decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "HomeTeam" not in rows[0]:
        raise ValueError("no parece un CSV de football-data (falta la columna HomeTeam)")
    matches = [r for r in rows if r["HomeTeam"]]                     # algunos ficheros traen filas vacías al final
    if len(matches) != MATCHES_PER_SEASON:
        raise ValueError(f"{len(matches)} partidos, se esperaban {MATCHES_PER_SEASON}")


# --- Understat --------------------------------------------------------------

def understat_url(start_year):
    return f"https://understat.com/getLeagueData/La_liga/{start_year}"


def validate_understat(content):
    data = json.loads(content)
    if not {"dates", "teams"} <= data.keys():
        raise ValueError("el JSON no tiene los bloques 'dates' y 'teams'")
    played = [m for m in data["dates"] if m["isResult"]]
    if len(played) != MATCHES_PER_SEASON:
        raise ValueError(f"{len(played)} partidos jugados, se esperaban {MATCHES_PER_SEASON}")


SOURCES = {
    "football_data": {
        "seasons": [*BURN_IN_SEASONS, *SEASONS],
        "url": football_data_url,
        "filename": lambda y: f"SP1_{season_label(y)}.csv",
        "headers": BROWSER_HEADERS,
        "validate": validate_football_data,
    },
    "understat": {
        "seasons": SEASONS,
        "url": understat_url,
        "filename": lambda y: f"La_liga_{season_label(y)}.json",
        # Understat solo entrega el JSON a peticiones hechas desde su propia web
        "headers": {**BROWSER_HEADERS, "X-Requested-With": "XMLHttpRequest"},
        "validate": validate_understat,
    },
}


def log_download(path, url):
    is_new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["fichero", "url", "descargado_utc"])
        writer.writerow([path.relative_to(RAW).as_posix(), url,
                         datetime.now(timezone.utc).isoformat(timespec="seconds")])


def get_with_retries(url, headers, attempts=3):
    """Reintenta fallos pasajeros (servidor lento o caído), esperando más en cada intento."""
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()                              # 4xx/5xx -> error
            return response
        except requests.RequestException as error:
            if attempt == attempts:
                raise
            wait = 10 * attempt
            print(f"    fallo ({error.__class__.__name__}), reintento en {wait} s")
            time.sleep(wait)


def download(source_name, start_year):
    source = SOURCES[source_name]
    path = RAW / source_name / source["filename"](start_year)
    if path.exists():
        print(f"  = {path.name} ya existe")
        return

    url = source["url"](start_year)
    response = get_with_retries(url, source["headers"])
    source["validate"](response.content)                             # contenido incorrecto -> error

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)                               # bytes tal cual, sin transformar
    log_download(path, url)
    print(f"  + {path.name} ({len(response.content) / 1024:.0f} KB)")
    time.sleep(PAUSE_SECONDS)


def main():
    for source_name, source in SOURCES.items():
        print(source_name)
        for start_year in source["seasons"]:
            download(source_name, start_year)


if __name__ == "__main__":
    main()
