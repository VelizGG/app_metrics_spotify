# Proyecto EDA & Análisis Spotify

Análisis exploratorio de datos (EDA) y análisis avanzado sobre logs de reproducción de Spotify.

## Estructura del Proyecto

```
├─ data/                          # Datos del proyecto
│  ├─ raw/                        # JSON/NDJSON originales
│  └─ curated/                    # Parquet/CSV limpiados
├─ notebooks/
│  └─ 01_EDA_exploratorio.ipynb   # Notebook principal de EDA
├─ src/
│  ├─ __init__.py
│  ├─ data_pipeline.py            # Funciones de ingest y limpieza
│  ├─ features.py                 # Ingeniería de features
│  ├─ eda.py                      # Funciones de EDA
│  └─ models.py                   # Modelos
├─ reports/
│  └─ figures/                    # Gráficos generados
├─ dashboards/
└─ tests/
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
# Transformar datos
python src/data_pipeline.py data/raw/sample.ndjson data/curated/sample.parquet

# Abrir Jupyter
jupyter lab

# Tests
pytest -q
```

## Privacidad

⚠️ No subir datos sensibles en `data/raw/` o `data/curated/`. Estos directorios están en `.gitignore`.
