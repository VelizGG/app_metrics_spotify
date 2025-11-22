# Proyecto EDA & Análisis Avanzado — Spotify Streaming Analytics

[![CI](https://github.com/VelizGG/app_metrics_spotify/workflows/CI/badge.svg)](https://github.com/VelizGG/app_metrics_spotify/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Análisis exploratorio de datos (EDA) y análisis avanzado sobre logs de reproducción de Spotify (`end_song` events). Incluye estructura de repo, notebooks, scripts reproducibles, tests, CI/CD, y dashboard interactivo.

## 📊 Características

- **Pipeline de datos completo**: Carga, limpieza y transformación de logs JSON/NDJSON
- **Análisis exploratorio**: Visualizaciones interactivas con Plotly
- **Feature engineering**: Sessionization, agregados por usuario, rolling windows
- **Modelos predictivos**: Predicción de skips con scikit-learn
- **Dashboard interactivo**: Streamlit app para explorar métricas en tiempo real
- **Tests automatizados**: Suite de tests con pytest
- **CI/CD**: GitHub Actions para testing continuo

## 🗂️ Estructura del Proyecto

```
app_metrics_spotify/
├─ data/                           # Datos del proyecto (NO subir a GitHub)
│  ├─ raw/                         # JSON/NDJSON originales de Spotify
│  └─ curated/                     # Parquet/CSV limpiados y procesados
├─ notebooks/
│  ├─ 01_EDA_exploratorio.ipynb    # Análisis exploratorio principal
│  ├─ 02_feature_engineering.ipynb # Sessionization y features
│  ├─ 03_time_series.ipynb         # Análisis de series temporales
│  ├─ 04_modelos_skip_prediction.ipynb # Modelos predictivos
│  └─ 05_recomendador_basico.ipynb # Sistema de recomendación
├─ src/
│  ├─ __init__.py
│  ├─ data_pipeline.py             # Funciones de ingest y limpieza
│  ├─ features.py                  # Ingeniería de features
│  ├─ eda.py                       # Funciones de análisis y visualización
│  └─ models.py                    # Entrenamiento y evaluación de modelos
├─ dashboards/
│  └─ streamlit_app.py             # Dashboard interactivo
├─ tests/
│  └─ test_data_pipeline.py        # Tests unitarios
├─ reports/
│  └─ figures/                     # Gráficos generados
├─ .github/workflows/
│  └─ ci.yml                       # Configuración de CI/CD
├─ requirements.txt                # Dependencias del proyecto
├─ .gitignore
└─ README.md
```

## 🚀 Quick Start

### 1. Clonar el repositorio

```bash
git clone https://github.com/VelizGG/app_metrics_spotify.git
cd app_metrics_spotify
```

### 2. Crear entorno virtual e instalar dependencias

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Preparar los datos

Coloca tus datos de Spotify en formato JSON/NDJSON en `data/raw/`. Luego ejecuta el pipeline de transformación:

```bash
python src/data_pipeline.py data/raw/tu_archivo.json data/curated/spotify_data.parquet
```

### 4. Explorar los datos

**Opción A: Notebooks**
```bash
jupyter lab
# Abre notebooks/01_EDA_exploratorio.ipynb
```

**Opción B: Dashboard interactivo**
```bash
streamlit run dashboards/streamlit_app.py
```

## 📝 Uso Detallado

### Pipeline de Datos

El módulo `data_pipeline.py` proporciona funciones para:

```python
from src.data_pipeline import load_ndjson, normalize_columns, clean_data, add_derived_columns

# Cargar datos
df = load_ndjson('data/raw/streaming_history.ndjson')

# Normalizar y limpiar
df = normalize_columns(df)
df = clean_data(df)
df = add_derived_columns(df)

# Guardar en formato optimizado
from src.data_pipeline import save_parquet
save_parquet(df, 'data/curated/clean_data.parquet')
```

### Feature Engineering

```python
from src.features import sessionize, user_aggregates, track_features

# Crear sesiones (gap de 30 minutos)
df = sessionize(df, user_col='username', ts_col='ts')

# Calcular métricas por usuario
users = user_aggregates(df)

# Calcular métricas por track
tracks = track_features(df)
```

### Análisis Exploratorio

```python
from src.eda import *

# Estadísticas generales
stats = summary_stats(df)

# Top tracks y artistas
top_tracks = top_tracks(df, n=20, by='plays')
top_artists = top_artists(df, n=20, by='time')

# Visualizaciones interactivas
fig = plot_plays_over_time(df, freq='D')
fig.show()

fig = plot_hourly_heatmap(df)
fig.show()
```

## 🧪 Tests

Ejecutar la suite completa de tests:

```bash
pytest tests/ -v
```

Con cobertura:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## 📊 Notebooks Disponibles

1. **01_EDA_exploratorio.ipynb**: Análisis exploratorio completo
   - Carga y validación de datos
   - Estadísticas descriptivas
   - Patrones temporales
   - Top tracks y artistas
   - Análisis de skips

2. **02_feature_engineering.ipynb**: Ingeniería de features (próximamente)
   - Sessionization
   - Agregados por usuario y sesión
   - Rolling windows

3. **03_time_series.ipynb**: Análisis temporal avanzado (próximamente)
   - Tendencias y estacionalidad
   - Detección de anomalías

4. **04_modelos_skip_prediction.ipynb**: Modelos predictivos (próximamente)
   - Predicción de skips
   - Feature importance
   - Evaluación de modelos

5. **05_recomendador_basico.ipynb**: Sistema de recomendación (próximamente)
   - Content-based filtering
   - Collaborative filtering

## 🔒 Privacidad y Buenas Prácticas

⚠️ **IMPORTANTE**: 
- **NO subir** datos sensibles en `data/raw/` o `data/curated/`
- Estos directorios están en `.gitignore` por defecto
- Si compartes el proyecto, enmascara o elimina información personal (usernames, IPs, etc.)
- No publiques tokens de API de Spotify en el código

## 📈 Esquema de Datos Esperado

El proyecto asume que tus datos de Spotify tienen el siguiente esquema JSON:

```json
{
  "ts": "2024-01-01 10:00:00",
  "username": "user123",
  "platform": "Windows",
  "ms_played": 180000,
  "conn_country": "US",
  "master_metadata_track_name": "Song Title",
  "master_metadata_album_artist_name": "Artist Name",
  "master_metadata_album_album_name": "Album Name",
  "spotify_track_uri": "spotify:track:...",
  "reason_start": "clickrow",
  "reason_end": "endplay",
  "shuffle": false,
  "skipped": false,
  "offline": false,
  "incognito_mode": false
}
```

## 🤝 Contribuir

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más información.

## 📧 Contacto

Proyecto Link: [https://github.com/VelizGG/app_metrics_spotify](https://github.com/VelizGG/app_metrics_spotify)

---

Desarrollado con ❤️ para el análisis de datos de Spotify
