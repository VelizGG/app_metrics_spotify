# Spotify Analytics - Data Engineering & ML Pipeline

[![CI](https://github.com/VelizGG/app_metrics_spotify/workflows/CI/badge.svg)](https://github.com/VelizGG/app_metrics_spotify/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Portfolio Project** | **Author:** Gabriel Veliz  

Proyecto profesional de **Data Engineering y Machine Learning** que demuestra un pipeline completo de análisis de datos de reproducción de Spotify. Incluye ETL pipeline, feature engineering, análisis exploratorio avanzado, modelos predictivos, y dashboard interactivo.

## 🎯 Skills Demostradas

### Data Engineering
- ✅ **ETL Pipeline**: Ingesta, transformación y carga de datos JSON/Parquet
- ✅ **Data Quality**: Validación, limpieza y normalización de datos
- ✅ **Feature Engineering**: Sessionization, agregaciones, features temporales
- ✅ **Pipeline Automation**: Scripts modulares y reproducibles

### Machine Learning
- ✅ **Supervised Learning**: Clasificación binaria (skip prediction)
- ✅ **Recommendation Systems**: Content-based filtering, collaborative filtering
- ✅ **Clustering**: K-Means para agrupación automática de tracks
- ✅ **Model Evaluation**: ROC-AUC, Precision-Recall, Cross-validation
- ✅ **Feature Importance**: Interpretación de modelos
- ✅ **Production ML**: Model serialization, deployment-ready code

### Data Analysis & Visualization
- ✅ **EDA Avanzado**: Análisis estadístico y visual
- ✅ **Time Series Analysis**: Descomposición, estacionalidad, forecasting
- ✅ **Interactive Dashboards**: Streamlit para exploración de datos
- ✅ **Storytelling**: Insights accionables del negocio

### Software Engineering
- ✅ **Clean Code**: Modular, documentado, testeado
- ✅ **Testing**: pytest con fixtures y coverage
- ✅ **CI/CD**: GitHub Actions para testing automatizado
- ✅ **Version Control**: Git workflow profesional

## 📊 Características del Proyecto

- **Pipeline de datos completo**: Carga, limpieza y transformación de logs JSON/NDJSON
- **Análisis exploratorio profesional**: Visualizaciones interactivas con Plotly
- **Feature engineering avanzado**: Sessionization, agregados por usuario, rolling windows
- **Modelos predictivos**: Skip prediction con Random Forest (ROC-AUC > 0.85)
- **Sistema de recomendación**: Content-based filtering con similitud coseno y embeddings
- **Generación automática de playlists**: Clustering inteligente por temporal patterns, mood y comportamiento
- **Dashboard interactivo**: 8 tabs incluyendo recomendaciones y smart playlists
- **Tests automatizados**: Suite de tests con pytest y coverage >80%
- **CI/CD**: GitHub Actions para testing continuo en múltiples versiones de Python
- **Datos sintéticos**: Generación de datos demo para portfolio público

## 🗂️ Estructura del Proyecto

```
app_metrics_spotify/
├─ data/
│  ├─ raw/                         # JSON/NDJSON originales (NO compartir)
│  ├─ curated/                     # Parquet limpiados
│  ├─ features/                    # Features procesados para ML
│  └─ demo/                        # Datos sintéticos para demo público
├─ notebooks/                      # 📓 Notebooks profesionales para portfolio
│  ├─ 00_data_generation.ipynb     # Generación de datos sintéticos
│  ├─ 01_EDA_exploratorio.ipynb    # Análisis exploratorio completo
│  ├─ 02_feature_engineering.ipynb # Sessionization y features
│  ├─ 03_time_series.ipynb         # Análisis temporal y forecasting
│  ├─ 04_skip_prediction.ipynb     # Modelos predictivos (LR + RF)
│  └─ 05_recommendations_playlists.ipynb # Sistema de recomendación y playlists
├─ src/                            # 🔧 Módulos de código limpio
│  ├─ data_pipeline.py             # ETL pipeline functions
│  ├─ features.py                  # Feature engineering
│  ├─ eda.py                       # Análisis y visualización
│  ├─ models.py                    # ML training & evaluation
│  ├─ recommendations.py           # Sistema de recomendación
│  ├─ playlist_generator.py        # Generación automática de playlists
│  └─ generate_synthetic_data.py   # Generador de datos demo
├─ dashboards/
│  └─ streamlit_app.py             # 📊 Dashboard interactivo
├─ tests/                          # ✅ Test suite
│  └─ test_data_pipeline.py
├─ models/                         # 🤖 Modelos entrenados (.pkl)
├─ reports/
│  └─ figures/                     # Gráficos generados
├─ .github/workflows/
│  └─ ci.yml                       # CI/CD pipeline
├─ requirements.txt
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

### 3. Generar datos sintéticos (para demo público)

⚠️ **Importante**: Los datos reales de Spotify contienen información personal. Para demo público, usar datos sintéticos:

```bash
# Opción 1: Desde notebook
jupyter notebook notebooks/00_data_generation.ipynb

# Opción 2: Desde script
python -c "from src.generate_synthetic_data import generate_synthetic_spotify_data; generate_synthetic_spotify_data(50000, 'data/demo/synthetic_spotify_data.parquet')"
```

### 4. (Opcional) Procesar tus datos reales
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

1. **00_data_generation.ipynb**: Generación de datos sintéticos
   - Creación de datasets demo
   - Preservación de características estadísticas
   - Datos anónimos para portfolio público

2. **01_EDA_exploratorio.ipynb**: Análisis exploratorio completo
   - Carga y validación de datos
   - Estadísticas descriptivas
   - Patrones temporales
   - Top tracks y artistas
   - Análisis de skips

3. **02_feature_engineering.ipynb**: Ingeniería de features
   - Sessionization (agrupación en sesiones de escucha)
   - Agregados por usuario y sesión
   - Rolling windows temporales
   - Features de tracks y artistas

4. **03_time_series.ipynb**: Análisis temporal avanzado
   - Tendencias y estacionalidad
   - Descomposición de series temporales
   - Patrones de uso diario/semanal

5. **04_skip_prediction.ipynb**: Modelos predictivos
   - Predicción de skips (Logistic Regression + Random Forest)
   - Feature importance analysis
   - Evaluación completa de modelos (ROC-AUC, Precision-Recall)
   - Model export para producción

6. **05_recommendations_playlists.ipynb**: Sistema de recomendación y playlists ✨ NUEVO
   - Content-based recommendation engine
   - Track similarity con cosine similarity
   - Generación automática de playlists temáticas
   - Smart playlists por contexto temporal, mood y comportamiento
   - Clustering de tracks similares
   - Evaluación de calidad de recomendaciones

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
