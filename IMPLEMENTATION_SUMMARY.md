# Sistema de Recomendación y Generación de Playlists - Resumen de Implementación

## 📋 Resumen Ejecutivo

Se ha implementado un **sistema completo de recomendación de canciones y generación automática de playlists** que extiende significativamente las capacidades de machine learning del proyecto Spotify Analytics.

### ✅ Componentes Implementados

1. **recommendations.py** (680+ líneas)
   - Motor de recomendación content-based
   - Cálculo de similitud entre tracks usando cosine similarity
   - Sistema de embeddings densos con reducción dimensional (TruncatedSVD)
   - Múltiples estrategias de recomendación

2. **playlist_generator.py** (600+ líneas)
   - Generación automática de playlists temáticas
   - Clustering de tracks usando K-Means
   - Análisis de patrones temporales y comportamiento

3. **Extensiones a models.py** (200+ líneas nuevas)
   - Funciones de scoring combinado
   - Evaluación de recomendaciones
   - Análisis de diversidad de playlists

4. **notebook 05_recommendations_playlists.ipynb**
   - Demostración completa del sistema
   - Ejemplos de uso con visualizaciones
   - Métricas de evaluación

5. **Integración Streamlit** (400+ líneas)
   - 2 nuevos tabs en el dashboard
   - Interfaz interactiva para recomendaciones
   - Visualización de playlists generadas

6. **test_recommendations.py** (450+ líneas)
   - Suite completa de tests unitarios
   - Cobertura de casos edge
   - Tests para todos los componentes principales

---

## 🎯 Funcionalidades Principales

### 1. Sistema de Recomendación (TrackRecommender)

#### Características Clave:
- **Content-Based Filtering**: Analiza características de tracks (duración, skip rate, patrones temporales)
- **Track Embeddings**: Representación densa de tracks en espacio vectorial
- **Matriz de Similitud**: Cosine similarity entre todos los tracks del catálogo
- **Perfil de Usuario**: Construcción automática de preferencias basadas en historial

#### Estrategias de Recomendación:

```python
# 1. Basado en Favoritos
recommender.recommend_based_on_favorites(n=20, min_similarity=0.6)
# Recomienda tracks similares a los más escuchados del usuario

# 2. Contextual (Temporal)
recommender.recommend_for_context(hour=10, day_of_week=2, n=20)
# Recomienda según hora del día y día de semana

# 3. Skip-Resistant
recommender.recommend_skip_resistant(n=20)
# Tracks con alta probabilidad de completarse (no ser skippeados)

# 4. Híbrido (Recomendado)
recommender.get_recommendations(strategy='hybrid', n=25)
# Combina todas las estrategias para mejor balance
```

#### Métricas de Evaluación:
- **Precision@K**: Precisión de las top-K recomendaciones
- **Recall@K**: Cobertura de tracks relevantes
- **F1@K**: Media armónica de precision y recall
- **Coverage**: Porcentaje del catálogo recomendado
- **Diversity**: Diversidad de artistas en recomendaciones

---

### 2. Generador de Playlists (PlaylistGenerator)

#### Tipos de Playlists Generadas:

**A. Playlists Temporales** (basadas en hora del día):
- ☀️ Morning Energy (6-12h)
- 🌤️ Afternoon Vibes (12-17h)
- 🌆 Evening Chill (17-22h)
- 🌙 Late Night (22-2h)
- 📅 Weekday Focus (Lunes-Viernes)
- 🎉 Weekend Mood (Sábado-Domingo)

**B. Playlists por Comportamiento**:
- 🎯 Never Skip Hits (skip rate < 10%)
- 📚 Deep Focus (tracks largos, consistentes)
- ⚡ Quick Hits (tracks cortos, populares)
- 💎 All-Time Favorites (más reproducidos)
- 🔀 Shuffle Favorites (alta tasa de shuffle)

**C. Playlists por Mood** (inferido):
- 🏃 High Energy (tempo rápido, horas activas)
- 🧘 Relaxation (duración larga, bajo skip rate)
- 🎵 Anytime Classics (baja variabilidad temporal)

**D. Playlists por Artista**:
- 🎤 Best of [Artista] (top tracks de artistas favoritos)

**E. Playlists por Clustering**:
- 🧩 Cluster Mix 1-N (agrupación inteligente por características similares)

**F. Playlist de Descubrimiento**:
- 🔍 Rediscover (tracks con escuchas moderadas, bajo skip rate)

#### Uso:

```python
# Generar todas las playlists automáticamente
generator = create_playlist_generator(df, generate_all=True)

# O generar selectivamente
generator.generate_temporal_playlists()
generator.generate_behavior_playlists()
generator.generate_cluster_playlists(n_clusters=5)

# Acceder a playlists
morning_playlist = generator.get_playlist('Morning Energy')
never_skip = generator.get_playlist('Never Skip Hits')

# Analizar diversidad
diversity_metrics = analyze_playlist_diversity(morning_playlist)
```

---

## 🏗️ Arquitectura Técnica

### Stack Tecnológico:
- **Similarity**: scikit-learn (cosine_similarity)
- **Dimensionality Reduction**: TruncatedSVD
- **Clustering**: K-Means, DBSCAN
- **Embeddings**: StandardScaler + feature engineering
- **Visualization**: Plotly, Streamlit
- **Testing**: pytest

### Pipeline de Recomendación:

```
1. Data Ingestion (df histórico)
   ↓
2. Feature Engineering
   - Agregados por track (avg_duration, skip_rate, popularity)
   - Agregados por usuario (favoritos, patrones temporales)
   ↓
3. Track Embeddings
   - Normalización (StandardScaler)
   - Reducción dimensional (TruncatedSVD si necesario)
   ↓
4. Similarity Matrix
   - Cosine similarity entre todos los tracks
   ↓
5. Recommendation Scoring
   - Combina similitud + popularidad + skip resistance
   ↓
6. Ranking & Filtering
   - Diversificación
   - Filtrado por contexto
   ↓
7. Output: Top-N Recomendaciones
```

### Pipeline de Playlists:

```
1. Track Features Aggregation
   ↓
2. Temporal Analysis
   - Típica hora de escucha por track
   - Día de semana preferido
   ↓
3. Behavioral Analysis
   - Skip rate
   - Completion rate
   - Shuffle preference
   ↓
4. Clustering (opcional)
   - K-Means sobre features normalizados
   ↓
5. Playlist Generation
   - Filtrado por criterios (hora, skip rate, etc.)
   - Ranking por popularidad o scores
   ↓
6. Output: Múltiples playlists temáticas
```

---

## 📊 Métricas de Calidad

### Recomendaciones:
- **Coverage**: % del catálogo recomendado
- **Artist Diversity**: Diversidad de artistas en top-K
- **Avg Popularity**: Popularidad promedio de recomendaciones
- **Avg Completion Rate**: Tasa de completitud esperada
- **Precision@K**: Si hay test set disponible
- **Recall@K**: Si hay test set disponible

### Playlists:
- **n_tracks**: Número de tracks en playlist
- **n_artists**: Artistas únicos
- **artist_diversity**: Ratio artistas/tracks
- **avg_skip_rate**: Skip rate promedio de la playlist
- **hour_std**: Variabilidad temporal (dispersión de horas)
- **duration_std**: Variabilidad en duración de tracks

---

## 🎨 Integración en Dashboard

### Tab "✨ Recomendaciones":
- Selector de estrategia (Híbrida, Favoritos, Contextual, Anti-Skip)
- Control de número de recomendaciones (5-50)
- Configuración de contexto temporal para recomendaciones contextuales
- Visualización de perfil de usuario
- Tabla interactiva con recomendaciones
- Gráficos de distribución (artistas, horas)
- Descarga de recomendaciones en CSV

### Tab "🎧 Smart Playlists":
- Checkboxes para seleccionar tipos de playlists a generar
- Control de número de clusters para clustering
- Lista de todas las playlists generadas
- Selector de playlist para ver detalles
- Métricas de cada playlist (tracks, artistas, diversidad, skip rate)
- Visualizaciones por playlist (top artistas, distribución temporal)
- Exportación de playlists individuales

---

## 🧪 Testing

### Cobertura de Tests:

**TrackRecommender** (12 tests):
- Inicialización
- Construcción de features
- Creación de embeddings
- Matriz de similitud
- Perfil de usuario
- Tracks similares
- Recomendaciones contextuales
- Recomendaciones por favoritos
- Recomendaciones skip-resistant
- Recomendaciones híbridas
- Evaluación de recomendaciones

**PlaylistGenerator** (11 tests):
- Inicialización
- Playlists temporales
- Playlists por comportamiento
- Playlists por mood
- Playlists por artista
- Clustering
- Playlist de descubrimiento
- Generación completa
- Obtención de playlist
- Listado de playlists

**Model Functions** (3 tests):
- Recommendation scoring
- Evaluación de calidad
- Análisis de diversidad

**Edge Cases** (3 tests):
- DataFrame vacío
- Single track
- Columnas faltantes

### Ejecución:
```bash
pytest tests/test_recommendations.py -v
```

---

## 📈 Próximas Mejoras Sugeridas

### Corto Plazo:
1. **Collaborative Filtering**: Añadir filtrado colaborativo cuando haya múltiples usuarios
2. **Hybrid Model**: Combinar content-based + collaborative filtering
3. **Audio Features**: Integrar datos de Spotify API (tempo, energy, valence)
4. **A/B Testing**: Framework para evaluar estrategias de recomendación

### Mediano Plazo:
1. **Deep Learning**: Neural networks para embeddings más sofisticados
2. **Sequential Models**: RNN/LSTM para considerar orden de reproducción
3. **Context-Aware**: Integrar más contexto (ubicación, clima, actividad)
4. **Cold Start**: Mejor manejo de tracks nuevos sin historial

### Largo Plazo:
1. **Reinforcement Learning**: Optimizar recomendaciones por feedback
2. **Multi-objective**: Optimizar por múltiples objetivos (diversidad + relevancia)
3. **Explainability**: Sistema de explicación de por qué se recomienda cada track
4. **Real-time**: Sistema de recomendación en tiempo real con streaming data

---

## 🎓 Valor para Portfolio

Este sistema demuestra:

✅ **Machine Learning Avanzado**:
- Content-based filtering
- Clustering no supervisado
- Feature engineering sofisticado
- Evaluación de sistemas de recomendación

✅ **Software Engineering**:
- Código modular y reutilizable
- Arquitectura escalable
- Testing comprehensivo
- Documentación profesional

✅ **Product Thinking**:
- Múltiples estrategias para diferentes use cases
- UX considerado en dashboard
- Métricas de calidad bien definidas
- Features orientadas a usuario final

✅ **Data Science End-to-End**:
- Desde datos crudos hasta producto funcional
- Análisis exploratorio → Modelado → Deployment
- Evaluación rigurosa
- Iteración y mejora continua

---

## 📝 Uso Rápido

### Recomendaciones:
```python
from recommendations import create_recommender_from_data

# Inicializar
recommender = create_recommender_from_data(df)

# Obtener recomendaciones
recs = recommender.get_recommendations(
    strategy='hybrid',
    n=20,
    hour=10,
    day_of_week=2
)
```

### Playlists:
```python
from playlist_generator import create_playlist_generator

# Generar todas
generator = create_playlist_generator(df, generate_all=True)

# Ver disponibles
playlists = generator.list_playlists()

# Obtener una específica
morning = generator.get_playlist('Morning Energy')
```

### Dashboard:
```bash
streamlit run dashboards/streamlit_app.py
```

Navegar a tabs "✨ Recomendaciones" y "🎧 Smart Playlists"

---

**Desarrollado por Gabriel Veliz** | [GitHub](https://github.com/VelizGG)
