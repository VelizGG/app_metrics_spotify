"""Dashboard interactivo con Streamlit para visualizar métricas de Spotify."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_pipeline import load_curated_data
from eda import *
from features import user_aggregates, sessionize, session_features

# Configuración de la página
st.set_page_config(
    page_title="Spotify Analytics Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎵 Spotify Streaming Analytics Dashboard")
st.markdown("---")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Cargar datos
@st.cache_data
def load_data():
    """Carga los datos curados con caché."""
    data_path = Path(__file__).parent.parent / 'data' / 'curated' / 'spotify_data.parquet'
    try:
        df = load_curated_data(data_path)
        return df
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo de datos curados.")
        st.info("Por favor, ejecuta primero el pipeline de transformación:\n"
                "```python src/data_pipeline.py data/raw/sample.json data/curated/spotify_data.parquet```")
        return None

df = load_data()

if df is not None:
    # Filtros en sidebar
    st.sidebar.subheader("📊 Filtros")
    
    # Filtro de fecha
    date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=(df['ts'].min(), df['ts'].max()),
        min_value=df['ts'].min().date(),
        max_value=df['ts'].max().date()
    )
    
    # Filtro de plataforma
    platforms = ['Todas'] + sorted(df['platform'].unique().tolist())
    selected_platform = st.sidebar.selectbox("Plataforma", platforms)
    
    # Aplicar filtros
    df_filtered = df.copy()
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['ts'].dt.date >= date_range[0]) &
            (df_filtered['ts'].dt.date <= date_range[1])
        ]
    if selected_platform != 'Todas':
        df_filtered = df_filtered[df_filtered['platform'] == selected_platform]
    
    # Métricas principales
    st.header("📈 Métricas Principales")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reproducciones", f"{len(df_filtered):,}")
    
    with col2:
        st.metric("Usuarios Únicos", f"{df_filtered['username'].nunique():,}")
    
    with col3:
        total_hours = df_filtered['ms_played'].sum() / 3600000
        st.metric("Total Horas", f"{total_hours:,.1f}")
    
    with col4:
        st.metric("Tracks Únicos", f"{df_filtered['spotify_track_uri'].nunique():,}")
    
    st.markdown("---")
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Temporal", "🎵 Top Charts", "👤 Usuarios", "🔍 Análisis Avanzado"])
    
    with tab1:
        st.subheader("Análisis Temporal")
        
        # Gráfico de reproducciones en el tiempo
        freq = st.selectbox("Frecuencia", ['D', 'W', 'M'], format_func=lambda x: {'D': 'Diaria', 'W': 'Semanal', 'M': 'Mensual'}[x])
        fig_time = plot_plays_over_time(df_filtered, freq=freq)
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Heatmap de actividad
        st.subheader("Heatmap de Actividad")
        fig_heatmap = plot_hourly_heatmap(df_filtered)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.subheader("Top Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Tracks")
            n_tracks = st.slider("Número de tracks", 5, 50, 20, key='tracks')
            by_tracks = st.radio("Criterio", ['plays', 'time'], format_func=lambda x: 'Reproducciones' if x == 'plays' else 'Tiempo', key='by_tracks')
            fig_tracks = plot_top_tracks(df_filtered, n=n_tracks, by=by_tracks)
            st.plotly_chart(fig_tracks, use_container_width=True)
        
        with col2:
            st.subheader("Top Artistas")
            n_artists = st.slider("Número de artistas", 5, 50, 20, key='artists')
            by_artists = st.radio("Criterio", ['plays', 'time'], format_func=lambda x: 'Reproducciones' if x == 'plays' else 'Tiempo', key='by_artists')
            fig_artists = plot_top_artists(df_filtered, n=n_artists, by=by_artists)
            st.plotly_chart(fig_artists, use_container_width=True)
    
    with tab3:
        st.subheader("Análisis de Usuarios")
        
        # Calcular agregados por usuario
        with st.spinner("Calculando métricas por usuario..."):
            users_df = user_aggregates(df_filtered)
        
        st.dataframe(users_df.head(20), use_container_width=True)
        
        # Distribución de horas reproducidas por usuario
        fig_user_dist = px.histogram(
            users_df, 
            x='total_hours_played',
            nbins=30,
            title='Distribución de Horas Reproducidas por Usuario',
            labels={'total_hours_played': 'Horas Totales'}
        )
        st.plotly_chart(fig_user_dist, use_container_width=True)
    
    with tab4:
        st.subheader("Análisis Avanzado")
        
        # Distribución de plataformas
        fig_platform = plot_platform_distribution(df_filtered)
        st.plotly_chart(fig_platform, use_container_width=True)
        
        # Distribución de duración
        fig_duration = plot_listening_duration_dist(df_filtered)
        st.plotly_chart(fig_duration, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Spotify Analytics Dashboard** | Desarrollado con Streamlit 🎵")

else:
    st.warning("No hay datos disponibles. Por favor, carga los datos curados primero.")
