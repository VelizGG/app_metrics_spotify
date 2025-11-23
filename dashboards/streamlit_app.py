"""Dashboard interactivo con Streamlit para visualizar métricas de Spotify."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_pipeline import load_curated_data
from eda import *
from features import sessionize, session_features, track_features, rolling_user_features

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
        st.metric("Artistas Únicos", f"{df_filtered['master_metadata_album_artist_name'].nunique():,}")
    
    with col3:
        total_hours = df_filtered['ms_played'].sum() / 3600000
        st.metric("Total Horas", f"{total_hours:,.1f}")
    
    with col4:
        st.metric("Tracks Únicos", f"{df_filtered['spotify_track_uri'].nunique():,}")
    
    st.markdown("---")
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Temporal", 
        "🎵 Top Charts", 
        "🌍 Análisis General",
        "🎯 Sesiones",
        "🔍 Análisis Avanzado",
        "📥 Exportar Datos"
    ])
    
    with tab1:
        st.subheader("Análisis Temporal")
        
        # Gráfico de reproducciones en el tiempo
        freq = st.selectbox("Frecuencia", ['D', 'W', 'M'], format_func=lambda x: {'D': 'Diaria', 'W': 'Semanal', 'M': 'Mensual'}[x])
        fig_time = plot_plays_over_time(df_filtered, freq=freq)
        st.plotly_chart(fig_time, width='stretch')
        
        # Heatmap de actividad
        st.subheader("Heatmap de Actividad")
        fig_heatmap = plot_hourly_heatmap(df_filtered)
        st.plotly_chart(fig_heatmap, width='stretch')
    
    with tab2:
        st.subheader("Top Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Tracks")
            n_tracks = st.slider("Número de tracks", 5, 50, 20, key='tracks')
            by_tracks = st.radio("Criterio", ['plays', 'time'], format_func=lambda x: 'Reproducciones' if x == 'plays' else 'Tiempo', key='by_tracks')
            fig_tracks = plot_top_tracks(df_filtered, n=n_tracks, by=by_tracks)
            st.plotly_chart(fig_tracks, width='stretch')
        
        with col2:
            st.subheader("Top Artistas")
            n_artists = st.slider("Número de artistas", 5, 50, 20, key='artists')
            by_artists = st.radio("Criterio", ['plays', 'time'], format_func=lambda x: 'Reproducciones' if x == 'plays' else 'Tiempo', key='by_artists')
            fig_artists = plot_top_artists(df_filtered, n=n_artists, by=by_artists)
            st.plotly_chart(fig_artists, width='stretch')
    
    with tab3:
        st.subheader("Análisis de Escucha")
        
        # Métricas generales
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Países desde donde escuchas", df_filtered['conn_country'].nunique())
            st.metric("Álbumes únicos", df_filtered['master_metadata_album_album_name'].nunique())
        
        with col2:
            avg_duration = df_filtered['minutes_played'].mean()
            st.metric("Duración promedio (min)", f"{avg_duration:.2f}")
            skip_rate = df_filtered['skipped'].mean() * 100 if 'skipped' in df_filtered.columns else 0
            st.metric("Skip Rate", f"{skip_rate:.1f}%")
        
        # Top países
        st.subheader("Top Países")
        country_counts = df_filtered['conn_country'].value_counts().head(10).reset_index()
        country_counts.columns = ['País', 'Reproducciones']
        fig_countries = px.bar(country_counts, x='País', y='Reproducciones', 
                              title='Top 10 Países')
        fig_countries.update_traces(marker_color='#1DB954')
        st.plotly_chart(fig_countries, width='stretch')
    
    with tab4:
        st.subheader("🎯 Análisis de Sesiones")
        
        # Calcular sesiones
        with st.spinner("Calculando sesiones de escucha..."):
            df_sessions = sessionize(df_filtered.copy(), user_col=None)
            sessions_summary = session_features(df_sessions)
        
        # Métricas de sesiones
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sesiones", f"{sessions_summary['session_id'].nunique():,}")
        with col2:
            avg_duration = sessions_summary['session_duration_minutes'].mean()
            st.metric("Duración Promedio", f"{avg_duration:.1f} min")
        with col3:
            avg_tracks = sessions_summary['tracks_played'].mean()
            st.metric("Tracks por Sesión", f"{avg_tracks:.1f}")
        with col4:
            if 'artist_diversity' in sessions_summary.columns:
                diversity = sessions_summary['artist_diversity'].mean()
                st.metric("Diversidad Artistas", f"{diversity:.2%}")
        
        # Distribución de duración de sesiones
        st.subheader("Distribución de Duración de Sesiones")
        fig_session_dur = px.histogram(
            sessions_summary, 
            x='session_duration_minutes',
            nbins=50,
            title='Duración de Sesiones de Escucha',
            labels={'session_duration_minutes': 'Duración (minutos)'}
        )
        fig_session_dur.update_traces(marker_color='#1DB954')
        st.plotly_chart(fig_session_dur, width='stretch')
        
        # Tracks por sesión
        col1, col2 = st.columns(2)
        with col1:
            fig_tracks = px.histogram(
                sessions_summary,
                x='tracks_played',
                nbins=30,
                title='Tracks Reproducidos por Sesión',
                labels={'tracks_played': 'Número de Tracks'}
            )
            fig_tracks.update_traces(marker_color='#1DB954')
            st.plotly_chart(fig_tracks, width='stretch')
        
        with col2:
            if 'unique_artists' in sessions_summary.columns:
                fig_artists = px.histogram(
                    sessions_summary,
                    x='unique_artists',
                    nbins=30,
                    title='Artistas Únicos por Sesión',
                    labels={'unique_artists': 'Número de Artistas'}
                )
                fig_artists.update_traces(marker_color='#1DB954')
                st.plotly_chart(fig_artists, width='stretch')
        
        # Top sesiones más largas
        st.subheader("Top 10 Sesiones Más Largas")
        top_sessions = sessions_summary.nlargest(10, 'session_duration_minutes')[
            ['session_start', 'session_duration_minutes', 'tracks_played', 'minutes_played']
        ].copy()
        top_sessions['session_start'] = pd.to_datetime(top_sessions['session_start'])
        st.dataframe(top_sessions, width='stretch')
    
    with tab5:
        st.subheader("🔍 Análisis Avanzado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de plataformas
            st.subheader("Distribución por Plataforma")
            fig_platform = plot_platform_distribution(df_filtered)
            st.plotly_chart(fig_platform, width='stretch')
        
        with col2:
            # Skip rate por hora
            if 'skipped' in df_filtered.columns:
                st.subheader("Skip Rate por Hora")
                skip_by_hour = df_filtered.groupby('hour')['skipped'].mean().reset_index()
                fig_skip = px.line(
                    skip_by_hour, 
                    x='hour', 
                    y='skipped',
                    title='Tendencia de Skips Durante el Día',
                    labels={'hour': 'Hora del día', 'skipped': 'Skip Rate'}
                )
                fig_skip.update_traces(line_color='#1DB954')
                fig_skip.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_skip, width='stretch')
        
        # Distribución de duración
        st.subheader("Distribución de Duración de Reproducción")
        fig_duration = plot_listening_duration_dist(df_filtered)
        st.plotly_chart(fig_duration, width='stretch')
        
        # Análisis de razones de inicio y fin
        st.subheader("Razones de Inicio y Fin de Reproducción")
        col1, col2 = st.columns(2)
        
        with col1:
            reason_start = df_filtered['reason_start'].value_counts().head(10).reset_index()
            reason_start.columns = ['Razón', 'Conteo']
            fig_start = px.bar(
                reason_start,
                x='Conteo',
                y='Razón',
                orientation='h',
                title='Top 10 Razones de Inicio'
            )
            fig_start.update_traces(marker_color='#1DB954')
            fig_start.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_start, width='stretch')
        
        with col2:
            reason_end = df_filtered['reason_end'].value_counts().head(10).reset_index()
            reason_end.columns = ['Razón', 'Conteo']
            fig_end = px.bar(
                reason_end,
                x='Conteo',
                y='Razón',
                orientation='h',
                title='Top 10 Razones de Fin'
            )
            fig_end.update_traces(marker_color='#1ED760')
            fig_end.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_end, width='stretch')
        
        # Análisis de tracks más populares
        st.subheader("Análisis de Popularidad de Tracks")
        with st.spinner("Calculando features de tracks..."):
            tracks_df = track_features(df_filtered)
        
        # Top tracks por popularidad
        top_popularity = tracks_df.nlargest(20, 'total_plays')
        st.write(f"Mostrando top 20 de {len(tracks_df):,} tracks únicos")
        st.dataframe(
            top_popularity[['spotify_track_uri', 'total_plays', 'unique_listeners', 'avg_ms_played', 'skip_rate']].head(15),
            width='stretch'
        )
        
        # Reproducción por día de la semana
        st.subheader("Actividad por Día de la Semana")
        plays_by_day = df_filtered.groupby('day_name').size().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).reset_index()
        plays_by_day.columns = ['Día', 'Reproducciones']
        plays_by_day['Día'] = plays_by_day['Día'].map({
            'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
            'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        })
        
        fig_weekday = px.bar(
            plays_by_day,
            x='Día',
            y='Reproducciones',
            title='Reproducciones por Día de la Semana'
        )
        fig_weekday.update_traces(marker_color='#1DB954')
        st.plotly_chart(fig_weekday, width='stretch')
        
        # Tipo de contenido
        st.subheader("Tipo de Contenido")
        col1, col2 = st.columns(2)
        
        with col1:
            content_dist = df_filtered['content_type'].value_counts().reset_index()
            content_dist.columns = ['Tipo', 'Cantidad']
            fig_content = px.pie(
                content_dist,
                values='Cantidad',
                names='Tipo',
                title='Distribución de Tipo de Contenido'
            )
            st.plotly_chart(fig_content, width='stretch')
        
        with col2:
            # Evolución mensual
            monthly_plays = df_filtered.groupby(df_filtered['ts'].dt.to_period('M')).size().reset_index()
            monthly_plays.columns = ['Mes', 'Reproducciones']
            monthly_plays['Mes'] = monthly_plays['Mes'].astype(str)
            
            fig_monthly = px.line(
                monthly_plays,
                x='Mes',
                y='Reproducciones',
                title='Evolución Mensual de Reproducciones'
            )
            fig_monthly.update_traces(line_color='#1DB954')
            fig_monthly.update_xaxes(tickangle=45)
            st.plotly_chart(fig_monthly, width='stretch')
    
    with tab6:
        st.subheader("📥 Exportar Datos y Estadísticas")
        
        st.write("### 📊 Resumen Completo")
        
        # Crear resumen de estadísticas
        summary_data = {
            'Métrica': [
                'Total Reproducciones',
                'Periodo',
                'Total Días',
                'Total Horas Reproducidas',
                'Tracks Únicos',
                'Artistas Únicos',
                'Álbumes Únicos',
                'Países',
                'Plataformas Usadas',
                'Duración Promedio (min)',
                'Skip Rate (%)'
            ],
            'Valor': [
                f"{len(df_filtered):,}",
                f"{df_filtered['ts'].min().date()} a {df_filtered['ts'].max().date()}",
                f"{(df_filtered['ts'].max() - df_filtered['ts'].min()).days + 1:,}",
                f"{df_filtered['ms_played'].sum() / 3600000:,.1f}",
                f"{df_filtered['spotify_track_uri'].nunique():,}",
                f"{df_filtered['master_metadata_album_artist_name'].nunique():,}",
                f"{df_filtered['master_metadata_album_album_name'].nunique():,}",
                f"{df_filtered['conn_country'].nunique():,}",
                f"{df_filtered['platform'].nunique():,}",
                f"{df_filtered['minutes_played'].mean():.2f}",
                f"{df_filtered['skipped'].mean() * 100:.1f}" if 'skipped' in df_filtered.columns else "N/A"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='stretch', hide_index=True)
        
        st.write("---")
        
        # Opciones de descarga
        st.write("### 💾 Descargar Reportes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 50 tracks
            top_50_tracks = top_tracks(df_filtered, n=50, by='plays')
            csv_tracks = top_50_tracks.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Top 50 Tracks (CSV)",
                data=csv_tracks,
                file_name="top_50_tracks.csv",
                mime="text/csv"
            )
            
            # Top 50 artistas
            top_50_artists = top_artists(df_filtered, n=50, by='plays')
            csv_artists = top_50_artists.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Top 50 Artistas (CSV)",
                data=csv_artists,
                file_name="top_50_artists.csv",
                mime="text/csv"
            )
        
        with col2:
            # Resumen de estadísticas
            st.download_button(
                label="📥 Descargar Resumen (CSV)",
                data=summary_df.to_csv(index=False),
                file_name="resumen_spotify.csv",
                mime="text/csv"
            )
            
            # Datos de sesiones (si están calculados)
            if st.button("🎯 Generar Reporte de Sesiones"):
                with st.spinner("Generando reporte de sesiones..."):
                    df_sess = sessionize(df_filtered.copy())
                    sess_features = session_features(df_sess)
                    csv_sessions = sess_features.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Datos de Sesiones (CSV)",
                        data=csv_sessions,
                        file_name="sesiones_spotify.csv",
                        mime="text/csv"
                    )
        
        st.write("---")
        
        # Insights personalizados
        st.write("### 🎯 Insights Personalizados")
        
        # Track más reproducido
        most_played = df_filtered.groupby('master_metadata_track_name')['ms_played'].sum().idxmax()
        most_played_times = df_filtered[df_filtered['master_metadata_track_name'] == most_played].shape[0]
        
        # Artista favorito
        fav_artist = df_filtered.groupby('master_metadata_album_artist_name').size().idxmax()
        fav_artist_plays = df_filtered[df_filtered['master_metadata_album_artist_name'] == fav_artist].shape[0]
        
        # Hora favorita
        fav_hour = df_filtered['hour'].mode()[0]
        
        # Día favorito
        fav_day_name = df_filtered['day_name'].mode()[0]
        day_translation = {
            'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
            'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        }
        fav_day = day_translation.get(fav_day_name, fav_day_name)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **🎵 Tu Track Más Escuchado:**
            
            "{most_played}"
            
            Reproducido **{most_played_times:,} veces**
            """)
            
            st.success(f"""
            **⏰ Tu Hora Favorita para Escuchar Música:**
            
            **{fav_hour}:00 horas**
            """)
        
        with col2:
            st.info(f"""
            **🎤 Tu Artista Favorito:**
            
            "{fav_artist}"
            
            **{fav_artist_plays:,} reproducciones**
            """)
            
            st.success(f"""
            **📅 Tu Día Favorito:**
            
            **{fav_day}**
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Spotify Analytics Dashboard** | Desarrollado con Streamlit 🎵")

else:
    st.warning("No hay datos disponibles. Por favor, carga los datos curados primero.")
