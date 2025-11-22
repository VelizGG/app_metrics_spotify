"""Funciones de análisis exploratorio de datos (EDA) y visualización."""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna estadísticas resumidas del dataset.
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        DataFrame con estadísticas clave
    """
    stats = {
        'Total reproducciones': [len(df)],
        'Usuarios únicos': [df['username'].nunique()],
        'Periodo (inicio)': [df['ts'].min()],
        'Periodo (fin)': [df['ts'].max()],
        'Días cubiertos': [(df['ts'].max() - df['ts'].min()).days + 1],
    }
    
    if 'spotify_track_uri' in df.columns:
        stats['Tracks únicos'] = [df['spotify_track_uri'].nunique()]
    
    if 'master_metadata_album_artist_name' in df.columns:
        stats['Artistas únicos'] = [df['master_metadata_album_artist_name'].nunique()]
    
    if 'ms_played' in df.columns:
        total_hours = df['ms_played'].sum() / 3600000
        stats['Total horas reproducidas'] = [f"{total_hours:,.1f}"]
    
    return pd.DataFrame(stats).T.rename(columns={0: 'Valor'})


def top_tracks(df: pd.DataFrame, n: int = 20, by: str = 'plays') -> pd.DataFrame:
    """
    Retorna los top N tracks.
    
    Args:
        df: DataFrame con los datos
        n: Número de tracks a retornar
        by: Criterio de ranking ('plays' o 'time')
        
    Returns:
        DataFrame con top tracks
    """
    if by == 'plays':
        top = (df.groupby('master_metadata_track_name')
               .size()
               .sort_values(ascending=False)
               .head(n)
               .reset_index(name='plays'))
    else:  # by == 'time'
        top = (df.groupby('master_metadata_track_name')['ms_played']
               .sum()
               .sort_values(ascending=False)
               .head(n)
               .reset_index()
               .assign(hours=lambda x: x['ms_played'] / 3600000)
               .drop(columns=['ms_played']))
    
    return top


def top_artists(df: pd.DataFrame, n: int = 20, by: str = 'plays') -> pd.DataFrame:
    """
    Retorna los top N artistas.
    
    Args:
        df: DataFrame con los datos
        n: Número de artistas a retornar
        by: Criterio de ranking ('plays' o 'time')
        
    Returns:
        DataFrame con top artistas
    """
    if by == 'plays':
        top = (df.groupby('master_metadata_album_artist_name')
               .size()
               .sort_values(ascending=False)
               .head(n)
               .reset_index(name='plays'))
    else:  # by == 'time'
        top = (df.groupby('master_metadata_album_artist_name')['ms_played']
               .sum()
               .sort_values(ascending=False)
               .head(n)
               .reset_index()
               .assign(hours=lambda x: x['ms_played'] / 3600000)
               .drop(columns=['ms_played']))
    
    return top


def plays_over_time(df: pd.DataFrame, freq: str = 'D') -> pd.Series:
    """
    Retorna serie temporal de reproducciones.
    
    Args:
        df: DataFrame con los datos
        freq: Frecuencia de resample ('H', 'D', 'W', 'M')
        
    Returns:
        Serie con conteo por periodo
    """
    return df.set_index('ts').resample(freq).size()


def plot_plays_over_time(df: pd.DataFrame, freq: str = 'D', 
                         title: str = 'Reproducciones en el tiempo') -> go.Figure:
    """
    Genera gráfico de línea de reproducciones en el tiempo.
    
    Args:
        df: DataFrame con los datos
        freq: Frecuencia de resample
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    series = plays_over_time(df, freq)
    
    fig = px.line(x=series.index, y=series.values, 
                  labels={'x': 'Fecha', 'y': 'Reproducciones'},
                  title=title)
    fig.update_traces(line_color='#1DB954')  # Spotify green
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    return fig


def plot_hourly_heatmap(df: pd.DataFrame, title: str = 'Heatmap: Hora vs Día de la semana') -> go.Figure:
    """
    Genera heatmap de reproducciones por hora del día y día de la semana.
    
    Args:
        df: DataFrame con los datos
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    # Crear pivot table
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='plays')
    heatmap_pivot = heatmap_data.pivot(index='hour', columns='day_of_week', values='plays')
    
    # Reordenar columnas (Lun-Dom)
    day_names = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    heatmap_pivot.columns = [day_names[i] for i in heatmap_pivot.columns]
    
    fig = px.imshow(heatmap_pivot, 
                    labels=dict(x="Día", y="Hora", color="Plays"),
                    x=day_names,
                    y=list(range(24)),
                    color_continuous_scale='Greens',
                    title=title)
    
    fig.update_layout(
        xaxis_title="Día de la semana",
        yaxis_title="Hora del día"
    )
    
    return fig


def plot_top_tracks(df: pd.DataFrame, n: int = 20, by: str = 'plays') -> go.Figure:
    """
    Genera gráfico de barras de top tracks.
    
    Args:
        df: DataFrame con los datos
        n: Número de tracks
        by: Criterio ('plays' o 'time')
        
    Returns:
        Figura de Plotly
    """
    top = top_tracks(df, n, by)
    
    y_col = 'plays' if by == 'plays' else 'hours'
    title = f'Top {n} Tracks por {"Reproducciones" if by == "plays" else "Tiempo"}'
    
    fig = px.bar(top, x=y_col, y='master_metadata_track_name',
                 orientation='h',
                 title=title,
                 labels={y_col: 'Reproducciones' if by == 'plays' else 'Horas',
                        'master_metadata_track_name': 'Track'})
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.update_traces(marker_color='#1DB954')
    
    return fig


def plot_top_artists(df: pd.DataFrame, n: int = 20, by: str = 'plays') -> go.Figure:
    """
    Genera gráfico de barras de top artistas.
    
    Args:
        df: DataFrame con los datos
        n: Número de artistas
        by: Criterio ('plays' o 'time')
        
    Returns:
        Figura de Plotly
    """
    top = top_artists(df, n, by)
    
    y_col = 'plays' if by == 'plays' else 'hours'
    title = f'Top {n} Artistas por {"Reproducciones" if by == "plays" else "Tiempo"}'
    
    fig = px.bar(top, x=y_col, y='master_metadata_album_artist_name',
                 orientation='h',
                 title=title,
                 labels={y_col: 'Reproducciones' if by == 'plays' else 'Horas',
                        'master_metadata_album_artist_name': 'Artista'})
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.update_traces(marker_color='#1DB954')
    
    return fig


def plot_listening_duration_dist(df: pd.DataFrame, 
                                  title: str = 'Distribución de duración de reproducción') -> go.Figure:
    """
    Genera histograma de duración de reproducción.
    
    Args:
        df: DataFrame con los datos
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    fig = px.histogram(df, x='minutes_played',
                      nbins=50,
                      title=title,
                      labels={'minutes_played': 'Minutos reproducidos'})
    
    fig.update_traces(marker_color='#1DB954')
    fig.update_layout(
        xaxis_title="Minutos",
        yaxis_title="Frecuencia",
        showlegend=False
    )
    
    return fig


def plot_platform_distribution(df: pd.DataFrame, 
                               title: str = 'Distribución por Plataforma') -> go.Figure:
    """
    Genera gráfico de pie de distribución por plataforma.
    
    Args:
        df: DataFrame con los datos
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    platform_counts = df['platform'].value_counts()
    
    fig = px.pie(values=platform_counts.values, 
                 names=platform_counts.index,
                 title=title)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def plot_skip_rate_analysis(df: pd.DataFrame, 
                            title: str = 'Análisis de Skip Rate') -> go.Figure:
    """
    Genera gráfico de skip rate por diferentes dimensiones.
    
    Args:
        df: DataFrame con los datos
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    if 'skipped' not in df.columns:
        raise ValueError("DataFrame debe tener columna 'skipped'")
    
    # Skip rate por hora
    skip_by_hour = df.groupby('hour')['skipped'].mean().reset_index()
    
    fig = px.line(skip_by_hour, x='hour', y='skipped',
                  title=title,
                  labels={'hour': 'Hora del día', 'skipped': 'Skip Rate'})
    
    fig.update_traces(line_color='#1DB954')
    fig.update_layout(
        yaxis_tickformat='.1%',
        hovermode='x unified'
    )
    
    return fig


def plot_country_distribution(df: pd.DataFrame, n: int = 15,
                              title: str = 'Top Países por Reproducciones') -> go.Figure:
    """
    Genera gráfico de barras de top países.
    
    Args:
        df: DataFrame con los datos
        n: Número de países
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    if 'conn_country' not in df.columns:
        raise ValueError("DataFrame debe tener columna 'conn_country'")
    
    top_countries = df['conn_country'].value_counts().head(n).reset_index()
    top_countries.columns = ['country', 'plays']
    
    fig = px.bar(top_countries, x='country', y='plays',
                 title=title,
                 labels={'country': 'País', 'plays': 'Reproducciones'})
    
    fig.update_traces(marker_color='#1DB954')
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame, 
                            title: str = 'Matriz de Correlación') -> go.Figure:
    """
    Genera heatmap de correlación entre features numéricas.
    
    Args:
        df: DataFrame con los datos
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    fig = px.imshow(corr,
                    text_auto='.2f',
                    title=title,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)
    
    return fig


def reason_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analiza las razones de inicio y fin de reproducción.
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        Tupla con DataFrames de reason_start y reason_end
    """
    reason_start = df['reason_start'].value_counts().reset_index()
    reason_start.columns = ['reason', 'count']
    reason_start['percentage'] = reason_start['count'] / reason_start['count'].sum() * 100
    
    reason_end = df['reason_end'].value_counts().reset_index()
    reason_end.columns = ['reason', 'count']
    reason_end['percentage'] = reason_end['count'] / reason_end['count'].sum() * 100
    
    return reason_start, reason_end


if __name__ == '__main__':
    print("Módulo de análisis exploratorio de datos (EDA)")
    print("\nFunciones disponibles:")
    print("  - summary_stats(): Estadísticas generales")
    print("  - top_tracks(): Top tracks")
    print("  - top_artists(): Top artistas")
    print("  - plot_plays_over_time(): Gráfico temporal")
    print("  - plot_hourly_heatmap(): Heatmap hora/día")
    print("  - plot_top_tracks(): Gráfico de top tracks")
    print("  - plot_top_artists(): Gráfico de top artistas")
    print("  - Y más...")
