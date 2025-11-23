"""Ingeniería de features: sessionization y agregados de usuario."""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

SESSION_GAP = pd.Timedelta('30min')


def sessionize(df: pd.DataFrame, user_col: str = None, ts_col: str = 'ts', 
               gap: pd.Timedelta = SESSION_GAP) -> pd.DataFrame:
    """
    Agrupa reproducciones en sesiones basadas en gaps temporales.
    
    Una nueva sesión comienza cuando pasan más de `gap` minutos entre
    reproducciones consecutivas.
    
    Args:
        df: DataFrame con los datos
        user_col: Nombre de la columna de usuario (opcional, si None se usa 'global')
        ts_col: Nombre de la columna de timestamp
        gap: Timedelta que define el gap entre sesiones (default: 30 min)
        
    Returns:
        DataFrame con columna 'session_id' agregada
    """
    # Si no hay columna de usuario, crear una columna 'user' con valor fijo
    if user_col is None or user_col not in df.columns:
        df = df.copy()
        df['_user_temp'] = 'global'
        user_col = '_user_temp'
        created_temp = True
    else:
        df = df.copy()
        created_temp = False
    
    df = df.sort_values([user_col, ts_col])
    
    # Calcular diferencia temporal entre filas consecutivas por usuario
    df['prev_ts'] = df.groupby(user_col)[ts_col].shift(1)
    df['delta'] = df[ts_col] - df['prev_ts']
    
    # Marcar inicio de nueva sesión
    df['new_session'] = (df['delta'] > gap) | df['prev_ts'].isna()
    
    # Crear ID de sesión acumulativo por usuario
    df['session_id'] = df.groupby(user_col)['new_session'].cumsum()
    
    # Limpiar columnas temporales
    df.drop(columns=['prev_ts', 'delta', 'new_session'], inplace=True)
    
    # Crear session_id global único
    if created_temp:
        df['session_id'] = 'session_' + df['session_id'].astype(str)
        df.drop(columns=['_user_temp'], inplace=True)
        print(f"✓ Creadas {df['session_id'].nunique():,} sesiones")
    else:
        df['session_id'] = df[user_col].astype(str) + '::' + df['session_id'].astype(str)
        print(f"✓ Creadas {df['session_id'].nunique():,} sesiones para {df[user_col].nunique():,} usuarios")
    
    return df


def session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features agregadas por sesión.
    
    Args:
        df: DataFrame con columna 'session_id'
        
    Returns:
        DataFrame con features por sesión
    """
    if 'session_id' not in df.columns:
        raise ValueError("DataFrame debe tener columna 'session_id'. Ejecuta sessionize() primero.")
    
    agg_dict = {
        'ts': ['min', 'max', 'count'],
        'ms_played': 'sum',
    }
    
    # Agregar columnas opcionales
    if 'spotify_track_uri' in df.columns:
        agg_dict['spotify_track_uri'] = 'nunique'
    if 'master_metadata_album_artist_name' in df.columns:
        agg_dict['master_metadata_album_artist_name'] = 'nunique'
    if 'skipped' in df.columns:
        agg_dict['skipped'] = 'mean'
    
    sessions = df.groupby('session_id').agg(agg_dict).reset_index()
    
    # Renombrar columnas
    sessions.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                        for col in sessions.columns.values]
    sessions.rename(columns={
        'ts_min': 'session_start',
        'ts_max': 'session_end',
        'ts_count': 'tracks_played',
        'ms_played_sum': 'total_ms_played',
        'spotify_track_uri_nunique': 'unique_tracks',
        'master_metadata_album_artist_name_nunique': 'unique_artists',
        'skipped_mean': 'skip_rate'
    }, inplace=True)
    
    # Calcular duración de sesión
    sessions['session_duration_minutes'] = (
        (sessions['session_end'] - sessions['session_start']).dt.total_seconds() / 60
    )
    
    # Calcular minutos reproducidos
    sessions['minutes_played'] = sessions['total_ms_played'] / 60000
    
    # Diversidad de artistas (ratio unique/total)
    if 'unique_artists' in sessions.columns:
        sessions['artist_diversity'] = sessions['unique_artists'] / sessions['tracks_played']
    
    print(f"✓ Calculadas features para {len(sessions):,} sesiones")
    
    return sessions


def user_aggregates(df: pd.DataFrame, user_col: str = 'username') -> pd.DataFrame:
    """
    Calcula métricas agregadas por usuario.
    
    Args:
        df: DataFrame con los datos
        user_col: Nombre de la columna de usuario
        
    Returns:
        DataFrame con agregados por usuario
    """
    agg_dict = {
        'ts': ['count', 'min', 'max'],
        'ms_played': ['sum', 'mean'],
    }
    
    # Agregar columnas opcionales
    if 'spotify_track_uri' in df.columns:
        agg_dict['spotify_track_uri'] = 'nunique'
    if 'master_metadata_album_artist_name' in df.columns:
        agg_dict['master_metadata_album_artist_name'] = 'nunique'
    if 'master_metadata_album_album_name' in df.columns:
        agg_dict['master_metadata_album_album_name'] = 'nunique'
    if 'skipped' in df.columns:
        agg_dict['skipped'] = 'mean'
    if 'platform' in df.columns:
        agg_dict['platform'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else None
    
    users = df.groupby(user_col).agg(agg_dict).reset_index()
    
    # Renombrar columnas
    users.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                     for col in users.columns.values]
    users.rename(columns={
        'ts_count': 'total_plays',
        'ts_min': 'first_play',
        'ts_max': 'last_play',
        'ms_played_sum': 'total_ms_played',
        'ms_played_mean': 'avg_ms_played',
        'spotify_track_uri_nunique': 'unique_tracks',
        'master_metadata_album_artist_name_nunique': 'unique_artists',
        'master_metadata_album_album_name_nunique': 'unique_albums',
        'skipped_mean': 'skip_rate',
        'platform_<lambda_0>': 'primary_platform'
    }, inplace=True)
    
    # Calcular métricas derivadas
    users['avg_minutes_played'] = users['avg_ms_played'] / 60000
    users['total_hours_played'] = users['total_ms_played'] / 3600000
    users['days_active'] = (users['last_play'] - users['first_play']).dt.days + 1
    users['plays_per_day'] = users['total_plays'] / users['days_active']
    
    print(f"✓ Calculadas métricas para {len(users):,} usuarios")
    
    return users


def rolling_user_features(df: pd.DataFrame, user_col: str = 'username', 
                          ts_col: str = 'ts', windows: list = [7, 30]) -> pd.DataFrame:
    """
    Calcula features de ventana deslizante (rolling) por usuario.
    
    Args:
        df: DataFrame con los datos
        user_col: Nombre de la columna de usuario
        ts_col: Nombre de la columna de timestamp
        windows: Lista de ventanas en días (default: [7, 30])
        
    Returns:
        DataFrame con features rolling por usuario y fecha
    """
    df = df.sort_values([user_col, ts_col]).copy()
    
    # Agrupar por usuario y fecha
    daily = df.groupby([user_col, df[ts_col].dt.date]).agg({
        'ms_played': 'sum',
        'spotify_track_uri': 'nunique',
        ts_col: 'count'
    }).reset_index()
    daily.rename(columns={
        ts_col: 'date',
        'spotify_track_uri': 'unique_tracks',
        ts_col + '_count': 'plays'
    }, inplace=True)
    
    # Calcular rolling features para cada ventana
    for window in windows:
        daily[f'plays_last_{window}d'] = (
            daily.groupby(user_col)['plays']
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(drop=True)
        )
        
        daily[f'hours_last_{window}d'] = (
            daily.groupby(user_col)['ms_played']
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(drop=True) / 3600000
        )
        
        daily[f'unique_tracks_last_{window}d'] = (
            daily.groupby(user_col)['unique_tracks']
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(drop=True)
        )
    
    print(f"✓ Calculadas features rolling para {windows} días")
    
    return daily


def track_features(df: pd.DataFrame, track_col: str = 'spotify_track_uri', 
                   artist_col: str = 'master_metadata_album_artist_name') -> pd.DataFrame:
    """
    Calcula features agregadas por track.
    
    Args:
        df: DataFrame con los datos
        track_col: Nombre de la columna de track
        artist_col: Nombre de la columna de artista (para contar artistas únicos si no hay username)
        
    Returns:
        DataFrame con features por track
    """
    agg_dict = {
        'ms_played': ['sum', 'mean', 'count'],
    }
    
    # Usar username si existe, sino contar por IP o simplemente no incluir unique_listeners
    if 'username' in df.columns:
        agg_dict['username'] = 'nunique'
    elif 'ip_addr' in df.columns:
        agg_dict['ip_addr'] = 'nunique'
    
    if 'skipped' in df.columns:
        agg_dict['skipped'] = 'mean'
    
    tracks = df.groupby(track_col).agg(agg_dict).reset_index()
    
    # Renombrar columnas
    tracks.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                      for col in tracks.columns.values]
    
    rename_dict = {
        'ms_played_sum': 'total_ms_played',
        'ms_played_mean': 'avg_ms_played',
        'ms_played_count': 'total_plays',
        'skipped_mean': 'skip_rate'
    }
    
    if 'username_nunique' in tracks.columns:
        rename_dict['username_nunique'] = 'unique_listeners'
    elif 'ip_addr_nunique' in tracks.columns:
        rename_dict['ip_addr_nunique'] = 'unique_listeners'
    else:
        # Si no hay columna de listeners únicos, usar 1 por defecto
        tracks['unique_listeners'] = 1
    
    tracks.rename(columns=rename_dict, inplace=True)
    
    # Calcular popularidad relativa
    tracks['popularity_score'] = (
        tracks['total_plays'] * 0.7 + 
        tracks['unique_listeners'] * 0.3
    )
    
    # Normalizar popularidad (0-100)
    if len(tracks) > 0:
        max_pop = tracks['popularity_score'].max()
        if max_pop > 0:
            tracks['popularity_normalized'] = (tracks['popularity_score'] / max_pop * 100)
    
    print(f"✓ Calculadas features para {len(tracks):,} tracks")
    
    return tracks


def add_position_in_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega posición de cada track dentro de su sesión.
    
    Args:
        df: DataFrame con columna 'session_id'
        
    Returns:
        DataFrame con columna 'position_in_session'
    """
    if 'session_id' not in df.columns:
        raise ValueError("DataFrame debe tener columna 'session_id'. Ejecuta sessionize() primero.")
    
    df = df.copy()
    df['position_in_session'] = df.groupby('session_id').cumcount() + 1
    
    return df


if __name__ == '__main__':
    print("Módulo de ingeniería de features para análisis de Spotify")
    print("\nFunciones disponibles:")
    print("  - sessionize(): Agrupa reproducciones en sesiones")
    print("  - session_features(): Features agregadas por sesión")
    print("  - user_aggregates(): Métricas por usuario")
    print("  - rolling_user_features(): Features de ventana deslizante")
    print("  - track_features(): Métricas por track")
    print("  - add_position_in_session(): Posición de track en sesión")
