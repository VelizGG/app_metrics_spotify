"""Tests para el módulo data_pipeline."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_pipeline import normalize_columns, clean_data, add_derived_columns


@pytest.fixture
def sample_data():
    """Fixture con datos de ejemplo."""
    return pd.DataFrame({
        'ts': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'],
        'username': ['user1', 'user1', 'user2'],
        'platform': ['Windows', 'Android', 'iOS'],
        'ms_played': [180000, 240000, 120000],
        'master_metadata_track_name': ['Song A', 'Song B', 'Song C'],
        'master_metadata_album_artist_name': ['Artist 1', 'Artist 2', 'Artist 1'],
        'spotify_track_uri': ['spotify:track:1', 'spotify:track:2', 'spotify:track:3'],
        'spotify_episode_uri': [None, None, None],
        'shuffle': [True, False, True],
        'skipped': [False, True, False],
        'offline': [False, False, False],
        'incognito_mode': [False, False, False],
        'reason_start': ['clickrow', 'fwdbtn', 'clickrow'],
        'reason_end': ['endplay', 'fwdbtn', 'endplay'],
        'conn_country': ['US', 'US', 'MX']
    })


def test_normalize_columns(sample_data):
    """Test de normalización de columnas."""
    df = normalize_columns(sample_data)
    
    # Verificar que timestamp es datetime
    assert pd.api.types.is_datetime64_any_dtype(df['ts'])
    
    # Verificar que ms_played es int
    assert df['ms_played'].dtype == int
    
    # Verificar que las columnas booleanas están correctas
    assert df['shuffle'].dtype == 'boolean'
    assert df['skipped'].dtype == 'boolean'


def test_clean_data(sample_data):
    """Test de limpieza de datos."""
    # Agregar datos inválidos
    df = sample_data.copy()
    df.loc[0, 'ms_played'] = -1000  # ms_played negativo
    df.loc[1, 'ts'] = None  # timestamp nulo
    df = df.append(df.iloc[0])  # duplicado
    
    df = normalize_columns(df)
    df_clean = clean_data(df)
    
    # Verificar que se eliminaron filas inválidas
    assert len(df_clean) < len(df)
    assert (df_clean['ms_played'] >= 0).all()
    assert df_clean['ts'].notna().all()


def test_add_derived_columns(sample_data):
    """Test de creación de columnas derivadas."""
    df = normalize_columns(sample_data)
    df = add_derived_columns(df)
    
    # Verificar que se crearon las columnas derivadas
    assert 'date' in df.columns
    assert 'hour' in df.columns
    assert 'day_of_week' in df.columns
    assert 'minutes_played' in df.columns
    assert 'hours_played' in df.columns
    assert 'content_type' in df.columns
    
    # Verificar valores
    assert (df['hour'] >= 0).all() and (df['hour'] <= 23).all()
    assert (df['day_of_week'] >= 0).all() and (df['day_of_week'] <= 6).all()
    assert (df['minutes_played'] == df['ms_played'] / 60000).all()


def test_empty_dataframe():
    """Test con DataFrame vacío."""
    df = pd.DataFrame()
    df_norm = normalize_columns(df)
    assert len(df_norm) == 0


def test_missing_columns(sample_data):
    """Test con columnas faltantes."""
    df = sample_data[['ts', 'username', 'ms_played']].copy()
    df = normalize_columns(df)
    
    # No debe fallar con columnas faltantes
    assert 'ts' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['ts'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
