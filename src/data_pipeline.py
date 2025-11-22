"""Funciones para cargar y limpiar los datos brutos de Spotify."""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import json

DATETIME_COL = 'ts'

def load_ndjson(path: Union[str, Path]) -> pd.DataFrame:
    """
    Carga archivo NDJSON (JSON lines) y retorna DataFrame.
    
    Args:
        path: Ruta al archivo NDJSON
        
    Returns:
        DataFrame con los datos cargados
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    
    df = pd.read_json(path, lines=True)
    print(f"✓ Cargadas {len(df):,} filas desde {path.name}")
    return df


def load_json_array(path: Union[str, Path]) -> pd.DataFrame:
    """
    Carga archivo JSON con array de objetos.
    
    Args:
        path: Ruta al archivo JSON
        
    Returns:
        DataFrame con los datos cargados
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"✓ Cargadas {len(df):,} filas desde {path.name}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas y convierte tipos de datos.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame con columnas normalizadas
    """
    df = df.copy()
    
    # Limpiar nombres de columnas
    df.rename(columns=lambda c: c.strip(), inplace=True)
    
    # Convertir timestamp
    if DATETIME_COL in df.columns:
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors='coerce')
        print(f"✓ Columna '{DATETIME_COL}' convertida a datetime")
    
    # Convertir ms_played a int
    if 'ms_played' in df.columns:
        df['ms_played'] = pd.to_numeric(df['ms_played'], errors='coerce').fillna(0).astype(int)
        print(f"✓ Columna 'ms_played' convertida a int")
    
    # Convertir columnas booleanas
    bool_cols = ['shuffle', 'skipped', 'offline', 'incognito_mode']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype('boolean')
    
    print(f"✓ {len(bool_cols)} columnas booleanas normalizadas")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia datos eliminando registros inválidos.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame limpio
    """
    initial_rows = len(df)
    df = df.copy()
    
    # Eliminar filas sin timestamp
    if DATETIME_COL in df.columns:
        df = df.dropna(subset=[DATETIME_COL])
    
    # Eliminar ms_played negativos o nulos
    if 'ms_played' in df.columns:
        df = df[df['ms_played'] >= 0]
    
    # Eliminar duplicados completos
    df = df.drop_duplicates()
    
    removed = initial_rows - len(df)
    print(f"✓ Eliminadas {removed:,} filas inválidas ({removed/initial_rows*100:.1f}%)")
    
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas derivadas útiles para el análisis.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame con columnas adicionales
    """
    df = df.copy()
    
    if DATETIME_COL in df.columns:
        # Extraer componentes temporales
        df['date'] = df[DATETIME_COL].dt.date
        df['hour'] = df[DATETIME_COL].dt.hour
        df['day_of_week'] = df[DATETIME_COL].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_name'] = df[DATETIME_COL].dt.day_name()
        df['week'] = df[DATETIME_COL].dt.isocalendar().week
        df['month'] = df[DATETIME_COL].dt.month
        df['year'] = df[DATETIME_COL].dt.year
        print(f"✓ Agregadas columnas temporales: date, hour, day_of_week, etc.")
    
    if 'ms_played' in df.columns:
        # Convertir a minutos y horas
        df['minutes_played'] = df['ms_played'] / 60000
        df['hours_played'] = df['ms_played'] / 3600000
        print(f"✓ Agregadas columnas: minutes_played, hours_played")
    
    # Identificar si es track o podcast
    if 'spotify_track_uri' in df.columns and 'spotify_episode_uri' in df.columns:
        df['content_type'] = df.apply(
            lambda x: 'podcast' if pd.notna(x['spotify_episode_uri']) else 'track',
            axis=1
        )
        print(f"✓ Agregada columna: content_type")
    
    return df


def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Guarda DataFrame en formato Parquet.
    
    Args:
        df: DataFrame a guardar
        path: Ruta de destino
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(path, index=False, compression='snappy')
    file_size = path.stat().st_size / (1024 * 1024)  # MB
    print(f"✓ Guardado {len(df):,} filas en {path.name} ({file_size:.2f} MB)")


def save_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Guarda DataFrame en formato CSV.
    
    Args:
        df: DataFrame a guardar
        path: Ruta de destino
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=False, encoding='utf-8')
    file_size = path.stat().st_size / (1024 * 1024)  # MB
    print(f"✓ Guardado {len(df):,} filas en {path.name} ({file_size:.2f} MB)")


def load_curated_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Carga datos curados desde Parquet.
    
    Args:
        path: Ruta al archivo Parquet
        
    Returns:
        DataFrame con datos curados
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    
    df = pd.read_parquet(path)
    print(f"✓ Cargadas {len(df):,} filas desde {path.name}")
    return df


def pipeline_transform(input_path: Union[str, Path], output_path: Union[str, Path]) -> pd.DataFrame:
    """
    Pipeline completo de transformación: carga, normaliza, limpia y guarda.
    
    Args:
        input_path: Ruta al archivo de entrada (NDJSON/JSON)
        output_path: Ruta al archivo de salida (Parquet)
        
    Returns:
        DataFrame procesado
    """
    print(f"\n{'='*60}")
    print(f"PIPELINE DE TRANSFORMACIÓN")
    print(f"{'='*60}\n")
    
    # Cargar datos
    if str(input_path).endswith('.ndjson'):
        df = load_ndjson(input_path)
    else:
        df = load_json_array(input_path)
    
    # Transformar
    df = normalize_columns(df)
    df = clean_data(df)
    df = add_derived_columns(df)
    
    # Guardar
    save_parquet(df, output_path)
    
    print(f"\n{'='*60}")
    print(f"TRANSFORMACIÓN COMPLETADA")
    print(f"{'='*60}\n")
    
    return df


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python data_pipeline.py <input_path> <output_path>")
        print("Ejemplo: python data_pipeline.py data/raw/sample.ndjson data/curated/sample.parquet")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    df = pipeline_transform(input_path, output_path)
    
    # Mostrar resumen
    print("\nRESUMEN DEL DATASET:")
    print(f"  - Filas: {len(df):,}")
    print(f"  - Columnas: {len(df.columns)}")
    print(f"  - Periodo: {df['ts'].min()} a {df['ts'].max()}")
    print(f"  - Usuarios únicos: {df['username'].nunique():,}")
