"""
Generador de datos sintéticos de Spotify para demostración pública.
Este script crea datos que mantienen las características estadísticas
de los datos reales pero con información completamente anonimizada.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_synthetic_spotify_data(n_rows: int = 10000, output_path: str = None) -> pd.DataFrame:
    """
    Genera datos sintéticos de Spotify para demostración.
    
    Args:
        n_rows: Número de filas a generar
        output_path: Path donde guardar el archivo (opcional)
    
    Returns:
        DataFrame con datos sintéticos
    """
    np.random.seed(42)
    
    # Artistas y tracks ficticios
    artists = [
        'The Synthetics', 'Data Dreams', 'Algorithm Avenue', 'Neural Network',
        'Pipeline Players', 'ETL Ensemble', 'Cloud Computing Crew', 'API Artists',
        'Database Dynamics', 'Machine Learning Music', 'DevOps DJs', 'Kubernetes Kollective',
        'Docker Disciples', 'Python Performers', 'SQL Singers', 'Analytics All-Stars'
    ]
    
    track_prefixes = ['Introduction to', 'Deep Dive into', 'Understanding', 'Mastering', 
                     'Exploring', 'Learning', 'Building', 'Deploying']
    track_topics = ['Data Pipelines', 'Feature Engineering', 'Model Training', 'ETL Processes',
                   'Data Quality', 'Pipeline Optimization', 'Cloud Architecture', 'API Design']
    
    tracks = [f"{prefix} {topic}" for prefix in track_prefixes for topic in track_topics]
    
    platforms = ['Windows 10', 'Android OS', 'iOS', 'Web Player', 'Linux']
    countries = ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'ES', 'NL']
    reason_starts = ['trackdone', 'fwdbtn', 'clickrow', 'playbtn', 'appload']
    reason_ends = ['trackdone', 'endplay', 'fwdbtn', 'backbtn', 'logout']
    
    # Generar timestamps (últimos 2 años)
    start_date = datetime.now() - timedelta(days=730)
    timestamps = [start_date + timedelta(
        days=np.random.exponential(scale=2),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60),
        seconds=np.random.randint(0, 60)
    ) for _ in range(n_rows)]
    timestamps.sort()
    
    # Crear dataset
    data = {
        'ts': timestamps,
        'platform': np.random.choice(platforms, n_rows, p=[0.4, 0.25, 0.2, 0.1, 0.05]),
        'ms_played': np.random.lognormal(mean=11.5, sigma=0.8, size=n_rows).astype(int),
        'conn_country': np.random.choice(countries, n_rows, p=[0.3, 0.15, 0.12, 0.1, 0.1, 0.08, 0.08, 0.07]),
        'ip_addr': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_rows)],
        'master_metadata_track_name': np.random.choice(tracks, n_rows),
        'master_metadata_album_artist_name': np.random.choice(artists, n_rows),
        'master_metadata_album_album_name': [f"Album {np.random.randint(1, 20)}" for _ in range(n_rows)],
        'spotify_track_uri': [f"spotify:track:synthetic{i:08d}" for i in np.random.randint(1, len(tracks)*10, n_rows)],
        'episode_name': [None] * n_rows,
        'episode_show_name': [None] * n_rows,
        'spotify_episode_uri': [None] * n_rows,
        'reason_start': np.random.choice(reason_starts, n_rows),
        'reason_end': np.random.choice(reason_ends, n_rows),
        'shuffle': np.random.choice([True, False], n_rows, p=[0.6, 0.4]),
        'skipped': np.random.choice([True, False, None], n_rows, p=[0.15, 0.7, 0.15]),
        'offline': np.random.choice([True, False], n_rows, p=[0.1, 0.9]),
        'offline_timestamp': [None] * n_rows,
        'incognito_mode': np.random.choice([True, False], n_rows, p=[0.05, 0.95]),
    }
    
    df = pd.DataFrame(data)
    
    # Agregar columnas derivadas (simulando el pipeline)
    df['date'] = pd.to_datetime(df['ts']).dt.date
    df['hour'] = pd.to_datetime(df['ts']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['ts']).dt.dayofweek
    df['day_name'] = pd.to_datetime(df['ts']).dt.day_name()
    df['minutes_played'] = df['ms_played'] / 60000
    df['hours_played'] = df['minutes_played'] / 60
    df['content_type'] = 'track'
    
    # Guardar si se especifica path
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False, compression='snappy')
        print(f"✓ Datos sintéticos guardados en {output_file}")
        print(f"  Filas: {len(df):,}")
        print(f"  Período: {df['ts'].min()} a {df['ts'].max()}")
    
    return df


if __name__ == "__main__":
    # Generar dataset sintético
    df = generate_synthetic_spotify_data(
        n_rows=50000,
        output_path='data/demo/synthetic_spotify_data.parquet'
    )
    
    print("\n📊 Resumen de datos sintéticos:")
    print(f"  Total reproducciones: {len(df):,}")
    print(f"  Artistas únicos: {df['master_metadata_album_artist_name'].nunique()}")
    print(f"  Tracks únicos: {df['spotify_track_uri'].nunique()}")
    print(f"  Horas totales: {df['hours_played'].sum():.1f}")
