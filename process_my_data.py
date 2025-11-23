"""Script para procesar tus datos de Spotify."""
from pathlib import Path
import pandas as pd
import sys
sys.path.append('src')

from data_pipeline import pipeline_transform

# Procesar todos los archivos JSON en data/raw
raw_path = Path('data/raw')
curated_path = Path('data/curated')

print("="*70)
print("PROCESAMIENTO DE DATOS DE SPOTIFY")
print("="*70)

# Procesar solo archivos de Audio (no Video por ahora)
json_files = sorted(raw_path.glob('Streaming_History_Audio_*.json'))
print(f"\nEncontrados {len(json_files)} archivos de audio para procesar\n")

parquet_files = []

for i, json_file in enumerate(json_files, 1):
    output_file = curated_path / f"{json_file.stem}.parquet"
    print(f"[{i}/{len(json_files)}] Procesando {json_file.name}...")
    
    try:
        df = pipeline_transform(json_file, output_file)
        parquet_files.append(output_file)
    except Exception as e:
        print(f"  ⚠️ Error procesando {json_file.name}: {e}")
        continue

# Combinar todos los archivos
if len(parquet_files) > 1:
    print("\n" + "="*70)
    print("COMBINANDO TODOS LOS ARCHIVOS")
    print("="*70 + "\n")
    
    dfs = []
    for f in parquet_files:
        df_temp = pd.read_parquet(f)
        dfs.append(df_temp)
        print(f"  ✓ {f.name}: {len(df_temp):,} filas")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total antes de limpiar: {len(df_combined):,} filas")
    
    # Eliminar duplicados
    initial_len = len(df_combined)
    df_combined = df_combined.drop_duplicates()
    print(f"  Duplicados eliminados: {initial_len - len(df_combined):,}")
    
    # Ordenar por fecha
    df_combined = df_combined.sort_values('ts')
    
    # Guardar archivo final
    final_file = curated_path / 'spotify_data.parquet'
    df_combined.to_parquet(final_file, index=False)
    
    file_size = final_file.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*70)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("="*70)
    print(f"\nArchivo final: {final_file}")
    print(f"  • Total filas: {len(df_combined):,}")
    print(f"  • Tamaño: {file_size:.2f} MB")
    print(f"  • Periodo: {df_combined['ts'].min().date()} → {df_combined['ts'].max().date()}")
    print(f"  • Días: {(df_combined['ts'].max() - df_combined['ts'].min()).days + 1}")
    print(f"  • Total horas reproducidas: {df_combined['ms_played'].sum() / 3600000:,.1f}")
    print(f"  • Tracks únicos: {df_combined['spotify_track_uri'].nunique():,}")
    print(f"  • Artistas únicos: {df_combined['master_metadata_album_artist_name'].nunique():,}")
    
    print("\n🎉 ¡Listo! Ahora puedes:")
    print("  1. Abrir Jupyter: jupyter lab")
    print("  2. Ejecutar dashboard: streamlit run dashboards/streamlit_app.py")
    print("="*70)

else:
    print("\n⚠️ No se procesaron suficientes archivos para combinar")
