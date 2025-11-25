"""Generación automática de playlists usando clustering y análisis de patrones."""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PlaylistGenerator:
    """
    Generador automático de playlists temáticas usando clustering.
    
    Crea playlists basadas en:
    - Patrones temporales (hora del día, día de semana)
    - Características de escucha (duración, skip rate)
    - Clustering de comportamiento similar
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el generador con datos históricos.
        
        Args:
            df: DataFrame con historial de reproducciones
        """
        self.df = df.copy()
        self.track_features = None
        self.playlists = {}
        self.scaler = StandardScaler()
        
        print("✓ PlaylistGenerator inicializado")
    
    def _build_track_features(self) -> pd.DataFrame:
        """Construye features agregados por track."""
        if self.track_features is not None:
            return self.track_features
        
        track_col = 'spotify_track_uri'
        
        agg_dict = {
            'ms_played': ['mean', 'std', 'count'],
            'hour': ['mean', 'std'],
            'day_of_week': 'mean',
        }
        
        if 'skipped' in self.df.columns:
            agg_dict['skipped'] = 'mean'
        
        if 'shuffle' in self.df.columns:
            agg_dict['shuffle'] = 'mean'
        
        # Agregar
        features = self.df.groupby(track_col).agg(agg_dict).reset_index()
        
        # Flatten columns
        features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in features.columns.values]
        
        # Renombrar
        rename_dict = {
            'ms_played_mean': 'avg_duration',
            'ms_played_std': 'duration_std',
            'ms_played_count': 'play_count',
            'hour_mean': 'typical_hour',
            'hour_std': 'hour_variability',
            'day_of_week_mean': 'typical_day',
            'skipped_mean': 'skip_rate',
            'shuffle_mean': 'shuffle_rate',
        }
        
        features.rename(columns=rename_dict, inplace=True)
        
        # Agregar metadata
        if 'master_metadata_album_artist_name' in self.df.columns:
            artist_map = self.df.groupby(track_col)['master_metadata_album_artist_name'].first()
            features['artist'] = features[track_col].map(artist_map)
        
        if 'master_metadata_track_name' in self.df.columns:
            track_map = self.df.groupby(track_col)['master_metadata_track_name'].first()
            features['track_name'] = features[track_col].map(track_map)
        
        self.track_features = features
        
        return features
    
    def generate_temporal_playlists(self) -> Dict[str, pd.DataFrame]:
        """
        Genera playlists basadas en patrones temporales.
        
        Returns:
            Diccionario con playlists temáticas
        """
        if self.track_features is None:
            self._build_track_features()
        
        playlists = {}
        
        # Morning playlist (6-11am)
        morning_tracks = self.track_features[
            (self.track_features['typical_hour'] >= 6) & 
            (self.track_features['typical_hour'] < 12)
        ].nlargest(30, 'play_count')
        
        if len(morning_tracks) > 0:
            playlists['Morning Energy'] = morning_tracks
            print(f"✓ 'Morning Energy' playlist: {len(morning_tracks)} tracks")
        
        # Afternoon playlist (12-5pm)
        afternoon_tracks = self.track_features[
            (self.track_features['typical_hour'] >= 12) & 
            (self.track_features['typical_hour'] < 17)
        ].nlargest(30, 'play_count')
        
        if len(afternoon_tracks) > 0:
            playlists['Afternoon Vibes'] = afternoon_tracks
            print(f"✓ 'Afternoon Vibes' playlist: {len(afternoon_tracks)} tracks")
        
        # Evening playlist (5-10pm)
        evening_tracks = self.track_features[
            (self.track_features['typical_hour'] >= 17) & 
            (self.track_features['typical_hour'] < 22)
        ].nlargest(30, 'play_count')
        
        if len(evening_tracks) > 0:
            playlists['Evening Chill'] = evening_tracks
            print(f"✓ 'Evening Chill' playlist: {len(evening_tracks)} tracks")
        
        # Late night playlist (10pm-2am)
        late_night_tracks = self.track_features[
            ((self.track_features['typical_hour'] >= 22) | 
             (self.track_features['typical_hour'] < 2))
        ].nlargest(30, 'play_count')
        
        if len(late_night_tracks) > 0:
            playlists['Late Night'] = late_night_tracks
            print(f"✓ 'Late Night' playlist: {len(late_night_tracks)} tracks")
        
        # Weekday vs Weekend
        weekday_tracks = self.track_features[
            self.track_features['typical_day'] < 5
        ].nlargest(30, 'play_count')
        
        if len(weekday_tracks) > 0:
            playlists['Weekday Focus'] = weekday_tracks
            print(f"✓ 'Weekday Focus' playlist: {len(weekday_tracks)} tracks")
        
        weekend_tracks = self.track_features[
            self.track_features['typical_day'] >= 5
        ].nlargest(30, 'play_count')
        
        if len(weekend_tracks) > 0:
            playlists['Weekend Mood'] = weekend_tracks
            print(f"✓ 'Weekend Mood' playlist: {len(weekend_tracks)} tracks")
        
        self.playlists.update(playlists)
        
        return playlists
    
    def generate_behavior_playlists(self) -> Dict[str, pd.DataFrame]:
        """
        Genera playlists basadas en comportamiento de escucha.
        
        Returns:
            Diccionario con playlists temáticas
        """
        if self.track_features is None:
            self._build_track_features()
        
        playlists = {}
        
        # Never Skip - tracks con muy baja skip rate
        if 'skip_rate' in self.track_features.columns:
            never_skip = self.track_features[
                (self.track_features['skip_rate'] < 0.10) &
                (self.track_features['play_count'] >= 3)
            ].nlargest(50, 'play_count')
            
            if len(never_skip) > 0:
                playlists['Never Skip Hits'] = never_skip
                print(f"✓ 'Never Skip Hits' playlist: {len(never_skip)} tracks")
        
        # Deep Focus - tracks largos y baja variabilidad de hora
        if 'hour_variability' in self.track_features.columns:
            focus_tracks = self.track_features[
                (self.track_features['avg_duration'] > 180000) &  # > 3 min
                (self.track_features['hour_variability'] < 2)  # Consistente
            ].nlargest(30, 'play_count')
            
            if len(focus_tracks) > 0:
                playlists['Deep Focus'] = focus_tracks
                print(f"✓ 'Deep Focus' playlist: {len(focus_tracks)} tracks")
        
        # Quick Hits - tracks cortos y populares
        quick_tracks = self.track_features[
            self.track_features['avg_duration'] < 180000  # < 3 min
        ].nlargest(30, 'play_count')
        
        if len(quick_tracks) > 0:
            playlists['Quick Hits'] = quick_tracks
            print(f"✓ 'Quick Hits' playlist: {len(quick_tracks)} tracks")
        
        # Top played
        top_tracks = self.track_features.nlargest(50, 'play_count')
        playlists['All-Time Favorites'] = top_tracks
        print(f"✓ 'All-Time Favorites' playlist: {len(top_tracks)} tracks")
        
        # Shuffle favorites - tracks con alta shuffle rate
        if 'shuffle_rate' in self.track_features.columns:
            shuffle_fav = self.track_features[
                self.track_features['shuffle_rate'] > 0.7
            ].nlargest(30, 'play_count')
            
            if len(shuffle_fav) > 0:
                playlists['Shuffle Favorites'] = shuffle_fav
                print(f"✓ 'Shuffle Favorites' playlist: {len(shuffle_fav)} tracks")
        
        self.playlists.update(playlists)
        
        return playlists
    
    def generate_artist_playlists(self, top_n_artists: int = 5, 
                                 tracks_per_artist: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Genera playlists por artista favorito.
        
        Args:
            top_n_artists: Número de artistas para generar playlists
            tracks_per_artist: Tracks por playlist
            
        Returns:
            Diccionario con playlists por artista
        """
        if self.track_features is None:
            self._build_track_features()
        
        if 'artist' not in self.track_features.columns:
            print("⚠️ No hay información de artista disponible")
            return {}
        
        playlists = {}
        
        # Identificar top artistas
        top_artists = (
            self.track_features.groupby('artist')['play_count']
            .sum()
            .nlargest(top_n_artists)
            .index
        )
        
        for artist in top_artists:
            artist_tracks = (
                self.track_features[self.track_features['artist'] == artist]
                .nlargest(tracks_per_artist, 'play_count')
            )
            
            if len(artist_tracks) > 0:
                playlist_name = f'Best of {artist}'
                playlists[playlist_name] = artist_tracks
                print(f"✓ '{playlist_name}' playlist: {len(artist_tracks)} tracks")
        
        self.playlists.update(playlists)
        
        return playlists
    
    def generate_cluster_playlists(self, n_clusters: int = 5, 
                                  min_tracks: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Genera playlists usando K-Means clustering.
        
        Args:
            n_clusters: Número de clusters a crear
            min_tracks: Mínimo de tracks por playlist
            
        Returns:
            Diccionario con playlists generadas por clustering
        """
        if self.track_features is None:
            self._build_track_features()
        
        # Seleccionar features para clustering
        feature_cols = [
            'avg_duration', 'typical_hour', 'typical_day', 'play_count'
        ]
        
        if 'skip_rate' in self.track_features.columns:
            feature_cols.append('skip_rate')
        
        if 'hour_variability' in self.track_features.columns:
            feature_cols.append('hour_variability')
        
        # Filtrar solo columnas existentes
        feature_cols = [c for c in feature_cols if c in self.track_features.columns]
        
        # Extraer features
        X = self.track_features[feature_cols].fillna(0)
        
        if len(X) < n_clusters * min_tracks:
            print(f"⚠️ No hay suficientes tracks para {n_clusters} clusters")
            return {}
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Agregar cluster labels
        self.track_features['cluster'] = clusters
        
        # Generar playlists por cluster
        playlists = {}
        cluster_names = [
            'Cluster Mix 1', 'Cluster Mix 2', 'Cluster Mix 3',
            'Cluster Mix 4', 'Cluster Mix 5', 'Cluster Mix 6',
            'Cluster Mix 7', 'Cluster Mix 8'
        ]
        
        for i in range(n_clusters):
            cluster_tracks = self.track_features[
                self.track_features['cluster'] == i
            ].nlargest(30, 'play_count')
            
            if len(cluster_tracks) >= min_tracks:
                playlist_name = cluster_names[i] if i < len(cluster_names) else f'Cluster Mix {i+1}'
                playlists[playlist_name] = cluster_tracks
                
                # Analizar características del cluster
                avg_hour = cluster_tracks['typical_hour'].mean()
                avg_skip = cluster_tracks.get('skip_rate', pd.Series([0])).mean()
                
                print(f"✓ '{playlist_name}': {len(cluster_tracks)} tracks "
                      f"(hora típica: {avg_hour:.1f}h, skip rate: {avg_skip:.1%})")
        
        self.playlists.update(playlists)
        
        return playlists
    
    def generate_mood_playlists(self) -> Dict[str, pd.DataFrame]:
        """
        Genera playlists basadas en 'mood' inferido de patrones.
        
        Returns:
            Diccionario con playlists por mood
        """
        if self.track_features is None:
            self._build_track_features()
        
        playlists = {}
        
        # High Energy - tracks cortos, típicamente en horas activas
        high_energy = self.track_features[
            (self.track_features['avg_duration'] < 240000) &  # < 4 min
            (self.track_features['typical_hour'].between(7, 20))
        ].nlargest(30, 'play_count')
        
        if len(high_energy) > 0:
            playlists['High Energy'] = high_energy
            print(f"✓ 'High Energy' playlist: {len(high_energy)} tracks")
        
        # Relaxation - tracks largos, baja skip rate
        if 'skip_rate' in self.track_features.columns:
            relaxation = self.track_features[
                (self.track_features['avg_duration'] > 240000) &
                (self.track_features['skip_rate'] < 0.15)
            ].nlargest(30, 'play_count')
            
            if len(relaxation) > 0:
                playlists['Relaxation'] = relaxation
                print(f"✓ 'Relaxation' playlist: {len(relaxation)} tracks")
        
        # Consistent favorites - baja variabilidad temporal
        if 'hour_variability' in self.track_features.columns:
            consistent = self.track_features[
                (self.track_features['hour_variability'] < 3) &
                (self.track_features['play_count'] >= 5)
            ].nlargest(30, 'play_count')
            
            if len(consistent) > 0:
                playlists['Anytime Classics'] = consistent
                print(f"✓ 'Anytime Classics' playlist: {len(consistent)} tracks")
        
        self.playlists.update(playlists)
        
        return playlists
    
    def generate_discovery_playlist(self, n: int = 30, 
                                   min_plays: int = 2,
                                   max_plays: int = 10) -> pd.DataFrame:
        """
        Genera playlist de 'descubrimiento' con tracks menos escuchados.
        
        Args:
            n: Número de tracks
            min_plays: Mínimo de reproducciones
            max_plays: Máximo de reproducciones
            
        Returns:
            DataFrame con playlist de descubrimiento
        """
        if self.track_features is None:
            self._build_track_features()
        
        # Tracks con escuchas moderadas
        discovery = self.track_features[
            (self.track_features['play_count'] >= min_plays) &
            (self.track_features['play_count'] <= max_plays)
        ]
        
        # Si hay skip rate, preferir tracks con baja skip rate
        if 'skip_rate' in discovery.columns and len(discovery) > 0:
            discovery = discovery.nsmallest(n, 'skip_rate')
        else:
            discovery = discovery.sample(min(n, len(discovery)), random_state=42)
        
        if len(discovery) > 0:
            self.playlists['Rediscover'] = discovery
            print(f"✓ 'Rediscover' playlist: {len(discovery)} tracks")
        
        return discovery
    
    def generate_all_playlists(self, include_clusters: bool = True,
                              include_artists: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Genera todas las playlists disponibles.
        
        Args:
            include_clusters: Incluir playlists de clustering
            include_artists: Incluir playlists por artista
            
        Returns:
            Diccionario con todas las playlists
        """
        print("\n" + "="*60)
        print("GENERANDO PLAYLISTS AUTOMÁTICAS")
        print("="*60 + "\n")
        
        # Temporal playlists
        print("📅 Playlists temporales:")
        self.generate_temporal_playlists()
        
        # Behavior playlists
        print("\n🎵 Playlists por comportamiento:")
        self.generate_behavior_playlists()
        
        # Mood playlists
        print("\n😊 Playlists por mood:")
        self.generate_mood_playlists()
        
        # Discovery
        print("\n🔍 Playlist de descubrimiento:")
        self.generate_discovery_playlist()
        
        # Artist playlists
        if include_artists:
            print("\n🎤 Playlists por artista:")
            self.generate_artist_playlists()
        
        # Cluster playlists
        if include_clusters:
            print("\n🧩 Playlists por clustering:")
            self.generate_cluster_playlists()
        
        print("\n" + "="*60)
        print(f"✓ Total: {len(self.playlists)} playlists generadas")
        print("="*60)
        
        return self.playlists
    
    def get_playlist(self, name: str) -> pd.DataFrame:
        """
        Obtiene una playlist específica.
        
        Args:
            name: Nombre de la playlist
            
        Returns:
            DataFrame con tracks de la playlist
        """
        if name not in self.playlists:
            raise ValueError(f"Playlist '{name}' no encontrada. "
                           f"Disponibles: {list(self.playlists.keys())}")
        
        return self.playlists[name]
    
    def list_playlists(self) -> List[Tuple[str, int]]:
        """
        Lista todas las playlists generadas.
        
        Returns:
            Lista de tuplas (nombre, número_de_tracks)
        """
        return [(name, len(df)) for name, df in self.playlists.items()]
    
    def export_playlist(self, name: str, output_path: str, 
                       include_scores: bool = False) -> None:
        """
        Exporta playlist a CSV o JSON.
        
        Args:
            name: Nombre de la playlist
            output_path: Path de salida
            include_scores: Incluir scores y features técnicos
        """
        from pathlib import Path
        
        playlist = self.get_playlist(name)
        
        # Seleccionar columnas para export
        export_cols = ['spotify_track_uri']
        
        if 'track_name' in playlist.columns:
            export_cols.append('track_name')
        if 'artist' in playlist.columns:
            export_cols.append('artist')
        
        export_cols.append('play_count')
        
        if include_scores:
            # Incluir todas las columnas
            export_df = playlist
        else:
            export_df = playlist[export_cols]
        
        # Exportar
        output_file = Path(output_path)
        
        if output_file.suffix == '.json':
            export_df.to_json(output_file, orient='records', indent=2)
        else:
            export_df.to_csv(output_file, index=False)
        
        print(f"✓ Playlist '{name}' exportada a {output_file}")


def create_playlist_generator(df: pd.DataFrame, 
                              generate_all: bool = True) -> PlaylistGenerator:
    """
    Función helper para crear generador listo para usar.
    
    Args:
        df: DataFrame con datos históricos
        generate_all: Si True, genera todas las playlists automáticamente
        
    Returns:
        PlaylistGenerator inicializado
    """
    generator = PlaylistGenerator(df)
    
    if generate_all:
        generator.generate_all_playlists()
    
    return generator


if __name__ == '__main__':
    print("Generador Automático de Playlists")
    print("\nClase disponible:")
    print("  - PlaylistGenerator: Generador principal")
    print("\nMétodos principales:")
    print("  - generate_all_playlists(): Genera todas las playlists")
    print("  - generate_temporal_playlists(): Por hora del día")
    print("  - generate_behavior_playlists(): Por comportamiento")
    print("  - generate_mood_playlists(): Por mood inferido")
    print("  - generate_cluster_playlists(): Por clustering")
    print("  - generate_artist_playlists(): Por artista")
