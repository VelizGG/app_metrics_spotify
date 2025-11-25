"""Sistema de recomendación de canciones basado en patrones de escucha y skip behavior."""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TrackRecommender:
    """
    Sistema de recomendación basado en contenido y comportamiento de usuario.
    
    Combina:
    - Content-based filtering (features de tracks)
    - Skip behavior patterns
    - User listening preferences
    - Temporal patterns
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el recomendador con datos históricos.
        
        Args:
            df: DataFrame con historial de reproducciones
        """
        self.df = df.copy()
        self.track_features = None
        self.user_profile = None
        self.scaler = StandardScaler()
        self.track_embeddings = None
        self.similarity_matrix = None
        
        print("✓ TrackRecommender inicializado")
    
    def build_track_features(self) -> pd.DataFrame:
        """
        Construye matriz de features por track.
        
        Returns:
            DataFrame con features agregados por track
        """
        # Agregar por track
        agg_dict = {
            'ms_played': ['mean', 'std', 'count'],
            'hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),
            'day_of_week': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),
        }
        
        if 'skipped' in self.df.columns:
            agg_dict['skipped'] = 'mean'
        
        if 'shuffle' in self.df.columns:
            agg_dict['shuffle'] = 'mean'
        
        if 'platform' in self.df.columns:
            agg_dict['platform'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        
        # Agrupar por track
        track_col = 'spotify_track_uri'
        if track_col not in self.df.columns:
            raise ValueError(f"DataFrame debe tener columna '{track_col}'")
        
        features = self.df.groupby(track_col).agg(agg_dict).reset_index()
        
        # Flatten multi-level columns
        features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in features.columns.values]
        
        # Renombrar columnas
        rename_dict = {
            'ms_played_mean': 'avg_duration',
            'ms_played_std': 'duration_variability',
            'ms_played_count': 'total_plays',
            'hour_<lambda_0>': 'typical_hour',
            'hour_<lambda>': 'typical_hour',
            'day_of_week_<lambda_0>': 'typical_day',
            'day_of_week_<lambda>': 'typical_day',
            'skipped_mean': 'skip_rate',
            'shuffle_mean': 'shuffle_rate',
            'platform_<lambda_0>': 'primary_platform',
            'platform_<lambda>': 'primary_platform',
        }
        
        features.rename(columns=rename_dict, inplace=True)
        
        # Agregar metadata si está disponible
        if 'master_metadata_album_artist_name' in self.df.columns:
            artist_map = self.df.groupby(track_col)['master_metadata_album_artist_name'].first()
            features['artist'] = features[track_col].map(artist_map)
        
        if 'master_metadata_track_name' in self.df.columns:
            track_name_map = self.df.groupby(track_col)['master_metadata_track_name'].first()
            features['track_name'] = features[track_col].map(track_name_map)
        
        # Calcular popularidad
        features['popularity'] = (
            features['total_plays'] / features['total_plays'].max() * 100
        )
        
        # Calcular completion rate (inverso de skip rate)
        if 'skip_rate' in features.columns:
            features['completion_rate'] = 1 - features['skip_rate']
        else:
            features['completion_rate'] = 1.0
        
        self.track_features = features
        print(f"✓ Features construidos para {len(features):,} tracks")
        
        return features
    
    def build_track_embeddings(self, n_components: int = 50) -> np.ndarray:
        """
        Crea embeddings densos de tracks usando features numéricos.
        
        Args:
            n_components: Dimensiones para reducción (si hay muchos features)
            
        Returns:
            Array de embeddings [n_tracks x n_features]
        """
        if self.track_features is None:
            self.build_track_features()
        
        # Seleccionar features numéricos
        numeric_features = [
            'avg_duration', 'duration_variability', 'total_plays',
            'typical_hour', 'typical_day', 'popularity', 'completion_rate'
        ]
        
        if 'skip_rate' in self.track_features.columns:
            numeric_features.append('skip_rate')
        
        if 'shuffle_rate' in self.track_features.columns:
            numeric_features.append('shuffle_rate')
        
        # Filtrar solo columnas que existen
        numeric_features = [f for f in numeric_features if f in self.track_features.columns]
        
        # Extraer features
        X = self.track_features[numeric_features].fillna(0)
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X)
        
        # Reducción dimensional si es necesario
        if X_scaled.shape[1] > n_components and len(self.track_features) > n_components:
            svd = TruncatedSVD(n_components=min(n_components, X_scaled.shape[1] - 1))
            X_scaled = svd.fit_transform(X_scaled)
            print(f"✓ Embeddings reducidos a {X_scaled.shape[1]} dimensiones")
        
        self.track_embeddings = X_scaled
        print(f"✓ Track embeddings creados: {X_scaled.shape}")
        
        return X_scaled
    
    def build_similarity_matrix(self) -> np.ndarray:
        """
        Calcula matriz de similitud entre todos los tracks.
        
        Returns:
            Matriz de similitud coseno [n_tracks x n_tracks]
        """
        if self.track_embeddings is None:
            self.build_track_embeddings()
        
        # Calcular similitud coseno
        self.similarity_matrix = cosine_similarity(self.track_embeddings)
        
        print(f"✓ Matriz de similitud calculada: {self.similarity_matrix.shape}")
        
        return self.similarity_matrix
    
    def build_user_profile(self) -> Dict[str, any]:
        """
        Construye perfil de preferencias del usuario basado en historial.
        
        Returns:
            Diccionario con preferencias del usuario
        """
        profile = {}
        
        # Top artistas
        if 'master_metadata_album_artist_name' in self.df.columns:
            top_artists = (
                self.df['master_metadata_album_artist_name']
                .value_counts()
                .head(20)
                .index.tolist()
            )
            profile['favorite_artists'] = top_artists
        
        # Horas preferidas de escucha
        if 'hour' in self.df.columns:
            hour_counts = self.df['hour'].value_counts().sort_index()
            profile['listening_hours'] = hour_counts.to_dict()
            profile['peak_hours'] = hour_counts.nlargest(3).index.tolist()
        
        # Días preferidos
        if 'day_of_week' in self.df.columns:
            day_counts = self.df['day_of_week'].value_counts()
            profile['listening_days'] = day_counts.to_dict()
        
        # Skip tolerance (qué tan tolerante es con canciones largas)
        if 'skipped' in self.df.columns:
            profile['avg_skip_rate'] = self.df['skipped'].mean()
            profile['skip_tolerance'] = 'high' if profile['avg_skip_rate'] < 0.15 else 'medium' if profile['avg_skip_rate'] < 0.30 else 'low'
        
        # Duración promedio preferida
        if 'ms_played' in self.df.columns:
            profile['avg_listening_duration_ms'] = self.df['ms_played'].mean()
            profile['preferred_duration_range'] = (
                self.df['ms_played'].quantile(0.25),
                self.df['ms_played'].quantile(0.75)
            )
        
        # Plataforma preferida
        if 'platform' in self.df.columns:
            profile['primary_platform'] = self.df['platform'].mode()[0]
        
        # Preferencia por shuffle
        if 'shuffle' in self.df.columns:
            profile['shuffle_preference'] = self.df['shuffle'].mean()
        
        self.user_profile = profile
        print(f"✓ Perfil de usuario construido")
        
        return profile
    
    def get_similar_tracks(self, track_uri: str, n: int = 10, 
                          exclude_listened: bool = False) -> pd.DataFrame:
        """
        Encuentra tracks similares a uno dado.
        
        Args:
            track_uri: URI del track de referencia
            n: Número de recomendaciones
            exclude_listened: Si True, excluye tracks ya escuchados
            
        Returns:
            DataFrame con tracks similares y scores
        """
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        # Encontrar índice del track
        track_col = 'spotify_track_uri'
        if track_uri not in self.track_features[track_col].values:
            raise ValueError(f"Track {track_uri} no encontrado en el dataset")
        
        track_idx = self.track_features[self.track_features[track_col] == track_uri].index[0]
        
        # Obtener similitudes
        similarities = self.similarity_matrix[track_idx]
        
        # Crear DataFrame con resultados
        results = self.track_features.copy()
        results['similarity_score'] = similarities
        
        # Excluir el track mismo
        results = results[results[track_col] != track_uri]
        
        # Excluir ya escuchados si se requiere
        if exclude_listened:
            listened_tracks = self.df[track_col].unique()
            results = results[~results[track_col].isin(listened_tracks)]
        
        # Ordenar por similitud
        results = results.sort_values('similarity_score', ascending=False).head(n)
        
        return results
    
    def recommend_for_context(self, hour: int = None, day_of_week: int = None,
                              n: int = 20, diversity_factor: float = 0.3) -> pd.DataFrame:
        """
        Recomienda tracks basado en contexto temporal.
        
        Args:
            hour: Hora del día (0-23)
            day_of_week: Día de la semana (0=Lunes, 6=Domingo)
            n: Número de recomendaciones
            diversity_factor: Factor de diversidad (0=solo popularidad, 1=máxima diversidad)
            
        Returns:
            DataFrame con tracks recomendados
        """
        if self.track_features is None:
            self.build_track_features()
        
        if self.user_profile is None:
            self.build_user_profile()
        
        # Filtrar tracks por contexto temporal si se especifica
        candidates = self.track_features.copy()
        
        # Score base: popularidad + completion rate
        candidates['base_score'] = (
            candidates['popularity'] * 0.5 +
            candidates['completion_rate'] * 100 * 0.5
        )
        
        # Ajustar por contexto temporal
        if hour is not None and 'typical_hour' in candidates.columns:
            # Bonus si el track típicamente se escucha a esa hora
            hour_diff = np.abs(candidates['typical_hour'] - hour)
            hour_diff = np.minimum(hour_diff, 24 - hour_diff)  # Circular distance
            hour_bonus = (1 - hour_diff / 12) * 20  # 0-20 puntos
            candidates['base_score'] += hour_bonus
        
        if day_of_week is not None and 'typical_day' in candidates.columns:
            day_diff = np.abs(candidates['typical_day'] - day_of_week)
            day_bonus = (1 - day_diff / 7) * 10  # 0-10 puntos
            candidates['base_score'] += day_bonus
        
        # Aplicar factor de diversidad
        if diversity_factor > 0:
            # Penalizar tracks muy populares para aumentar diversidad
            popularity_penalty = candidates['popularity'] * diversity_factor * 0.3
            candidates['final_score'] = candidates['base_score'] - popularity_penalty
        else:
            candidates['final_score'] = candidates['base_score']
        
        # Ordenar y retornar top N
        recommendations = candidates.sort_values('final_score', ascending=False).head(n)
        
        return recommendations
    
    def recommend_based_on_favorites(self, n: int = 20, 
                                    min_similarity: float = 0.6) -> pd.DataFrame:
        """
        Recomienda tracks similares a los favoritos del usuario.
        
        Args:
            n: Número de recomendaciones
            min_similarity: Similitud mínima requerida
            
        Returns:
            DataFrame con recomendaciones
        """
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        # Identificar tracks favoritos (top 10% más escuchados)
        track_col = 'spotify_track_uri'
        play_counts = self.df[track_col].value_counts()
        top_threshold = play_counts.quantile(0.9)
        favorite_tracks = play_counts[play_counts >= top_threshold].index.tolist()
        
        # Obtener índices de favoritos
        favorite_indices = self.track_features[
            self.track_features[track_col].isin(favorite_tracks)
        ].index.tolist()
        
        if not favorite_indices:
            # Fallback: tomar los 5 más escuchados
            favorite_tracks = play_counts.head(5).index.tolist()
            favorite_indices = self.track_features[
                self.track_features[track_col].isin(favorite_tracks)
            ].index.tolist()
        
        # Calcular similitud promedio con favoritos
        similarities = self.similarity_matrix[favorite_indices].mean(axis=0)
        
        # Crear DataFrame con resultados
        results = self.track_features.copy()
        results['similarity_score'] = similarities
        
        # Excluir tracks ya escuchados
        listened_tracks = self.df[track_col].unique()
        results = results[~results[track_col].isin(listened_tracks)]
        
        # Filtrar por similitud mínima
        results = results[results['similarity_score'] >= min_similarity]
        
        # Combinar con popularidad
        results['recommendation_score'] = (
            results['similarity_score'] * 0.7 +
            results['popularity'] / 100 * 0.3
        )
        
        # Ordenar y retornar top N
        recommendations = results.sort_values('recommendation_score', ascending=False).head(n)
        
        return recommendations
    
    def recommend_skip_resistant(self, n: int = 20) -> pd.DataFrame:
        """
        Recomienda tracks con baja probabilidad de skip.
        
        Args:
            n: Número de recomendaciones
            
        Returns:
            DataFrame con tracks recomendados
        """
        if self.track_features is None:
            self.build_track_features()
        
        candidates = self.track_features.copy()
        
        # Filtrar tracks con suficiente data
        candidates = candidates[candidates['total_plays'] >= 3]
        
        if 'skip_rate' not in candidates.columns:
            # Fallback a popularidad si no hay skip data
            return candidates.nlargest(n, 'popularity')
        
        # Score basado en completion rate y popularidad
        candidates['skip_resistant_score'] = (
            candidates['completion_rate'] * 0.7 +
            (candidates['popularity'] / 100) * 0.3
        )
        
        # Ordenar y retornar top N
        recommendations = candidates.sort_values('skip_resistant_score', ascending=False).head(n)
        
        return recommendations
    
    def get_recommendations(self, strategy: str = 'hybrid', n: int = 20,
                           hour: int = None, day_of_week: int = None,
                           **kwargs) -> pd.DataFrame:
        """
        Método unificado para obtener recomendaciones.
        
        Args:
            strategy: Estrategia de recomendación
                - 'hybrid': Combinación de múltiples estrategias
                - 'similar': Basado en tracks similares a favoritos
                - 'context': Basado en contexto temporal
                - 'skip_resistant': Tracks con baja probabilidad de skip
            n: Número de recomendaciones
            hour: Hora del día para recomendaciones contextuales
            day_of_week: Día de la semana para recomendaciones contextuales
            **kwargs: Argumentos adicionales para estrategias específicas
            
        Returns:
            DataFrame con tracks recomendados y scores
        """
        if strategy == 'similar':
            return self.recommend_based_on_favorites(n=n, **kwargs)
        
        elif strategy == 'context':
            return self.recommend_for_context(hour=hour, day_of_week=day_of_week, n=n, **kwargs)
        
        elif strategy == 'skip_resistant':
            return self.recommend_skip_resistant(n=n)
        
        elif strategy == 'hybrid':
            # Combinar múltiples estrategias
            recs_similar = self.recommend_based_on_favorites(n=n*2)
            recs_context = self.recommend_for_context(hour=hour, day_of_week=day_of_week, n=n*2)
            recs_skip = self.recommend_skip_resistant(n=n*2)
            
            # Combinar y rankear
            track_col = 'spotify_track_uri'
            all_recs = pd.concat([recs_similar, recs_context, recs_skip])
            
            # Calcular score híbrido
            score_cols = [col for col in all_recs.columns if 'score' in col.lower()]
            if score_cols:
                all_recs['hybrid_score'] = all_recs[score_cols].mean(axis=1)
            else:
                all_recs['hybrid_score'] = all_recs['popularity']
            
            # Eliminar duplicados (mantener el de mayor score)
            all_recs = all_recs.sort_values('hybrid_score', ascending=False)
            all_recs = all_recs.drop_duplicates(subset=[track_col], keep='first')
            
            return all_recs.head(n)
        
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")
    
    def evaluate_recommendations(self, recommendations: pd.DataFrame,
                                test_df: pd.DataFrame = None) -> Dict[str, float]:
        """
        Evalúa calidad de recomendaciones.
        
        Args:
            recommendations: DataFrame con tracks recomendados
            test_df: DataFrame de test (si está disponible)
            
        Returns:
            Diccionario con métricas de evaluación
        """
        metrics = {}
        
        track_col = 'spotify_track_uri'
        
        # Coverage: % del catálogo recomendado
        total_tracks = self.track_features[track_col].nunique()
        recommended_tracks = recommendations[track_col].nunique()
        metrics['coverage'] = recommended_tracks / total_tracks
        
        # Diversity: Número de artistas únicos
        if 'artist' in recommendations.columns:
            metrics['artist_diversity'] = recommendations['artist'].nunique() / len(recommendations)
        
        # Average popularity
        if 'popularity' in recommendations.columns:
            metrics['avg_popularity'] = recommendations['popularity'].mean()
        
        # Average completion rate
        if 'completion_rate' in recommendations.columns:
            metrics['avg_completion_rate'] = recommendations['completion_rate'].mean()
        
        # Si hay test set, calcular precision@k
        if test_df is not None:
            actual_listened = set(test_df[track_col].unique())
            recommended = set(recommendations[track_col].unique())
            hits = len(recommended.intersection(actual_listened))
            metrics['precision@k'] = hits / len(recommended) if len(recommended) > 0 else 0
            metrics['recall@k'] = hits / len(actual_listened) if len(actual_listened) > 0 else 0
        
        return metrics


def create_recommender_from_data(df: pd.DataFrame) -> TrackRecommender:
    """
    Función helper para crear recomendador listo para usar.
    
    Args:
        df: DataFrame con datos históricos
        
    Returns:
        TrackRecommender inicializado y entrenado
    """
    recommender = TrackRecommender(df)
    recommender.build_track_features()
    recommender.build_track_embeddings()
    recommender.build_similarity_matrix()
    recommender.build_user_profile()
    
    print("\n✓ Recomendador listo para usar")
    print(f"  Tracks en catálogo: {len(recommender.track_features):,}")
    if recommender.user_profile:
        if 'favorite_artists' in recommender.user_profile:
            print(f"  Artistas favoritos: {len(recommender.user_profile['favorite_artists'])}")
        if 'skip_tolerance' in recommender.user_profile:
            print(f"  Skip tolerance: {recommender.user_profile['skip_tolerance']}")
    
    return recommender


if __name__ == '__main__':
    print("Sistema de Recomendación de Tracks")
    print("\nClases disponibles:")
    print("  - TrackRecommender: Recomendador principal")
    print("\nMétodos principales:")
    print("  - get_recommendations(): Método unificado")
    print("  - recommend_based_on_favorites(): Basado en similitud")
    print("  - recommend_for_context(): Basado en contexto temporal")
    print("  - recommend_skip_resistant(): Menor probabilidad de skip")
    print("  - get_similar_tracks(): Tracks similares a uno dado")
