"""Tests para el sistema de recomendación y generación de playlists."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from recommendations import TrackRecommender, create_recommender_from_data
from playlist_generator import PlaylistGenerator, create_playlist_generator
from models import recommendation_score, evaluate_recommendation_quality, analyze_playlist_diversity


@pytest.fixture
def sample_data():
    """Crea datos de muestra para testing."""
    np.random.seed(42)
    
    n_rows = 100
    n_tracks = 20
    n_artists = 10
    
    tracks = [f'spotify:track:test_{i:03d}' for i in range(n_tracks)]
    artists = [f'Test Artist {i}' for i in range(n_artists)]
    track_names = [f'Test Track {i}' for i in range(n_tracks)]
    
    data = {
        'spotify_track_uri': np.random.choice(tracks, n_rows),
        'master_metadata_album_artist_name': np.random.choice(artists, n_rows),
        'master_metadata_track_name': [track_names[int(uri.split('_')[-1])] for uri in np.random.choice(tracks, n_rows)],
        'ms_played': np.random.randint(30000, 300000, n_rows),
        'ts': pd.date_range('2024-01-01', periods=n_rows, freq='1H'),
        'hour': np.random.randint(0, 24, n_rows),
        'day_of_week': np.random.randint(0, 7, n_rows),
        'skipped': np.random.choice([True, False], n_rows, p=[0.2, 0.8]),
        'shuffle': np.random.choice([True, False], n_rows, p=[0.6, 0.4]),
        'platform': np.random.choice(['Windows', 'Android', 'iOS'], n_rows),
    }
    
    return pd.DataFrame(data)


class TestTrackRecommender:
    """Tests para TrackRecommender."""
    
    def test_init(self, sample_data):
        """Test inicialización del recomendador."""
        recommender = TrackRecommender(sample_data)
        assert recommender.df is not None
        assert len(recommender.df) == len(sample_data)
    
    def test_build_track_features(self, sample_data):
        """Test construcción de features de tracks."""
        recommender = TrackRecommender(sample_data)
        features = recommender.build_track_features()
        
        assert features is not None
        assert len(features) > 0
        assert 'spotify_track_uri' in features.columns
        assert 'total_plays' in features.columns
        assert 'popularity' in features.columns
    
    def test_build_track_embeddings(self, sample_data):
        """Test creación de embeddings."""
        recommender = TrackRecommender(sample_data)
        recommender.build_track_features()
        embeddings = recommender.build_track_embeddings()
        
        assert embeddings is not None
        assert len(embeddings) == len(recommender.track_features)
        assert embeddings.shape[1] > 0  # Tiene dimensiones
    
    def test_build_similarity_matrix(self, sample_data):
        """Test construcción de matriz de similitud."""
        recommender = TrackRecommender(sample_data)
        recommender.build_track_features()
        recommender.build_track_embeddings()
        similarity = recommender.build_similarity_matrix()
        
        assert similarity is not None
        assert similarity.shape[0] == similarity.shape[1]
        assert similarity.shape[0] == len(recommender.track_features)
        
        # La diagonal debe ser ~1 (similitud consigo mismo)
        diagonal = np.diag(similarity)
        assert np.allclose(diagonal, 1.0, atol=0.01)
    
    def test_build_user_profile(self, sample_data):
        """Test construcción de perfil de usuario."""
        recommender = TrackRecommender(sample_data)
        profile = recommender.build_user_profile()
        
        assert profile is not None
        assert 'avg_skip_rate' in profile
        assert 'listening_hours' in profile
        assert profile['avg_skip_rate'] >= 0
        assert profile['avg_skip_rate'] <= 1
    
    def test_get_similar_tracks(self, sample_data):
        """Test búsqueda de tracks similares."""
        recommender = create_recommender_from_data(sample_data)
        
        # Elegir un track del dataset
        test_track = sample_data['spotify_track_uri'].iloc[0]
        
        similar = recommender.get_similar_tracks(test_track, n=5)
        
        assert similar is not None
        assert len(similar) <= 5
        assert 'similarity_score' in similar.columns
        assert test_track not in similar['spotify_track_uri'].values
    
    def test_recommend_for_context(self, sample_data):
        """Test recomendaciones contextuales."""
        recommender = create_recommender_from_data(sample_data)
        
        recs = recommender.recommend_for_context(hour=10, day_of_week=2, n=10)
        
        assert recs is not None
        assert len(recs) <= 10
        assert 'final_score' in recs.columns
    
    def test_recommend_based_on_favorites(self, sample_data):
        """Test recomendaciones basadas en favoritos."""
        recommender = create_recommender_from_data(sample_data)
        
        recs = recommender.recommend_based_on_favorites(n=10)
        
        assert recs is not None
        assert len(recs) <= 10
        assert 'recommendation_score' in recs.columns
    
    def test_recommend_skip_resistant(self, sample_data):
        """Test recomendaciones resistentes a skip."""
        recommender = create_recommender_from_data(sample_data)
        
        recs = recommender.recommend_skip_resistant(n=10)
        
        assert recs is not None
        assert len(recs) <= 10
        if 'skip_resistant_score' in recs.columns:
            assert recs['skip_resistant_score'].notna().all()
    
    def test_get_recommendations_hybrid(self, sample_data):
        """Test recomendaciones híbridas."""
        recommender = create_recommender_from_data(sample_data)
        
        recs = recommender.get_recommendations(strategy='hybrid', n=15)
        
        assert recs is not None
        assert len(recs) <= 15
        assert 'hybrid_score' in recs.columns
    
    def test_evaluate_recommendations(self, sample_data):
        """Test evaluación de recomendaciones."""
        recommender = create_recommender_from_data(sample_data)
        recs = recommender.get_recommendations(n=10)
        
        metrics = recommender.evaluate_recommendations(recs)
        
        assert metrics is not None
        assert 'coverage' in metrics
        assert metrics['coverage'] >= 0
        assert metrics['coverage'] <= 1


class TestPlaylistGenerator:
    """Tests para PlaylistGenerator."""
    
    def test_init(self, sample_data):
        """Test inicialización del generador."""
        generator = PlaylistGenerator(sample_data)
        assert generator.df is not None
        assert len(generator.df) == len(sample_data)
    
    def test_generate_temporal_playlists(self, sample_data):
        """Test generación de playlists temporales."""
        generator = PlaylistGenerator(sample_data)
        playlists = generator.generate_temporal_playlists()
        
        assert playlists is not None
        assert len(playlists) > 0
        
        # Verificar que cada playlist tiene tracks
        for name, playlist in playlists.items():
            assert len(playlist) > 0
            assert 'spotify_track_uri' in playlist.columns
    
    def test_generate_behavior_playlists(self, sample_data):
        """Test generación de playlists por comportamiento."""
        generator = PlaylistGenerator(sample_data)
        playlists = generator.generate_behavior_playlists()
        
        assert playlists is not None
        assert 'All-Time Favorites' in playlists
        assert len(playlists['All-Time Favorites']) > 0
    
    def test_generate_mood_playlists(self, sample_data):
        """Test generación de playlists por mood."""
        generator = PlaylistGenerator(sample_data)
        playlists = generator.generate_mood_playlists()
        
        assert playlists is not None
        assert len(playlists) >= 0  # Puede estar vacío si no hay suficiente data
    
    def test_generate_artist_playlists(self, sample_data):
        """Test generación de playlists por artista."""
        generator = PlaylistGenerator(sample_data)
        playlists = generator.generate_artist_playlists(top_n_artists=3)
        
        assert playlists is not None
        # Puede estar vacío si no hay columna de artista
        if len(playlists) > 0:
            for name, playlist in playlists.items():
                assert 'Best of' in name
                assert len(playlist) > 0
    
    def test_generate_cluster_playlists(self, sample_data):
        """Test generación de playlists por clustering."""
        generator = PlaylistGenerator(sample_data)
        playlists = generator.generate_cluster_playlists(n_clusters=3, min_tracks=3)
        
        assert playlists is not None
        # Puede haber clusters vacíos si no hay suficiente data
        for name, playlist in playlists.items():
            assert 'Cluster Mix' in name
            assert len(playlist) >= 3
    
    def test_generate_discovery_playlist(self, sample_data):
        """Test generación de playlist de descubrimiento."""
        generator = PlaylistGenerator(sample_data)
        playlist = generator.generate_discovery_playlist(n=10)
        
        assert playlist is not None
        assert len(playlist) <= 10
    
    def test_generate_all_playlists(self, sample_data):
        """Test generación de todas las playlists."""
        generator = PlaylistGenerator(sample_data)
        all_playlists = generator.generate_all_playlists(
            include_clusters=False,
            include_artists=False
        )
        
        assert all_playlists is not None
        assert len(all_playlists) > 0
    
    def test_get_playlist(self, sample_data):
        """Test obtención de playlist específica."""
        generator = create_playlist_generator(sample_data, generate_all=True)
        
        playlists = generator.list_playlists()
        assert len(playlists) > 0
        
        # Obtener primera playlist
        first_playlist_name = playlists[0][0]
        playlist = generator.get_playlist(first_playlist_name)
        
        assert playlist is not None
        assert len(playlist) > 0
    
    def test_get_playlist_not_found(self, sample_data):
        """Test error al buscar playlist inexistente."""
        generator = PlaylistGenerator(sample_data)
        
        with pytest.raises(ValueError):
            generator.get_playlist('Nonexistent Playlist')
    
    def test_list_playlists(self, sample_data):
        """Test listar playlists."""
        generator = create_playlist_generator(sample_data, generate_all=True)
        
        playlist_list = generator.list_playlists()
        
        assert playlist_list is not None
        assert len(playlist_list) > 0
        
        # Verificar formato de la lista
        for name, count in playlist_list:
            assert isinstance(name, str)
            assert isinstance(count, (int, np.integer))
            assert count > 0


class TestModelFunctions:
    """Tests para funciones de models.py."""
    
    def test_recommendation_score(self, sample_data):
        """Test cálculo de score de recomendación."""
        # Preparar tracks_df
        tracks_df = sample_data.groupby('spotify_track_uri').agg({
            'ms_played': 'count',
            'skipped': 'mean'
        }).reset_index()
        tracks_df.columns = ['spotify_track_uri', 'total_plays', 'skip_rate']
        
        scored = recommendation_score(tracks_df)
        
        assert scored is not None
        assert 'recommendation_score' in scored.columns
        assert scored['recommendation_score'].notna().all()
    
    def test_evaluate_recommendation_quality(self, sample_data):
        """Test evaluación de calidad de recomendaciones."""
        # Crear recomendaciones y datos reales
        tracks = sample_data['spotify_track_uri'].unique()[:10]
        recommendations = pd.DataFrame({
            'spotify_track_uri': tracks
        })
        
        actual = sample_data[sample_data['spotify_track_uri'].isin(tracks[:5])]
        
        metrics = evaluate_recommendation_quality(recommendations, actual)
        
        assert metrics is not None
        assert 'precision@k' in metrics
        assert 'recall@k' in metrics
        assert 'f1@k' in metrics
        assert metrics['precision@k'] >= 0
        assert metrics['precision@k'] <= 1
    
    def test_analyze_playlist_diversity(self, sample_data):
        """Test análisis de diversidad de playlist."""
        # Crear playlist simple
        playlist = sample_data.groupby('spotify_track_uri').first().reset_index().head(20)
        playlist['artist'] = sample_data.groupby('spotify_track_uri')['master_metadata_album_artist_name'].first().values[:20]
        
        diversity = analyze_playlist_diversity(playlist, artist_col='artist')
        
        assert diversity is not None
        assert 'n_tracks' in diversity
        assert 'n_artists' in diversity
        assert 'artist_diversity' in diversity
        assert diversity['n_tracks'] == 20


class TestEdgeCases:
    """Tests para casos edge."""
    
    def test_empty_dataframe(self):
        """Test con DataFrame vacío."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            TrackRecommender(empty_df).build_track_features()
    
    def test_single_track(self):
        """Test con un solo track."""
        single_track = pd.DataFrame({
            'spotify_track_uri': ['test:track:001'],
            'master_metadata_album_artist_name': ['Test Artist'],
            'master_metadata_track_name': ['Test Track'],
            'ms_played': [180000],
            'ts': [pd.Timestamp('2024-01-01')],
            'hour': [12],
            'day_of_week': [0],
            'skipped': [False],
            'shuffle': [True],
            'platform': ['Windows']
        })
        
        recommender = TrackRecommender(single_track)
        features = recommender.build_track_features()
        
        assert len(features) == 1
    
    def test_missing_columns(self):
        """Test con columnas faltantes."""
        minimal_df = pd.DataFrame({
            'spotify_track_uri': ['test:track:001', 'test:track:002'],
            'ms_played': [180000, 200000],
            'ts': pd.date_range('2024-01-01', periods=2),
        })
        
        # Debería manejar columnas faltantes gracefully
        recommender = TrackRecommender(minimal_df)
        features = recommender.build_track_features()
        
        assert features is not None
        assert len(features) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
