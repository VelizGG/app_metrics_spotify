"""Funciones para entrenamiento y evaluación de modelos predictivos."""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def prepare_skip_prediction_data(df: pd.DataFrame, 
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> Tuple:
    """
    Prepara datos para predecir skipped.
    
    Args:
        df: DataFrame con los datos
        test_size: Proporción para test set
        random_state: Semilla aleatoria
        
    Returns:
        Tupla (X_train, X_test, y_train, y_test, feature_names)
    """
    # Features a usar
    numeric_features = ['ms_played', 'hour', 'day_of_week']
    categorical_features = ['platform']
    
    # Agregar features de sesión si existen
    if 'position_in_session' in df.columns:
        numeric_features.append('position_in_session')
    
    # Filtrar solo columnas que existen
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    all_features = numeric_features + categorical_features
    
    # Preparar X e y
    X = df[all_features].copy()
    y = df['skipped'].astype(int)
    
    # Manejar valores nulos
    X = X.fillna(X.median(numeric_only=True))
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Train set: {len(X_train):,} samples")
    print(f"✓ Test set: {len(X_test):,} samples")
    print(f"✓ Skip rate (train): {y_train.mean():.2%}")
    print(f"✓ Skip rate (test): {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, (numeric_features, categorical_features)


def create_preprocessing_pipeline(numeric_features: list, 
                                  categorical_features: list) -> ColumnTransformer:
    """
    Crea pipeline de preprocessing.
    
    Args:
        numeric_features: Lista de features numéricas
        categorical_features: Lista de features categóricas
        
    Returns:
        ColumnTransformer configurado
    """
    transformers = []
    
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), 
                           categorical_features))
    
    return ColumnTransformer(transformers=transformers)


def train_logistic_regression(X_train, X_test, y_train, y_test,
                               numeric_features, categorical_features) -> Tuple:
    """
    Entrena modelo de Regresión Logística.
    
    Args:
        X_train, X_test, y_train, y_test: Datos de train/test
        numeric_features, categorical_features: Listas de features
        
    Returns:
        Tupla (model, predictions, metrics)
    """
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION")
    print("="*60)
    
    # Crear pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Entrenar
    model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics)
    
    return model, y_pred, y_pred_proba, metrics


def train_random_forest(X_train, X_test, y_train, y_test,
                        numeric_features, categorical_features,
                        n_estimators: int = 100) -> Tuple:
    """
    Entrena modelo de Random Forest.
    
    Args:
        X_train, X_test, y_train, y_test: Datos de train/test
        numeric_features, categorical_features: Listas de features
        n_estimators: Número de árboles
        
    Returns:
        Tupla (model, predictions, metrics)
    """
    print("\n" + "="*60)
    print("RANDOM FOREST")
    print("="*60)
    
    # Crear pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, 
                                             random_state=42,
                                             n_jobs=-1))
    ])
    
    # Entrenar
    model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics)
    
    return model, y_pred, y_pred_proba, metrics


def calculate_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
    """
    Calcula métricas de clasificación.
    
    Args:
        y_true: Labels verdaderas
        y_pred: Predicciones
        y_pred_proba: Probabilidades predichas
        
    Returns:
        Diccionario con métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Imprime métricas de forma formateada."""
    print("\nMétricas:")
    for metric, value in metrics.items():
        print(f"  {metric.upper():12s}: {value:.4f}")


def plot_confusion_matrix(y_true, y_pred, 
                         title: str = 'Matriz de Confusión') -> go.Figure:
    """
    Genera heatmap de matriz de confusión.
    
    Args:
        y_true: Labels verdaderas
        y_pred: Predicciones
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Skip', 'Predicted Skip'],
        y=['Actual No Skip', 'Actual Skip'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicción',
        yaxis_title='Real'
    )
    
    return fig


def plot_roc_curve(y_true, y_pred_proba, 
                   title: str = 'Curva ROC') -> go.Figure:
    """
    Genera curva ROC.
    
    Args:
        y_true: Labels verdaderas
        y_pred_proba: Probabilidades predichas
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {auc:.3f})',
        line=dict(color='#1DB954', width=2)
    ))
    
    # Diagonal de referencia
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    return fig


def plot_precision_recall_curve(y_true, y_pred_proba,
                                title: str = 'Curva Precision-Recall') -> go.Figure:
    """
    Genera curva Precision-Recall.
    
    Args:
        y_true: Labels verdaderas
        y_pred_proba: Probabilidades predichas
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall',
        line=dict(color='#1DB954', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Recall',
        yaxis_title='Precision',
        showlegend=True
    )
    
    return fig


def plot_feature_importance(model, feature_names: list,
                           top_n: int = 10,
                           title: str = 'Feature Importance') -> go.Figure:
    """
    Genera gráfico de importancia de features (solo para Random Forest).
    
    Args:
        model: Modelo entrenado
        feature_names: Nombres de features
        top_n: Top N features a mostrar
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    # Obtener el clasificador del pipeline
    classifier = model.named_steps['classifier']
    
    if not hasattr(classifier, 'feature_importances_'):
        raise ValueError("El modelo no tiene feature_importances_")
    
    # Obtener feature names después del preprocessing
    preprocessor = model.named_steps['preprocessor']
    feature_names_transformed = preprocessor.get_feature_names_out()
    
    # Crear DataFrame de importancias
    importances = pd.DataFrame({
        'feature': feature_names_transformed,
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig = go.Figure(go.Bar(
        x=importances['importance'],
        y=importances['feature'],
        orientation='h',
        marker_color='#1DB954'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance',
        yaxis_title='Feature',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def compare_models(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compara métricas de múltiples modelos.
    
    Args:
        metrics_dict: Diccionario {model_name: metrics_dict}
        
    Returns:
        DataFrame con comparación
    """
    return pd.DataFrame(metrics_dict).T


def predict_skip_probability(model, X: pd.DataFrame) -> np.ndarray:
    """
    Predice probabilidad de skip para nuevos tracks.
    
    Args:
        model: Modelo entrenado (Pipeline)
        X: DataFrame con features
        
    Returns:
        Array con probabilidades de skip
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError("El modelo debe tener método predict_proba")
    
    # Predecir probabilidades
    proba = model.predict_proba(X)[:, 1]  # Probabilidad de skip (clase 1)
    
    return proba


def recommendation_score(tracks_df: pd.DataFrame, 
                        skip_model = None,
                        popularity_weight: float = 0.3,
                        completion_weight: float = 0.4,
                        skip_pred_weight: float = 0.3) -> pd.DataFrame:
    """
    Calcula score de recomendación combinando múltiples factores.
    
    Args:
        tracks_df: DataFrame con features de tracks
        skip_model: Modelo entrenado para predecir skip (opcional)
        popularity_weight: Peso de popularidad (0-1)
        completion_weight: Peso de completion rate (0-1)
        skip_pred_weight: Peso de predicción de skip (0-1)
        
    Returns:
        DataFrame con scores de recomendación
    """
    df = tracks_df.copy()
    
    # Normalizar popularidad (0-100)
    if 'total_plays' in df.columns:
        df['popularity_norm'] = (df['total_plays'] / df['total_plays'].max() * 100)
    else:
        df['popularity_norm'] = 50  # Default
    
    # Completion rate
    if 'skip_rate' in df.columns:
        df['completion_rate'] = (1 - df['skip_rate']) * 100
    elif 'completion_rate' in df.columns:
        df['completion_rate'] = df['completion_rate'] * 100
    else:
        df['completion_rate'] = 70  # Default
    
    # Predicción de skip (si hay modelo)
    if skip_model is not None:
        try:
            # Preparar features para el modelo
            feature_cols = ['ms_played', 'hour', 'day_of_week', 'platform']
            
            # Usar avg_duration si existe
            if 'avg_duration' in df.columns:
                df['ms_played'] = df['avg_duration']
            
            # Usar typical_hour/day si existen
            if 'typical_hour' in df.columns:
                df['hour'] = df['typical_hour']
            if 'typical_day' in df.columns:
                df['day_of_week'] = df['typical_day']
            
            # Filtrar features disponibles
            available_features = [f for f in feature_cols if f in df.columns]
            
            if len(available_features) >= 3:
                X_pred = df[available_features].fillna(0)
                skip_probs = predict_skip_probability(skip_model, X_pred)
                df['skip_resistance'] = (1 - skip_probs) * 100
            else:
                df['skip_resistance'] = df['completion_rate']
        except Exception as e:
            print(f"⚠️ No se pudo usar skip model: {e}")
            df['skip_resistance'] = df['completion_rate']
    else:
        df['skip_resistance'] = df['completion_rate']
    
    # Calcular score final
    df['recommendation_score'] = (
        df['popularity_norm'] * popularity_weight +
        df['completion_rate'] * completion_weight +
        df['skip_resistance'] * skip_pred_weight
    )
    
    return df


def evaluate_recommendation_quality(recommendations: pd.DataFrame,
                                    actual_played: pd.DataFrame,
                                    track_col: str = 'spotify_track_uri') -> Dict[str, float]:
    """
    Evalúa calidad de recomendaciones contra datos reales.
    
    Args:
        recommendations: DataFrame con tracks recomendados
        actual_played: DataFrame con tracks realmente escuchados
        track_col: Nombre de la columna de track
        
    Returns:
        Diccionario con métricas de evaluación
    """
    # Conjuntos
    recommended_set = set(recommendations[track_col].unique())
    actual_set = set(actual_played[track_col].unique())
    
    # Hits
    hits = len(recommended_set.intersection(actual_set))
    
    # Métricas
    metrics = {
        'precision@k': hits / len(recommended_set) if len(recommended_set) > 0 else 0,
        'recall@k': hits / len(actual_set) if len(actual_set) > 0 else 0,
        'coverage': len(recommended_set) / len(actual_set) if len(actual_set) > 0 else 0,
        'n_recommendations': len(recommended_set),
        'n_actual': len(actual_set),
        'hits': hits
    }
    
    # F1 score
    if metrics['precision@k'] + metrics['recall@k'] > 0:
        metrics['f1@k'] = 2 * (metrics['precision@k'] * metrics['recall@k']) / \
                         (metrics['precision@k'] + metrics['recall@k'])
    else:
        metrics['f1@k'] = 0
    
    return metrics


def plot_recommendation_evaluation(metrics: Dict[str, float],
                                   title: str = 'Evaluación de Recomendaciones') -> go.Figure:
    """
    Visualiza métricas de evaluación de recomendaciones.
    
    Args:
        metrics: Diccionario con métricas
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    # Métricas a visualizar
    metric_names = ['precision@k', 'recall@k', 'f1@k', 'coverage']
    metric_values = [metrics.get(m, 0) for m in metric_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metric_names,
            y=metric_values,
            text=[f'{v:.2%}' for v in metric_values],
            textposition='auto',
            marker_color='#1DB954'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Métrica',
        yaxis_title='Valor',
        yaxis_tickformat='.0%',
        height=400
    )
    
    return fig


def analyze_playlist_diversity(playlist: pd.DataFrame,
                               artist_col: str = 'artist',
                               track_col: str = 'spotify_track_uri') -> Dict[str, any]:
    """
    Analiza diversidad de una playlist.
    
    Args:
        playlist: DataFrame con tracks de la playlist
        artist_col: Nombre de la columna de artista
        track_col: Nombre de la columna de track
        
    Returns:
        Diccionario con métricas de diversidad
    """
    metrics = {}
    
    # Número de tracks
    metrics['n_tracks'] = len(playlist)
    
    # Artistas únicos
    if artist_col in playlist.columns:
        metrics['n_artists'] = playlist[artist_col].nunique()
        metrics['artist_diversity'] = metrics['n_artists'] / metrics['n_tracks']
        
        # Distribución de artistas
        artist_counts = playlist[artist_col].value_counts()
        metrics['max_tracks_per_artist'] = artist_counts.max()
        metrics['avg_tracks_per_artist'] = artist_counts.mean()
    
    # Variabilidad temporal
    if 'typical_hour' in playlist.columns:
        metrics['hour_std'] = playlist['typical_hour'].std()
        metrics['hour_range'] = (playlist['typical_hour'].min(), 
                                playlist['typical_hour'].max())
    
    # Variabilidad de duración
    if 'avg_duration' in playlist.columns:
        metrics['duration_std'] = playlist['avg_duration'].std()
        metrics['avg_duration'] = playlist['avg_duration'].mean()
    
    # Skip rate promedio
    if 'skip_rate' in playlist.columns:
        metrics['avg_skip_rate'] = playlist['skip_rate'].mean()
    
    return metrics


def plot_playlist_summary(playlist: pd.DataFrame,
                         title: str = 'Resumen de Playlist') -> go.Figure:
    """
    Visualiza resumen de una playlist.
    
    Args:
        playlist: DataFrame con tracks de la playlist
        title: Título del gráfico
        
    Returns:
        Figura de Plotly con subplots
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top Artistas', 'Distribución de Hora',
                       'Popularidad', 'Skip Rate'),
        specs=[[{'type': 'bar'}, {'type': 'histogram'}],
               [{'type': 'histogram'}, {'type': 'histogram'}]]
    )
    
    # Top artistas
    if 'artist' in playlist.columns:
        top_artists = playlist['artist'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=top_artists.values, y=top_artists.index, orientation='h',
                  marker_color='#1DB954'),
            row=1, col=1
        )
    
    # Distribución de hora
    if 'typical_hour' in playlist.columns:
        fig.add_trace(
            go.Histogram(x=playlist['typical_hour'], nbinsx=24,
                        marker_color='#1DB954'),
            row=1, col=2
        )
    
    # Popularidad
    if 'play_count' in playlist.columns:
        fig.add_trace(
            go.Histogram(x=playlist['play_count'],
                        marker_color='#1DB954'),
            row=2, col=1
        )
    
    # Skip rate
    if 'skip_rate' in playlist.columns:
        fig.add_trace(
            go.Histogram(x=playlist['skip_rate'],
                        marker_color='#1DB954'),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text=title,
        showlegend=False,
        height=600
    )
    
    return fig


if __name__ == '__main__':
    print("Módulo de entrenamiento y evaluación de modelos")
    print("\nFunciones disponibles:")
    print("  - prepare_skip_prediction_data(): Preparar datos para skip prediction")
    print("  - train_logistic_regression(): Entrenar Logistic Regression")
    print("  - train_random_forest(): Entrenar Random Forest")
    print("  - plot_confusion_matrix(): Matriz de confusión")
    print("  - plot_roc_curve(): Curva ROC")
    print("  - plot_feature_importance(): Importancia de features")
    print("  - compare_models(): Comparar modelos")
    print("\nFunciones de recomendación:")
    print("  - recommendation_score(): Score de recomendación combinado")
    print("  - predict_skip_probability(): Predecir probabilidad de skip")
    print("  - evaluate_recommendation_quality(): Evaluar recomendaciones")
    print("  - analyze_playlist_diversity(): Analizar diversidad de playlist")
    print("  - plot_recommendation_evaluation(): Visualizar evaluación")
    print("  - plot_playlist_summary(): Resumen visual de playlist")
