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
