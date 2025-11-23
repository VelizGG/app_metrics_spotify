# Best Practices - Spotify Analytics Project

## 📋 Índice

1. [Privacidad y Seguridad de Datos](#privacidad-y-seguridad-de-datos)
2. [Estructura de Notebooks](#estructura-de-notebooks)
3. [Código Limpio](#código-limpio)
4. [Documentación](#documentación)
5. [Testing](#testing)
6. [Git Workflow](#git-workflow)

---

## 🔒 Privacidad y Seguridad de Datos

### ⚠️ NUNCA compartir datos personales

Los datos reales de Spotify contienen:
- ✗ IPs
- ✗ Timestamps exactos
- ✗ Patrones de escucha personal
- ✗ Ubicaciones geográficas

### ✅ Para Portfolio Público

**Opción 1: Usar datos sintéticos**
```python
from src.generate_synthetic_data import generate_synthetic_spotify_data

# Generar dataset demo
df = generate_synthetic_spotify_data(
    n_rows=50000,
    output_path='data/demo/synthetic_spotify_data.parquet'
)
```

**Opción 2: Anonimizar datos reales**
```python
import pandas as pd

# Anonimizar datos
df['ip_addr'] = 'xxx.xxx.xxx.xxx'  # Enmascarar IPs
df['ts'] = df['ts'] - pd.Timedelta(days=365)  # Desplazar timestamps
df = df.sample(frac=0.1)  # Usar solo 10% de datos
```

### 📁 Estructura de Data Folders

```
data/
├── raw/           # ❌ NO COMPARTIR - Ignorado por git
├── curated/       # ❌ NO COMPARTIR - Ignorado por git
├── demo/          # ✅ SEGURO - Datos sintéticos para portfolio
└── features/      # ⚠️ VERIFICAR - Pueden contener info sensible
```

---

## 📓 Estructura de Notebooks

### Template Profesional

Cada notebook debe seguir esta estructura:

```markdown
# [Número] - [Título del Análisis]

**Autor:** Tu Nombre
**Fecha:** Mes Año
**Proyecto:** Nombre del Proyecto

---

## Contexto / Business Problem

[Explicar el problema de negocio y objetivos]

## Objetivos

1. Objetivo 1
2. Objetivo 2
3. ...

**Skills Demostradas:**
- Skill 1
- Skill 2

---

## 1. Setup

[Imports y configuración]

## 2-N. Secciones de Análisis

[Código + visualizaciones + insights]

## Conclusiones

### Hallazgos Clave
[Resumen de insights]

### Recomendaciones
[Acciones sugeridas]

### Next Steps
[Próximos análisis/mejoras]
```

### 🎨 Estilo de Código en Notebooks

```python
# ✅ BIEN: Celdas cortas y enfocadas
df = pd.read_parquet('data.parquet')
print(f"Datos cargados: {df.shape}")

# ❌ MAL: Celdas largas con múltiples tareas
df = pd.read_parquet('data.parquet')
df['new_col'] = df['col1'] * 2
df = df.dropna()
results = df.groupby('col2').agg({'col3': 'sum'})
# ... 50 líneas más ...
```

### 📊 Visualizaciones

```python
# ✅ BIEN: Configuración completa
fig = px.bar(
    data,
    x='category',
    y='value',
    title='Clear Descriptive Title',
    labels={'category': 'Category Name', 'value': 'Metric Name'},
    color_discrete_sequence=['#1DB954']
)
fig.update_layout(height=500)
fig.show()

# ❌ MAL: Plot sin contexto
df.plot()
```

---

## 🧹 Código Limpio

### Naming Conventions

```python
# ✅ Variables: snake_case descriptivo
total_plays = df['plays'].sum()
avg_duration_minutes = df['duration_ms'].mean() / 60000

# ✅ Funciones: verbos descriptivos
def calculate_session_duration(df):
    """Calcula la duración de cada sesión en minutos."""
    pass

# ❌ MAL: Nombres no descriptivos
x = df['plays'].sum()
calc = df['duration_ms'].mean() / 60000
def func1(d):
    pass
```

### Funciones Modulares

```python
# ✅ BIEN: Función pequeña con un propósito
def load_data(path: Path) -> pd.DataFrame:
    """Carga datos desde archivo parquet."""
    return pd.read_parquet(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia datos: elimina nulls y duplicados."""
    df = df.dropna(subset=['track_id'])
    df = df.drop_duplicates()
    return df

# Uso
df = load_data(data_path)
df = clean_data(df)

# ❌ MAL: Función monolítica
def process_everything(path):
    # 100 líneas de código haciendo todo
    pass
```

### Type Hints

```python
# ✅ BIEN: Con type hints
def sessionize(
    df: pd.DataFrame,
    ts_col: str = 'ts',
    gap: pd.Timedelta = pd.Timedelta('30min')
) -> pd.DataFrame:
    """Agrupa reproducciones en sesiones."""
    pass

# ❌ MAL: Sin types
def sessionize(df, ts_col='ts', gap=pd.Timedelta('30min')):
    pass
```

### Docstrings

```python
# ✅ BIEN: Docstring completo
def calculate_skip_rate(df: pd.DataFrame) -> float:
    """
    Calcula el skip rate del dataset.
    
    Args:
        df: DataFrame con columna 'skipped' (bool)
        
    Returns:
        Skip rate como float entre 0 y 1
        
    Example:
        >>> df = pd.DataFrame({'skipped': [True, False, True]})
        >>> calculate_skip_rate(df)
        0.667
    """
    return df['skipped'].mean()
```

---

## 📝 Documentación

### README.md

Debe incluir:
- ✅ Descripción clara del proyecto
- ✅ Skills demostradas
- ✅ Quick Start / Installation
- ✅ Estructura del proyecto
- ✅ Ejemplos de uso
- ✅ Advertencias de privacidad
- ✅ Contacto

### Comentarios en Código

```python
# ✅ BIEN: Explica el "por qué"
# Usar 30min gap basado en análisis exploratorio de comportamiento de usuario
SESSION_GAP = pd.Timedelta('30min')

# ❌ MAL: Explica el "qué" (obvio del código)
# Crear variable con timedelta
SESSION_GAP = pd.Timedelta('30min')
```

---

## ✅ Testing

### Estructura de Tests

```python
import pytest
import pandas as pd
from src.data_pipeline import clean_data

@pytest.fixture
def sample_df():
    """Fixture con datos de prueba."""
    return pd.DataFrame({
        'track_id': ['t1', 't2', None, 't3'],
        'duration_ms': [180000, 200000, 150000, None]
    })

def test_clean_data_removes_nulls(sample_df):
    """Test que clean_data elimina filas con nulls."""
    result = clean_data(sample_df)
    assert result['track_id'].isna().sum() == 0
    assert len(result) == 2  # Solo 2 filas sin nulls en ambas cols
```

### Coverage

Apuntar a >80% coverage en módulos core:
```bash
pytest --cov=src --cov-report=html
```

---

## 🌲 Git Workflow

### Commits

```bash
# ✅ BIEN: Mensaje descriptivo
git commit -m "feat: Add sessionization function with 30min gap"
git commit -m "fix: Handle null values in track_features function"
git commit -m "docs: Update README with synthetic data instructions"

# ❌ MAL: Mensaje vago
git commit -m "update"
git commit -m "fix bug"
git commit -m "changes"
```

### Branches

```bash
# Feature development
git checkout -b feature/time-series-analysis

# Bug fixes
git checkout -b fix/dashboard-username-error

# Merge cuando esté listo
git checkout main
git merge feature/time-series-analysis
```

### .gitignore

**Crítico**: NUNCA commitear datos personales
```gitignore
# Datos sensibles
data/raw/**
data/curated/**

# Permitir datos demo
!data/demo/**
```

---

## 📈 Checklist para Portfolio

Antes de compartir públicamente:

### Privacidad
- [ ] ✅ Usar datos sintéticos o anonimizados
- [ ] ✅ Verificar que data/raw está en .gitignore
- [ ] ✅ Revisar notebooks por info personal
- [ ] ✅ Enmascarar IPs, timestamps, ubicaciones

### Calidad de Código
- [ ] ✅ Código modular y limpio
- [ ] ✅ Type hints en funciones
- [ ] ✅ Docstrings completos
- [ ] ✅ Tests pasando (pytest)
- [ ] ✅ CI/CD verde

### Documentación
- [ ] ✅ README profesional
- [ ] ✅ Notebooks con estructura clara
- [ ] ✅ Comentarios en código complejo
- [ ] ✅ Skills destacadas claramente

### Presentación
- [ ] ✅ Visualizaciones profesionales
- [ ] ✅ Insights de negocio claros
- [ ] ✅ Conclusiones accionables
- [ ] ✅ Next steps definidos

---

## 🎓 Recursos Adicionales

- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Clean Code Python](https://github.com/zedr/clean-code-python)
- [Effective Pandas](https://pandas.pydata.org/docs/user_guide/style.ipynb)

---

**Recuerda**: El código es leído 10 veces más que escrito. ¡Hazlo claro y profesional! 🚀
