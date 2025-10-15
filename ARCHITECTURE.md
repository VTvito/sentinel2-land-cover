# 🏗️ Architettura del Sistema

## 📋 Indice
1. [Overview](#overview)
2. [Struttura dei Moduli](#struttura-dei-moduli)
3. [Flusso di Esecuzione](#flusso-di-esecuzione)
4. [Design Patterns](#design-patterns)
5. [Diagrammi di Sequenza](#diagrammi-di-sequenza)

---

## Overview

Il sistema è organizzato seguendo principi di **clean architecture** con separazione delle responsabilità:

```
satellite_analysis/
├── config/          → Configurazione centralizzata
├── downloaders/     → Download e autenticazione
├── preprocessors/   → Processing immagini [TODO]
├── analyzers/       → Algoritmi di analisi (clustering)
├── classifiers/     → Classificazione [TODO]
├── utils/           → Utilità condivise
└── pipelines/       → Orchestrazione workflow [TODO]
```

---

## Struttura dei Moduli

### 1. **Configuration Module** (`config/`)

**Responsabilità**: Gestione centralizzata della configurazione

```python
from satellite_analysis.config import Config

# Carica da YAML
config = Config.from_yaml("config/config.yaml")

# Accedi ai parametri
config.area.city              # "Milan"
config.sentinel.client_id     # OAuth2 credentials
config.sentinel.max_cloud_cover  # 10.0
```

**Componenti**:
- `settings.py`: Dataclasses per configurazione tipizzata
- `config.yaml`: File di configurazione in formato YAML

**Design Pattern**: **Configuration Object Pattern**

---

### 2. **Downloaders Module** (`downloaders/`)

**Responsabilità**: Autenticazione e ricerca nel catalogo Sentinel Hub

#### 2.1 Authentication (`downloaders/auth/`)

**Strategy Pattern** per supportare multiple strategie di autenticazione:

```python
from satellite_analysis.downloaders.auth import OAuth2AuthStrategy

# Crea strategia OAuth2
auth = OAuth2AuthStrategy(
    client_id="sh-xxx",
    client_secret="yyy"
)

# Ottieni sessione autenticata
session = auth.get_session()

# Verifica validità token
is_valid = auth.is_valid()
```

**Flusso Interno**:
```
OAuth2AuthStrategy
  │
  ├─ __init__(client_id, secret)
  │   └─ Inizializza attributi (_session, _token)
  │
  ├─ get_session()
  │   ├─ Check: session exists?
  │   │   └─ No → _authenticate()
  │   ├─ Check: is_valid()?
  │   │   └─ No → refresh()
  │   └─ Return: OAuth2Session
  │
  ├─ _authenticate()
  │   ├─ Create: BackendApplicationClient
  │   ├─ Create: OAuth2Session
  │   ├─ POST: https://identity.dataspace.copernicus.eu/.../token
  │   └─ Store: _token, _session
  │
  ├─ is_valid()
  │   ├─ Check: _session and _token exist?
  │   ├─ Check: expires_at field present?
  │   └─ Return: time.time() < (expires_at - 60)
  │
  └─ refresh()
      └─ Call: _authenticate()
```

**Componenti**:
- `AuthStrategy` (ABC): Interfaccia base
- `OAuth2AuthStrategy`: Implementazione OAuth2 per Copernicus

**Endpoint API**: `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`

---

#### 2.2 Catalog (`downloaders/catalog/`)

**Strategy Pattern** per supportare multiple fonti di catalogo:

```python
from satellite_analysis.downloaders.catalog import SentinelHubCatalog

# Crea catalogo con sessione autenticata
catalog = SentinelHubCatalog(session)

# Cerca prodotti
results = catalog.search(
    bbox=[9.0, 45.3, 9.3, 45.6],  # Milano
    start_date="2023-03-01",
    end_date="2023-03-15",
    collection="sentinel-2-l2a",
    cloud_cover_max=10.0,
    limit=5
)

# Risultati
features = results["features"]  # Lista di prodotti
```

**Flusso Interno**:
```
SentinelHubCatalog
  │
  ├─ __init__(session)
  │   └─ Store: session, CATALOG_URL
  │
  └─ search(bbox, dates, cloud_cover, limit)
      │
      ├─ Validation Phase
      │   ├─ _validate_bbox(bbox)
      │   │   └─ Check: -180 ≤ lon ≤ 180, -90 ≤ lat ≤ 90
      │   ├─ _validate_dates(start, end)
      │   │   └─ Check: start < end, formato ISO
      │   └─ _validate_cloud_cover(value)
      │       └─ Check: 0 ≤ value ≤ 100
      │
      ├─ Build Query
      │   └─ STAC format:
      │       {
      │         "collections": ["sentinel-2-l2a"],
      │         "datetime": "2023-03-01T00:00:00Z/2023-03-15T23:59:59Z",
      │         "bbox": [9.0, 45.3, 9.3, 45.6],
      │         "limit": 5
      │       }
      │
      ├─ API Request
      │   └─ POST: https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search
      │       Headers: {"Content-Type": "application/json"}
      │       Auth: OAuth2 token in session
      │
      ├─ Client-side Filtering
      │   └─ Filter by: eo:cloud_cover ≤ max_cloud_cover
      │
      └─ Return Results
          └─ {
              "features": [
                {
                  "properties": {
                    "datetime": "2023-03-12T10:28:21Z",
                    "eo:cloud_cover": 3.88,
                    "platform": "sentinel-2a",
                    ...
                  },
                  "geometry": {...},
                  "assets": {...}
                }
              ]
            }
```

**Componenti**:
- `CatalogStrategy` (ABC): Interfaccia base
- `SentinelHubCatalog`: Implementazione Sentinel Hub STAC API

**Endpoint API**: `https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search`

**Note**: Il filtraggio cloud cover è fatto client-side perché l'API non supporta filtri STAC avanzati.

---

### 3. **Analyzers Module** (`analyzers/clustering/`)

**Responsabilità**: Algoritmi di clustering per analisi immagini

**Factory Pattern** per creare algoritmi:

```python
from satellite_analysis.analyzers.clustering import ClusteringFactory

# Crea algoritmo tramite factory
clusterer = ClusteringFactory.create(
    strategy='kmeans++',
    n_clusters=5,
    max_iters=100
)

# Fit e predict
labels = clusterer.fit_predict(data)
```

**Flusso Interno**:
```
ClusteringFactory
  │
  ├─ create(strategy, **kwargs)
  │   ├─ Check: strategy in _strategies?
  │   └─ Return: _strategies[strategy](**kwargs)
  │
  └─ _strategies = {
      'kmeans': KMeansClusterer,
      'kmeans++': KMeansPlusPlusClusterer,
      'sklearn': SklearnKMeansClusterer
    }

KMeansPlusPlusClusterer (esempio)
  │
  ├─ __init__(n_clusters, max_iters, tol)
  │
  ├─ fit(X)
  │   ├─ _init_centers_plus_plus(X)
  │   │   └─ Algoritmo KMeans++:
  │   │       1. Centro casuale
  │   │       2. Loop n_clusters-1:
  │   │          - Calcola distanze da centri esistenti
  │   │          - Scegli nuovo centro con prob ∝ dist²
  │   │
  │   └─ Loop max_iters:
  │       ├─ Assegna punti a cluster più vicino
  │       ├─ Ricalcola centri (media punti)
  │       ├─ Check convergenza (centri stabili)
  │       └─ Break se convergenza
  │
  └─ predict(X)
      └─ Assegna ogni punto al cluster più vicino
```

**Algoritmi Disponibili**:
1. **KMeans Standard**: Centri iniziali casuali
2. **KMeans++**: Centri iniziali intelligenti (k-means++)
3. **Sklearn KMeans**: Wrapper scikit-learn

**Design Pattern**: **Strategy Pattern + Factory Pattern**

---

### 4. **Utils Module** (`utils/`)

**Responsabilità**: Funzioni di utilità condivise

```python
from satellite_analysis.utils import geospatial, visualization

# Geospatial utilities
coords = geospatial.extract_coordinates(geometry)
area = geospatial.calculate_area(polygon)

# Visualization
visualization.plot_rgb(image, bands=[3,2,1])
visualization.plot_clusters(data, labels)
```

---

## Flusso di Esecuzione

### 🔄 Workflow Completo (Auth + Catalog Search)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LOAD CONFIGURATION                                       │
├─────────────────────────────────────────────────────────────┤
│ File: config/config.yaml                                    │
│ Class: Config                                               │
│                                                             │
│ config = Config.from_yaml("config/config.yaml")            │
│   ↓                                                         │
│ Loads:                                                      │
│   - sentinel.client_id                                      │
│   - sentinel.client_secret                                  │
│   - area.bbox                                               │
│   - sentinel.max_cloud_cover                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. OAUTH2 AUTHENTICATION                                    │
├─────────────────────────────────────────────────────────────┤
│ Class: OAuth2AuthStrategy                                   │
│                                                             │
│ auth = OAuth2AuthStrategy(client_id, client_secret)        │
│   ↓                                                         │
│ session = auth.get_session()                                │
│   ↓                                                         │
│ Internal Flow:                                              │
│   1. Check if session exists                                │
│   2. If not, call _authenticate()                           │
│      ├─ Create BackendApplicationClient                     │
│      ├─ Create OAuth2Session                                │
│      ├─ POST to token endpoint                              │
│      └─ Store token + session                               │
│   3. Return authenticated OAuth2Session                     │
│                                                             │
│ API Endpoint:                                               │
│ https://identity.dataspace.copernicus.eu/.../token         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CATALOG SEARCH                                           │
├─────────────────────────────────────────────────────────────┤
│ Class: SentinelHubCatalog                                   │
│                                                             │
│ catalog = SentinelHubCatalog(session)                       │
│   ↓                                                         │
│ results = catalog.search(                                   │
│     bbox=[9.0, 45.3, 9.3, 45.6],                           │
│     start_date="2023-03-01",                                │
│     end_date="2023-03-15",                                  │
│     cloud_cover_max=10.0,                                   │
│     limit=5                                                 │
│ )                                                           │
│   ↓                                                         │
│ Internal Flow:                                              │
│   1. Validate parameters                                    │
│      ├─ bbox: -180≤lon≤180, -90≤lat≤90                     │
│      ├─ dates: ISO format, start < end                      │
│      └─ cloud_cover: 0 ≤ value ≤ 100                        │
│   2. Build STAC query                                       │
│   3. POST to catalog API                                    │
│   4. Filter results by cloud cover (client-side)            │
│   5. Return features list                                   │
│                                                             │
│ API Endpoint:                                               │
│ https://sh.dataspace.copernicus.eu/api/v1/catalog/...     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. RESULTS                                                  │
├─────────────────────────────────────────────────────────────┤
│ {                                                           │
│   "features": [                                             │
│     {                                                       │
│       "properties": {                                       │
│         "datetime": "2023-03-12T10:28:21Z",                │
│         "eo:cloud_cover": 3.88,                            │
│         "platform": "sentinel-2a"                          │
│       },                                                    │
│       "geometry": {...},                                    │
│       "assets": {                                           │
│         "B02": {"href": "..."},  # Blue                    │
│         "B03": {"href": "..."},  # Green                   │
│         "B04": {"href": "..."},  # Red                     │
│         "B08": {"href": "..."}   # NIR                     │
│       }                                                     │
│     }                                                       │
│   ]                                                         │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Design Patterns

### 1. **Strategy Pattern**

**Usato in**: Authentication, Catalog, Clustering

```python
# Abstract Strategy
class AuthStrategy(ABC):
    @abstractmethod
    def get_session(self): pass
    
    @abstractmethod
    def is_valid(self): pass

# Concrete Strategies
class OAuth2AuthStrategy(AuthStrategy):
    def get_session(self): ...
    def is_valid(self): ...

class UsernamePasswordAuth(AuthStrategy):  # Future
    def get_session(self): ...
    def is_valid(self): ...
```

**Vantaggi**:
- Facile aggiungere nuove strategie
- Switching tra strategie a runtime
- Testing isolato di ogni strategia

---

### 2. **Factory Pattern**

**Usato in**: Clustering algorithms

```python
class ClusteringFactory:
    _strategies = {
        'kmeans': KMeansClusterer,
        'kmeans++': KMeansPlusPlusClusterer,
        'sklearn': SklearnKMeansClusterer
    }
    
    @classmethod
    def create(cls, strategy: str, **kwargs):
        return cls._strategies[strategy](**kwargs)
```

**Vantaggi**:
- Centralizzazione creazione oggetti
- Facile registrare nuovi algoritmi
- Client code disaccoppiato

---

### 3. **Configuration Object Pattern**

**Usato in**: Config module

```python
@dataclass
class SentinelConfig:
    client_id: Optional[str]
    client_secret: Optional[str]
    platformname: str
    max_cloud_cover: float
```

**Vantaggi**:
- Type safety
- Validazione centralizzata
- Facile serializzazione/deserializzazione

---

## Diagrammi di Sequenza

### Sequence: Complete Auth + Search Flow

```
User          Config       OAuth2Auth    Token API    Catalog      STAC API
 │               │              │            │           │            │
 ├─from_yaml()──>│              │            │           │            │
 │<─────config───┤              │            │           │            │
 │               │              │            │           │            │
 ├─OAuth2Auth(id,secret)───────>│            │           │            │
 │               │              │            │           │            │
 ├─get_session()────────────────>│            │           │            │
 │               │              ├─POST token─>│           │            │
 │               │              │<──token─────┤           │            │
 │<─────session──────────────────┤            │           │            │
 │               │              │            │           │            │
 ├─Catalog(session)─────────────────────────────────────>│            │
 │               │              │            │           │            │
 ├─search(bbox,dates)───────────────────────────────────>│            │
 │               │              │            │           ├─POST query─>│
 │               │              │            │           │<──results───┤
 │               │              │            │           ├─filter──────┤
 │<─────results────────────────────────────────────────────────────────┤
```

---

## 📍 Entry Points

### 1. Test Script
```powershell
.venv\Scripts\python.exe test_sentinel_download.py
```

### 2. Interactive Notebook
```powershell
jupyter notebook notebooks/download_example.ipynb
```

### 3. Custom Python Script
```python
from satellite_analysis.config import Config
from satellite_analysis.downloaders.auth import OAuth2AuthStrategy
from satellite_analysis.downloaders.catalog import SentinelHubCatalog

# Your code here...
```

---

## 🔜 Prossimi Componenti da Implementare

1. **Downloader**: Download effettivo dei prodotti
2. **Preprocessor**: Estrazione e processing bande
3. **Pipeline**: Orchestrazione end-to-end
4. **Classifier**: Random Forest / SVM

---

## 📚 Riferimenti

- **Sentinel Hub API**: https://documentation.dataspace.copernicus.eu/
- **STAC Spec**: https://github.com/radiantearth/stac-spec
- **OAuth2**: https://oauth.net/2/
- **Design Patterns**: Gang of Four (GoF)
