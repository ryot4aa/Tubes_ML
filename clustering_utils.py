import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, SpectralClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / 'dataset'
MODELS_DIR = REPO_ROOT / 'models'
N_CLUSTERS = 3
SAMPLE_SIZE = 3000

FEATURE_COLUMNS_BY_DATASET = {
    'creditcard.csv': ['V1', 'V2', 'Amount'],
    'credit_card_fraud_10k.csv': ['amount', 'transaction_hour', 'velocity_last_24h'],
    'Synthetic_Financial_datasets_log.csv': ['amount', 'oldbalanceOrg', 'newbalanceOrig'],
    'fraudTrain.csv': ['amt', 'city_pop', 'unix_time'],
    'transactions.csv': ['amount', 'account_age_days', 'avg_amount_user']
}


def model_filename(dataset_stem: str, model_key: str) -> Path:
    """Return model path for a given dataset stem and model key."""
    safe_stem = dataset_stem.replace(' ', '_')
    key = model_key.lower().replace(' ', '_')
    return MODELS_DIR / f"{safe_stem}__{key}.pkl"


def ensure_models_directory():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_pickle(target_path: Path, obj):
    ensure_models_directory()
    with target_path.open('wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(target_path: Path):
    with target_path.open('rb') as handle:
        return pickle.load(handle)


def list_datasets():
    """Return list of CSV dataset filenames present in the dataset directory."""
    if not DATASET_DIR.exists():
        return []
    return [p.name for p in sorted(DATASET_DIR.glob('*.csv'))]


def load_dataset(filename: str, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    path = DATASET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset file tidak ditemukan: {path}")
    df = pd.read_csv(path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df.reset_index(drop=True)


def select_features_for_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    filename = Path(dataset_name).name
    if filename in FEATURE_COLUMNS_BY_DATASET:
        feature_columns = FEATURE_COLUMNS_BY_DATASET[filename]
        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Kolom fitur diperlukan tidak ditemukan untuk {filename}: {missing_cols}")
        return df[feature_columns].copy()

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    excluded = {'isFraud', 'is_fraud', 'Class', 'transaction_id', 'user_id', 'cc_num', 'step', 'unix_time'}
    feature_columns = [c for c in numeric_cols if c not in excluded]
    if len(feature_columns) < 3:
        raise ValueError(f"Tidak cukup fitur numerik untuk dataset {filename}: {feature_columns}")
    return df[feature_columns[:3]].copy()


def extract_features(df: pd.DataFrame, dataset_name: str = 'creditcard.csv') -> pd.DataFrame:
    return select_features_for_dataset(df, dataset_name)


def preprocess_features(feature_df: pd.DataFrame, scaler: Optional[StandardScaler] = None):
    imputer = SimpleImputer(strategy='mean')
    values = imputer.fit_transform(feature_df)

    if scaler is None:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)
    else:
        values = scaler.transform(values)

    return values, scaler


def get_feature_vector(x1: float, x2: float, amount: float, scaler: StandardScaler):
    values = np.array([[x1, x2, amount]], dtype=float)
    values = SimpleImputer(strategy='mean').fit_transform(values)
    return scaler.transform(values)


def fit_clustering_models(X: np.ndarray):
    models = {
        'K-Means': KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=5),
        'BIRCH': Birch(n_clusters=N_CLUSTERS),
        'Agglomerative': AgglomerativeClustering(n_clusters=N_CLUSTERS),
        'Spectral': SpectralClustering(
            n_clusters=N_CLUSTERS,
            random_state=42,
            assign_labels='discretize',
            n_neighbors=10
        )
    }

    fitted_models = {}
    assigners = {}
    metrics = {}

    for name, model in models.items():
        start_time = time.perf_counter()
        labels = model.fit_predict(X)
        duration = time.perf_counter() - start_time

        silhouette = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)

        fitted_models[name] = model
        metrics[name] = {
            'silhouette': float(np.round(silhouette, 4)),
            'davies_bouldin': float(np.round(dbi, 4)),
            'train_seconds': float(np.round(duration, 4))
        }

        if name in ('Agglomerative', 'Spectral'):
            assigner = NearestCentroid()
            assigner.fit(X, labels)
            assigners[name] = assigner

    return fitted_models, assigners, metrics


def assign_cluster_for_sample(model_name: str, models: dict, assigners: dict, sample: np.ndarray):
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' tidak ditemukan")

    if hasattr(model, 'predict'):
        return int(model.predict(sample)[0])

    assigner = assigners.get(model_name)
    if assigner is not None:
        return int(assigner.predict(sample)[0])

    raise ValueError(f"Tidak ada metode prediksi untuk model '{model_name}'")


def get_labels_for_model(model_name: str, models: dict, assigners: dict, X: np.ndarray):
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' tidak ditemukan")

    if hasattr(model, 'predict'):
        return model.predict(X)

    assigner = assigners.get(model_name)
    if assigner is not None:
        return assigner.predict(X)

    raise ValueError(f"Tidak ada metode prediksi untuk model '{model_name}'")


def save_clustering_artifacts(models: dict, scaler: StandardScaler, assigners: dict, dataset_name: str):
    ensure_models_directory()
    stem = Path(dataset_name).stem
    for name, model in models.items():
        save_pickle(model_filename(stem, name), model)

    save_pickle(model_filename(stem, 'scaler'), scaler)
    save_pickle(model_filename(stem, 'assigners'), assigners)


def load_saved_models_for_dataset(dataset_name: str):
    stem = Path(dataset_name).stem
    if not MODELS_DIR.exists():
        return None
    try:
        models = {
            'K-Means': load_pickle(model_filename(stem, 'k-means')),
            'BIRCH': load_pickle(model_filename(stem, 'birch')),
            'Agglomerative': load_pickle(model_filename(stem, 'agglomerative')),
            'Spectral': load_pickle(model_filename(stem, 'spectral'))
        }
        scaler = load_pickle(model_filename(stem, 'scaler'))
        assigners = load_pickle(model_filename(stem, 'assigners'))
        return models, scaler, assigners
    except FileNotFoundError:
        return None


def train_models_for_dataset(dataset_name: str):
    df = load_dataset(dataset_name)
    feature_df = extract_features(df, dataset_name)
    X_scaled, scaler = preprocess_features(feature_df)

    models, assigners, metrics = fit_clustering_models(X_scaled)
    save_clustering_artifacts(models, scaler, assigners, dataset_name)
    return models, scaler, assigners, metrics, X_scaled


def train_all_datasets():
    results = {}
    for fname in list_datasets():
        try:
            print(f"Training models for {fname} ...")
            models, scaler, assigners, metrics, X_scaled = train_models_for_dataset(fname)
            results[fname] = {
                'models': models,
                'scaler': scaler,
                'assigners': assigners,
                'metrics': metrics,
                'X': X_scaled
            }
        except Exception as e:
            print(f"Failed training {fname}: {e}")
    return results


def load_or_train_models(dataset_name: str = 'creditcard.csv'):
    """
    Load saved models for a dataset. DOES NOT train automatically.
    Returns tuple (models, scaler, assigners, metrics, X_scaled) if models exist, otherwise returns None.
    """
    loaded = load_saved_models_for_dataset(dataset_name)
    if loaded is None:
        return None

    models, scaler, assigners = loaded
    df = load_dataset(dataset_name)
    X_scaled, _ = preprocess_features(extract_features(df, dataset_name), scaler=scaler)
    _, _, metrics = fit_clustering_models(X_scaled)
    return models, scaler, assigners, metrics, X_scaled
def summarize_clusters(X: np.ndarray, labels: np.ndarray):
    df = pd.DataFrame(X, columns=['X1', 'X2', 'Amount'])
    df['cluster'] = labels
    summary = df.groupby('cluster').agg(
        mean_X1=('X1', 'mean'),
        mean_X2=('X2', 'mean'),
        mean_Amount=('Amount', 'mean'),
        count=('cluster', 'count')
    ).reset_index()

    ordered = summary.sort_values('mean_Amount').reset_index(drop=True)
    descriptions = {}
    for idx, row in ordered.iterrows():
        if idx == 0:
            descriptions[row['cluster']] = 'Normal / Transaksi wajar'
        elif idx == len(ordered) - 1:
            descriptions[row['cluster']] = 'Risiko tinggi / Potensi fraud'
        else:
            descriptions[row['cluster']] = 'Mencurigakan / Perlu pengawasan'

    ordered['description'] = ordered['cluster'].map(descriptions)
    return ordered, descriptions


def format_cluster_description(cluster: int, descriptions: dict, model_name: str):
    text = descriptions.get(cluster, 'Tidak ada deskripsi cluster tersedia.')
    return (
        f"Klaster {cluster} menggunakan model {model_name}: {text}. "
        "Gunakan informasi ini bersama nilai silhouette dan pola fitur untuk menilai risiko transaksi."
    )
