import sys
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title='Dashboard Fraud Clustering',
    page_icon='⚠️',
    layout='wide',
    initial_sidebar_state='expanded'
)

from clustering_utils import (
    assign_cluster_for_sample,
    extract_features,
    format_cluster_description,
    get_feature_vector,
    get_labels_for_model,
    load_or_train_models,
    load_dataset,
    list_datasets,
    summarize_clusters,
)

warnings.filterwarnings('ignore')
DATASET_NAME = 'creditcard.csv'
DATASET_PATH = BASE_DIR / 'dataset' / DATASET_NAME
MODEL_NAME_MAPPING = {
    'K-Means Clustering': 'K-Means',
    'BIRCH (Balanced Iterative Reducing Clustering)': 'BIRCH',
    'Agglomerative Clustering': 'Agglomerative',
    'Spectral Clustering': 'Spectral'
}

MODEL_DESCRIPTIONS = {
    'K-Means': 'K-Means membagi data menjadi klaster berdasarkan jarak ke centroid. Cocok untuk pola transaksi yang terdistribusi secara partisi.',
    'BIRCH': 'BIRCH membangun struktur hirarki ringkas untuk clustering skala besar dan dapat menangani noise dengan baik.',
    'Agglomerative': 'Agglomerative Clustering adalah metode hierarkis yang menggabungkan klaster berdasarkan kemiripan bertahap.',
    'Spectral': 'Spectral Clustering menggunakan representasi graf untuk menemukan pola klaster non-linier dan kompleks.'
}


def format_metrics_table(metrics):
    rows = []
    best_model = max(metrics, key=lambda name: metrics[name]['silhouette'])
    for name, values in metrics.items():
        label = 'Paling optimal' if name == best_model else 'Bagus' if values['silhouette'] >= 0.5 else 'Perlu analisis lebih lanjut'
        rows.append({
            'Algoritma': name,
            'Silhouette Score': values['silhouette'],
            'Davies-Bouldin': values['davies_bouldin'],
            'Keterangan': label
        })
    return pd.DataFrame(rows)


def render_detection_page(models, scaler, assigners, metrics, X_train, dataset_name):
    st.title('Sistem Pemetaan & Deteksi Anomali Transaksi')
    st.write(f'Aplikasi ini menggunakan preprocessing Mean Imputation + 4 algoritma clustering untuk mendeteksi pola transaksi pada dataset {dataset_name}.')

    st.markdown('''
    **Clustering apa?**
    - Model membagi transaksi menjadi 3 klaster berdasarkan fitur V1, V2, dan Amount.
    - Tujuan: memisahkan pola normal, mencurigakan, dan risiko tinggi.
    - Model yang tersedia: K-Means, BIRCH, Agglomerative, Spectral.
    ''')

    raw_df = load_dataset(dataset_name)
    feature_df = extract_features(raw_df, dataset_name)
    feature_means = feature_df.mean()
    feature_cols = list(feature_df.columns)
    f1_name, f2_name, f3_name = feature_cols[0], feature_cols[1], feature_cols[2]

    with st.expander('📦 Preview Data dan Nilai Mean Imputation'):
        st.write(f'Dataset yang digunakan: {dataset_name}. Sistem mengambil sample hingga 3.000 baris untuk pelatihan model.')
        st.dataframe(feature_df.head(8))
        st.write('Nilai rata-rata yang digunakan untuk Mean Imputation:')
        st.dataframe(feature_means.to_frame('Mean').T)

    st.subheader('📝 Input Transaksi Baru')
    col1, col2, col3 = st.columns(3)
    with col1:
        x1_input = st.text_input(f'Karakteristik {f1_name} (kosongkan untuk mean imputation):', value='')
    with col2:
        x2_input = st.text_input(f'Karakteristik {f2_name} (kosongkan untuk mean imputation):', value='')
    with col3:
        amount_input = st.text_input(f'Karakteristik {f3_name} (kosongkan untuk mean imputation):', value='')

    model_choice = st.selectbox('Pilih Model Algoritma Clustering:', list(MODEL_NAME_MAPPING.keys()))

    if st.button('Jalankan Deteksi Transaksi'):
        try:
            x1 = float(x1_input) if x1_input.strip() != '' else float(feature_means[f1_name])
            x2 = float(x2_input) if x2_input.strip() != '' else float(feature_means[f2_name])
            amount = float(amount_input) if amount_input.strip() != '' else float(feature_means[f3_name])
        except ValueError:
            st.error('Input tidak valid. Pastikan semua nilai angka benar atau biarkan kosong untuk mean imputation.')
            return

        if x1_input.strip() == '':
            st.warning(f'{f1_name} kosong, menggunakan Mean Imputation = {feature_means[f1_name]:.4f}')
        if x2_input.strip() == '':
            st.warning(f'{f2_name} kosong, menggunakan Mean Imputation = {feature_means[f2_name]:.4f}')
        if amount_input.strip() == '':
            st.warning(f'{f3_name} kosong, menggunakan Mean Imputation = {feature_means[f3_name]:.4f}')

        sample = get_feature_vector(x1, x2, amount, scaler)
        selected_model = MODEL_NAME_MAPPING[model_choice]
        cluster_id = assign_cluster_for_sample(selected_model, models, assigners, sample)

        labels = get_labels_for_model(selected_model, models, assigners, X_train)
        summary, descriptions = summarize_clusters(X_train, labels)
        description = format_cluster_description(cluster_id, descriptions, selected_model)

        st.write('---')
        st.success(f'Transaksi ini masuk ke dalam **Klaster {cluster_id}** menggunakan model **{selected_model}**.')
        st.info(description)
        st.write('**Metrik model:**')
        st.write(metrics[selected_model])
        st.write('**Penjelasan algoritma:**')
        st.write(MODEL_DESCRIPTIONS[selected_model])

        if st.checkbox('Tampilkan ringkasan cluster dataset untuk model ini'):
            st.write('Ringkasan cluster dari data pelatihan:')
            rename_map = {
                'cluster': 'Klaster',
                'mean_X1': f'Rata-rata {f1_name}',
                'mean_X2': f'Rata-rata {f2_name}',
                'mean_Amount': f'Rata-rata {f3_name}',
                'count': 'Jumlah Transaksi',
                'description': 'Deskripsi Klaster'
            }
            st.dataframe(summary.rename(columns=rename_map))
            st.bar_chart(summary.set_index('cluster')['count'])


def render_evaluation_page(metrics, dataset_name):
    st.title('📊 Analisis Komparatif Evaluasi Performa Model')
    st.write(f'Halaman ini menampilkan metrik nyata dari keempat model clustering yang dilatih pada dataset {dataset_name}.')

    df_metrics = format_metrics_table(metrics)
    st.table(df_metrics)

    st.markdown(
        '- **Silhouette Score**: semakin tinggi semakin baik pemisahan antar klaster.\n'
        '- **Davies-Bouldin**: semakin rendah semakin baik kualitas klaster.\n'
        '- **Mean Imputation**: ini adalah teknik preprocessing untuk mengisi nilai hilang, bukan algoritma clustering.\n'
    )
    st.info('Model terbaik berdasarkan silhouette score adalah **{}**.'.format(
        df_metrics.loc[df_metrics['Silhouette Score'].idxmax(), 'Algoritma']
    ))


def main():
    st.sidebar.title('Dashboard')
    if 'show_training_modal' not in st.session_state:
        st.session_state.show_training_modal = True

    # Show the training instruction modal/banner once on first load
    if st.session_state.get('show_training_modal'):
        # Prefer native modal when available, otherwise show an inline banner with a close button
        try:
            modal_ctx = st.modal
        except Exception:
            modal_ctx = None

        if modal_ctx is not None:
            try:
                with st.modal('Instruksi Pelatihan Model'):
                    st.write('Pastikan model sudah dilatih terlebih dahulu sebelum menggunakan aplikasi ini.')
                    st.code('python train_clustering_models.py')
                    st.write('Jalankan perintah ini di terminal proyek untuk membuat file model di folder `models/`.')
                    if st.button('Tutup', key='close_training_modal'):
                        st.session_state.show_training_modal = False
            except Exception:
                # Fallback to banner if modal fails at call time
                st.info('Pastikan model sudah dilatih terlebih dahulu sebelum menggunakan aplikasi ini.')
                st.code('python train_clustering_models.py')
                if st.button('Tutup', key='close_training_modal'):
                    st.session_state.show_training_modal = False
        else:
            st.info('Pastikan model sudah dilatih terlebih dahulu sebelum menggunakan aplikasi ini.')
            st.code('python train_clustering_models.py')
            st.write('Jalankan perintah ini di terminal proyek untuk membuat file model di folder `models/`.')
            if st.button('Tutup', key='close_training_modal'):
                st.session_state.show_training_modal = False
    datasets = list_datasets()
    selected_dataset = st.sidebar.selectbox('Pilih Dataset:', datasets, index=0 if datasets else None)
    menu = st.sidebar.selectbox('Pilih Halaman:', ['Sistem Deteksi Terpadu', 'Analisis Komparatif Evaluasi'])

    with st.spinner('Memuat model clustering...'):
        if selected_dataset:
                loaded = load_or_train_models(selected_dataset)
                if loaded is None:
                    st.error(f"Model untuk dataset '{selected_dataset}' belum ditemukan.")
                    st.warning('Silakan jalankan `python train_clustering_models.py` di terminal proyek untuk membuat model terlebih dahulu.')
                    st.stop()
                models, scaler, assigners, metrics, X_train = loaded
        else:
            st.error('Tidak ada dataset tersedia di folder dataset/.')
            return

    st.markdown(
        """
        <style>
            .appview-container .main .block-container {
                max-width: 1400px;
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            .css-1v0mbdj.e1fqkh3o4 {
                max-width: 1400px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if menu == 'Sistem Deteksi Terpadu':
        render_detection_page(models, scaler, assigners, metrics, X_train, selected_dataset)
    else:
        render_evaluation_page(metrics, selected_dataset)


if __name__ == '__main__':
    main()
