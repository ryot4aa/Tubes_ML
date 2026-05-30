from clustering_utils import train_all_datasets, list_datasets


if __name__ == '__main__':
    datasets = list_datasets()
    if not datasets:
        print('Tidak ada dataset .csv di folder dataset/.')
    else:
        print(f'Mulai melatih model clustering pada {len(datasets)} dataset...')
        results = train_all_datasets()
        print('\nPelatihan selesai. Ringkasan metrik per dataset:')
        for fname, res in results.items():
            print(f"\n-- {fname} --")
            for name, score in res['metrics'].items():
                print(f"- {name}: Silhouette={score['silhouette']}, Davies-Bouldin={score['davies_bouldin']}, waktu={score['train_seconds']} detik")
