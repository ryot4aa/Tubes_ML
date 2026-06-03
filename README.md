<img width="216" height="459" alt="struktur_code" src="https://github.com/user-attachments/assets/692f48b5-8a10-4ba7-a188-790ace073400" />

Struktur code harus sama

link dataset dan models : https://drive.google.com/drive/folders/1VxU7S33FZF-ER1FqRBS-tCMv0LCWvt7Q?usp=sharing

env_tugas_besar tidak perlu ada

nuhun

---

## Cara menjalankan

1. Pastikan sudah berada di lingkungan Python yang benar.
2. Jalankan pelatihan model clustering sekali saja:
   ```bash
   python train_clustering_models.py
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app/app.py
   ```

Folder `models/` akan berisi model nyata untuk K-Means, BIRCH, Agglomerative, dan Spectral. Mean Imputation digunakan sebagai preprocessing sebelum clustering.
