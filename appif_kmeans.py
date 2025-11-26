# appIF-KMeans.py
# Streamlit app untuk Deteksi Anomali Trafik Jaringan
# Menggunakan Isolation Forest + K-Means
# Siap dideploy ke Streamlit Cloud

import io
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Anomaly Detection (IF + KMeans)", layout="wide")
st.title("Deteksi Anomali Trafik Jaringan — Isolation Forest & K-Means")
st.markdown("Aplikasi demo untuk skripsi: Penerapan Isolation Forest & K-Means pada trafik jaringan PT. XYZ")

# ----------------------
# Sidebar: pengaturan
# ----------------------
st.sidebar.header("Pengaturan Model")
contamination = st.sidebar.slider(
    "Contamination (Isolation Forest)", min_value=0.001, max_value=0.2, value=0.05, step=0.001
)
n_clusters = st.sidebar.slider("Jumlah cluster (K-Means)", min_value=2, max_value=8, value=3, step=1)
random_state = st.sidebar.number_input("Random State", value=42, step=1)
small_cluster_threshold = st.sidebar.slider(
    "Ambang cluster kecil (persentase)", min_value=0.1, max_value=10.0, value=1.0, step=0.1
)
st.sidebar.markdown("---")
st.sidebar.markdown("Upload dataset CSV atau gunakan dataset dummy untuk demo.")

# ----------------------
# Upload file atau dummy
# ----------------------
upload_file = st.file_uploader("Upload file CSV (kolom numerik diperlukan)", type=["csv"])
df = None

if upload_file is not None:
    # Cek ukuran & isi file
    try:
        if upload_file.size == 0:
            st.error("File CSV kosong.")
            st.stop()
    except Exception:
        pass

    # Baca isi (aman)
    try:
        # coba baca langsung (streamlit InMemoryUploadedFile)
        df = pd.read_csv(upload_file)
    except Exception:
        try:
            upload_file.seek(0)
            content = upload_file.read().decode(errors="ignore")
            # baca dari string
            df = pd.read_csv(io.StringIO(content))
        except Exception as e:
            st.error(f"Gagal membaca file CSV: {e}")
            st.stop()

    st.success(f"File berhasil diupload — shape: {df.shape}")

else:
    if st.button("Gunakan dataset dummy"):
        # buat dataset dummy di memory (untuk demo)
        np.random.seed(42)
        n_normal = 900
        n_anom = 100
        normal = pd.DataFrame({
            "packet_size": np.random.normal(500, 50, n_normal).clip(40),
            "duration": np.random.exponential(0.5, n_normal),
            "bytes_in": np.random.normal(1000, 200, n_normal).clip(0),
            "bytes_out": np.random.normal(950, 180, n_normal).clip(0),
            "src_port": np.random.randint(1024, 50000, n_normal),
            "dst_port": np.random.choice([80,443,22,53], n_normal, p=[0.4,0.4,0.1,0.1])
        })
        anom = pd.DataFrame({
            "packet_size": np.random.normal(1500, 300, n_anom).clip(40),
            "duration": np.random.exponential(3, n_anom),
            "bytes_in": np.random.normal(5000, 1000, n_anom).clip(0),
            "bytes_out": np.random.normal(4500, 1000, n_anom).clip(0),
            "src_port": np.random.randint(1024, 50000, n_anom),
            "dst_port": np.random.choice([22,23,80,8080], n_anom, p=[0.2,0.1,0.4,0.3])
        })
        df = pd.concat([normal, anom], ignore_index=True).sample(frac=1).reset_index(drop=True)
        st.success(f"Menggunakan dataset dummy — shape: {df.shape}")
    else:
        st.info("Belum ada dataset. Upload CSV atau klik 'Gunakan dataset dummy'.")
        st.stop()

# ----------------------
# Validasi & preprocessing
# ----------------------
# Tampilkan preview awal
st.subheader("Preview Data")
st.dataframe(df.head())

# Pilih kolom numerik
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("Dataset tidak memiliki kolom numerik. Pastikan CSV berisi fitur numerik.")
    st.stop()

st.markdown("**Fitur numerik yang digunakan:** " + ", ".join(numeric_cols))

# Ambil salinan kolom numerik
X = df[numeric_cols].copy()

# Bersihkan nilai bermasalah: inf, -inf -> NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Bersihkan nilai yang terlalu besar (overflow) menjadi NaN
def _safe_overflow_to_nan(x):
    try:
        if isinstance(x, (int, float)) and (abs(x) > 1e308):
            return np.nan
        return x
    except Exception:
        return np.nan

X = X.applymap(_safe_overflow_to_nan)

# Drop baris yang ada NaN
before_rows = len(X)
X = X.dropna()
after_rows = len(X)
dropped = before_rows - after_rows

if dropped > 0:
    st.warning(f"{dropped} baris dihapus karena mengandung NaN/inf/overflow. Baris tersisa: {after_rows}")

# Sinkronkan df dengan X (hapus baris yang sama)
df = df.loc[X.index].reset_index(drop=True)
X = X.reset_index(drop=True)

if X.shape[0] == 0:
    st.error("Semua baris tidak valid setelah pembersihan. Periksa dataset.")
    st.stop()

# Scaling
scaler = StandardScaler()
try:
    X_scaled = scaler.fit_transform(X)
except Exception as e:
    st.error(f"Gagal melakukan scaling: {e}")
    st.stop()

# ----------------------
# Model: Isolation Forest
# ----------------------
iso = IsolationForest(contamination=contamination, random_state=int(random_state))
iso_labels = iso.fit_predict(X_scaled)  # -1 anomaly, 1 normal
iso_scores = iso.decision_function(X_scaled)  # higher -> more normal

df["anomaly_if"] = np.where(iso_labels == -1, 1, 0)
df["anomaly_score_if"] = -iso_scores  # inverted so higher = more anomalous

# ----------------------
# Model: K-Means clustering
# ----------------------
kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(random_state))
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

cluster_counts = df["cluster"].value_counts().sort_index()
threshold_count = max(1, int(len(df) * (small_cluster_threshold / 100.0)))
small_clusters = cluster_counts[cluster_counts <= threshold_count].index.tolist()
df["anomaly_kmeans_cluster"] = df["cluster"].apply(lambda c: 1 if c in small_clusters else 0)

# Gabungan hasil
df["anomaly_any"] = df[["anomaly_if", "anomaly_kmeans_cluster"]].max(axis=1)

# ----------------------
# Ringkasan hasil
# ----------------------
st.subheader("Ringkasan Hasil Deteksi")
col1, col2, col3 = st.columns(3)
col1.metric("Total baris", len(df))
col2.metric("Jumlah flagged (IF)", int(df["anomaly_if"].sum()))
col3.metric("Jumlah flagged (KMeans small cluster)", int(df["anomaly_kmeans_cluster"].sum()))
st.markdown("**Total flagged (gabungan)**: {}".format(int(df["anomaly_any"].sum())))

# ----------------------
# Visualisasi PCA 2D
# ----------------------
st.subheader("Visualisasi (PCA 2D)")
pca = PCA(n_components=2, random_state=int(random_state))
proj = pca.fit_transform(X_scaled)
df["pca1"] = proj[:, 0]
df["pca2"] = proj[:, 1]

fig, ax = plt.subplots(figsize=(8, 5))
for c in sorted(df["cluster"].unique()):
    sel = df[df["cluster"] == c]
    ax.scatter(sel["pca1"], sel["pca2"], label=f"Cluster {c}", alpha=0.6, s=20)

sel_anom = df[df["anomaly_any"] == 1]
if len(sel_anom) > 0:
    ax.scatter(sel_anom["pca1"], sel_anom["pca2"], facecolors="none", edgecolors="k", s=80, label="Flagged Anomaly (any)")

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend(markerscale=1)
ax.grid(True)
st.pyplot(fig)

# ----------------------
# Tabel & Download
# ----------------------
st.subheader("Tabel Data yang Dideteksi Anomali")
show_limit = st.number_input("Max rows ditampilkan", min_value=5, max_value=1000, value=50, step=5)
anom_df = df[df["anomaly_any"] == 1].sort_values("anomaly_score_if", ascending=False)
st.write(f"Menampilkan {min(len(anom_df), show_limit)} dari {len(anom_df)} anomali")
st.dataframe(anom_df.head(int(show_limit)))

csv = df.to_csv(index=False)
st.download_button("Download CSV Hasil Deteksi", csv, file_name="hasil_deteksi.csv")

# ----------------------
# Penjelasan
# ----------------------
st.markdown(
    """
**Catatan & Penjelasan**:

- `anomaly_if`: hasil Isolation Forest (1 = anomaly).
- `anomaly_score_if`: skor Isolation Forest (lebih tinggi = lebih anomali).
- `cluster`: label cluster hasil K-Means.
- `anomaly_kmeans_cluster`: pendeteksian cluster kecil sebagai indikasi outlier.
- `anomaly_any`: gabungan hasil deteksi Isolation Forest dan K-Means.
"""
)
