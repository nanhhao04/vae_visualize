# ===========================================================================
# Đọc 3 file cluster0.txt, cluster1.txt, cluster2.txt
# Chạy K-Means (cosine distance)
# Visualize Ground Truth vs K-Means
# Tính các metrics
# ===========================================================================

import ast
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# 0. Load Configuration
# ---------------------------------------------------------------------------
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
data_name = config.get("data_name", "cifar10")
print(f"Using dataset: {data_name}")

# ---------------------------------------------------------------------------
# 1. Đọc dữ liệu từ file txt trong thư mục tương ứng
# ---------------------------------------------------------------------------

def load_cluster_file(filename):
    """
    Mỗi file chứa nhiều dòng dạng:
    [0.1, 0.2, ..., 0.0],
    """
    with open(filename, "r") as f:
        content = f.read().strip()

    # Nếu file kết thúc bằng dấu phẩy, bỏ đi
    if content.endswith(","):
        content = content[:-1]

    # Bọc trong [] để parse thành list lớn
    data = ast.literal_eval("[" + content + "]")
    return np.array(data, dtype=np.float32)


# Load 3 cụm từ thư mục của data_name
X0 = load_cluster_file(os.path.join(data_name, "cluster0.txt"))   # ground truth label = 0
X1 = load_cluster_file(os.path.join(data_name, "cluster1.txt"))   # ground truth label = 1
X2 = load_cluster_file(os.path.join(data_name, "cluster2.txt"))   # ground truth label = 2

# Ghép dữ liệu
X = np.vstack([X0, X1, X2])
y = np.concatenate([
    np.zeros(len(X0), dtype=int),
    np.ones(len(X1), dtype=int),
    np.full(len(X2), 2, dtype=int)
])

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Samples per cluster:", [len(X0), len(X1), len(X2)])

# Lưu lại nếu muốn dùng sau
np.save("X_loaded.npy", X)
np.save("y_loaded.npy", y)

# ---------------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------------

def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-10, None)

def clustering_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)

    for pred, true in zip(y_pred, y_true):
        w[pred, true] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / len(y_true)

# ---------------------------------------------------------------------------
# 3. K-Means với cosine distance
# ---------------------------------------------------------------------------

X_norm = l2_normalize(X)

N_CLUSTERS = 3
N_RUNS = 20

nmi_list = []
ari_list = []
acc_list = []
sil_list = []
dbi_list = []
chi_list = []

for seed in range(N_RUNS):
    km = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=seed,
        n_init=10
    )

    labels = km.fit_predict(X_norm)

    nmi_list.append(
        normalized_mutual_info_score(
            y, labels, average_method="arithmetic"
        )
    )
    ari_list.append(adjusted_rand_score(y, labels))
    acc_list.append(clustering_accuracy(y, labels))
    sil_list.append(silhouette_score(X_norm, labels))
    dbi_list.append(davies_bouldin_score(X_norm, labels))
    chi_list.append(calinski_harabasz_score(X_norm, labels))

results = {
    "NMI":        (nmi_list, "high_better"),
    "ARI":        (ari_list, "high_better"),
    "ACC":        (acc_list, "high_better"),
    "Silhouette": (sil_list, "high_better"),
    "DBI":        (dbi_list, "low_better"),
    "CHI":        (chi_list, "high_better"),
}

# ---------------------------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------------------------

print(f"\nK-Means (cosine distance, K={N_CLUSTERS}, {N_RUNS} runs)\n")
print(f"{'Metric':<14} {'Mean':>10} {'Std':>10}  Direction")
print("-" * 50)

for name, (vals, direction) in results.items():
    print(
        f"{name:<14}"
        f"{np.mean(vals):>10.4f}"
        f"{np.std(vals):>10.4f}  "
        f"{direction}"
    )

# ---------------------------------------------------------------------------
# 5. Chạy lại 1 lần để visualize (seed = 0)
# ---------------------------------------------------------------------------

km_vis = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=0,
    n_init=10
)

labels_vis = km_vis.fit_predict(X_norm)

# PCA 2D
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_norm)

var_explained = pca.explained_variance_ratio_.sum() * 100

# ---------------------------------------------------------------------------
# 6. Visualize
# ---------------------------------------------------------------------------

COLORS = ["#1D9E75", "#7F77DD", "#D85A30"]
MARKERS = ["o", "s", "^"]

fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(
    1, 3,
    width_ratios=[1, 1, 1.2],
    wspace=0.35
)

# ---------------- Ground Truth ----------------
ax1 = fig.add_subplot(gs[0])

for k in range(3):
    mask = (y == k)
    ax1.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        c=COLORS[k],
        marker=MARKERS[k],
        s=18,
        alpha=0.6,
        label=f"Cụm {k}"
    )

ax1.set_title("Ground Truth", fontsize=11, fontweight="bold")
ax1.set_xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
    fontsize=9
)
ax1.set_ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
    fontsize=9
)
ax1.legend(fontsize=8)
ax1.tick_params(labelsize=8)

# ---------------- K-Means Prediction ----------------
ax2 = fig.add_subplot(gs[1])

for k in range(3):
    mask = (labels_vis == k)
    ax2.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        c=COLORS[k],
        marker=MARKERS[k],
        s=18,
        alpha=0.6,
        label=f"Pred {k}"
    )

ax2.set_title("K-Means (cosine, seed=0)", fontsize=11, fontweight="bold")
ax2.set_xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
    fontsize=9
)
ax2.set_ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
    fontsize=9
)
ax2.legend(fontsize=8)
ax2.tick_params(labelsize=8)

# ---------------- Metrics Table ----------------
ax3 = fig.add_subplot(gs[2])
ax3.axis("off")

rows = []
for name, (vals, direction) in results.items():
    rows.append([
        name,
        f"{np.mean(vals):.4f}",
        f"{np.std(vals):.4f}",
        direction
    ])

col_labels = ["Metric", "Mean", "Std", "Hướng"]

tbl = ax3.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

# Header style
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#3C3489")
    tbl[0, j].set_text_props(
        color="white",
        fontweight="bold"
    )

# Alternate row colors
for i in range(1, len(rows) + 1):
    bg = "#F1EFE8" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(bg)

ax3.set_title(
    f"K-Means — {N_RUNS} runs\n(Cosine Distance)",
    fontsize=10,
    fontweight="bold",
    pad=12
)

# ---------------- Overall Title ----------------
plt.suptitle(
    f"K-Means Clustering on Uploaded Data\n"
    f"PCA variance explained: {var_explained:.1f}%",
    fontsize=11,
    y=1.02
)

plt.tight_layout()
raw_plot_path = os.path.join(data_name, "kmeans_uploaded_clusters.png")
plt.savefig(raw_plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved raw plot: {raw_plot_path}")

# ---------------------------------------------------------------------------
# 7. VAE parameters
# ---------------------------------------------------------------------------
LATENT_DIMS      = [32]
PRETRAIN_EPOCHS  = 3
BATCH_SIZE       = 128
LEARNING_RATE    = 1e-3
BETA             = 3
SAMPLES_PER_DIST = 3000

N_CLUSTERS = len(np.unique(y))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE, '| N_clusters:', N_CLUSTERS)

# ---------------------------------------------------------------------------
# 8. Beta-VAE Model Definition
# ---------------------------------------------------------------------------
class BetaVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        enc_out = 128 * 4 * 4
        self.fc_mu     = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim, enc_out)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),   nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 128, 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

def beta_vae_loss(x_hat, x, mu, logvar, beta=4.0):
    recon = F.mse_loss(x_hat, x, reduction='mean')
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl

# ---------------------------------------------------------------------------
# 9. Dynamic Data Loading Helper
# ---------------------------------------------------------------------------
def load_data_and_labels(data_name, num_classes=10):
    if data_name == "cifar10":
        tfm = transforms.ToTensor()
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tfm)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tfm)
        targets_test = np.array(test_dataset.targets)
        return train_dataset, test_dataset, targets_test

    elif data_name == "agnew":
        from datasets import load_dataset
        hf_dataset = load_dataset("ag_news")
        
        # AG News has 4 classes (0 to 3)
        class AGNewsImageDataset(torch.utils.data.Dataset):
            def __init__(self, hf_dataset_split):
                self.items = list(hf_dataset_split)
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                item = self.items[idx]
                text = item['text']
                label = item['label']
                
                # Convert text to 3x32x32 image tensor
                arr = np.zeros(3072, dtype=np.float32)
                for i, char in enumerate(text[:3072]):
                    arr[i] = ord(char) / 255.0
                img_tensor = torch.tensor(arr.reshape(3, 32, 32), dtype=torch.float32)
                return img_tensor, label

        train_dataset = AGNewsImageDataset(hf_dataset['train'])
        test_dataset = AGNewsImageDataset(hf_dataset['test'])
        targets_test = np.array([item['label'] for item in hf_dataset['test']])
        return train_dataset, test_dataset, targets_test

    elif data_name == "speechcommand":
        import torchaudio
        import scipy.io.wavfile as wavfile
        
        def mock_torchaudio_load(filepath, *args, **kwargs):
            sr, data = wavfile.read(filepath)
            if data.dtype == np.int16:
                tensor_data = torch.tensor(data.astype(np.float32) / 32768.0, dtype=torch.float32)
            elif data.dtype == np.int32:
                tensor_data = torch.tensor(data.astype(np.float32) / 2147483648.0, dtype=torch.float32)
            else:
                tensor_data = torch.tensor(data, dtype=torch.float32)
            if tensor_data.ndim == 1:
                tensor_data = tensor_data.unsqueeze(0)
            elif tensor_data.ndim == 2:
                tensor_data = tensor_data.t()
            return tensor_data, sr
            
        torchaudio.load = mock_torchaudio_load
        
        import torchaudio.datasets as ad
        
        # Download torchaudio SPEECHCOMMANDS
        base_train = ad.SPEECHCOMMANDS(root='./data', download=True, subset='training')
        base_test = ad.SPEECHCOMMANDS(root='./data', download=True, subset='testing')
        
        # Google Speech Commands has 35 labels. Get the 10 labels:
        SELECTED_SPEECH_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        
        # Extract walker list to get actual labels fast
        available_labels = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in base_train._walker])))
        selected_labels = [l for l in SELECTED_SPEECH_LABELS if l in available_labels]
        if len(selected_labels) < 10:
            selected_labels = (selected_labels + [l for l in available_labels if l not in selected_labels])[:10]
        label_to_idx = {lbl: i for i, lbl in enumerate(selected_labels)}
        
        class SpeechCommandsImageDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, label_to_idx):
                self.dataset = base_dataset
                self.label_to_idx = label_to_idx
                self.indices = []
                self.labels = []
                for idx in range(len(base_dataset)):
                    file_path = base_dataset._walker[idx]
                    lbl = os.path.basename(os.path.dirname(file_path))
                    if lbl in label_to_idx:
                        self.indices.append(idx)
                        self.labels.append(label_to_idx[lbl])
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                base_idx = self.indices[idx]
                waveform, sr, label, speaker_id, utter_num = self.dataset[base_idx]
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Interpolate waveform to 3072 points
                val = F.interpolate(waveform.unsqueeze(0), size=3072, mode='linear', align_corners=False).squeeze()
                val = (val - val.min()) / torch.clamp(val.max() - val.min(), min=1e-5)
                img_tensor = val.view(3, 32, 32)
                return img_tensor, self.labels[idx]

        train_dataset = SpeechCommandsImageDataset(base_train, label_to_idx)
        test_dataset = SpeechCommandsImageDataset(base_test, label_to_idx)
        targets_test = np.array(test_dataset.labels)
        return train_dataset, test_dataset, targets_test

    else:
        raise ValueError(f"Unknown data_name: {data_name}")


print("Loading datasets...")
train_dataset, test_dataset, targets_test = load_data_and_labels(data_name, num_classes=10)
print(f"Loaded train_dataset: {len(train_dataset)} samples, test_dataset: {len(test_dataset)} samples")

pretrain_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0, pin_memory=(DEVICE == 'cuda'))

# ---------------------------------------------------------------------------
# 10. Training Loop
# ---------------------------------------------------------------------------
for LATENT_DIM in LATENT_DIMS:
    MODEL_PATH = f'beta_vae_{data_name}_latent{LATENT_DIM}.pth'
    model = BetaVAE(latent_dim=LATENT_DIM).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        print(f'\n[{LATENT_DIM}d] [OK] Load pretrained: {MODEL_PATH}')
    else:
        print(f'\n[{LATENT_DIM}d] [WARNING] No pretrained found -> training {PRETRAIN_EPOCHS} epochs...')
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        for epoch in range(PRETRAIN_EPOCHS):
            model.train()
            total = 0.0
            for xb, _ in pretrain_loader:
                xb = xb.to(DEVICE)
                optimizer.zero_grad()
                xh, mu, lv, _ = model(xb)
                loss, _, _ = beta_vae_loss(xh, xb, mu, lv, beta=BETA)
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f'  Epoch {epoch+1}/{PRETRAIN_EPOCHS} loss={total/len(pretrain_loader):.4f}')
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'  Saved -> {MODEL_PATH}')

print("\nDone VAE Training!")