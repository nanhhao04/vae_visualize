# ===========================================================================
# VAE Latent Space Inference and Clustering Evaluation
# ===========================================================================

import os
import pickle
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# 0. Load Configuration and Shared Data
# ---------------------------------------------------------------------------
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
data_name = config.get("data_name", "cifar10")
print(f"Inference using dataset: {data_name}")

assert os.path.exists("X_loaded.npy"), "X_loaded.npy not found. Run training.py first."
assert os.path.exists("y_loaded.npy"), "y_loaded.npy not found. Run training.py first."

X = np.load("X_loaded.npy")
y = np.load("y_loaded.npy")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

N_DISTRIBUTIONS = len(X)
print('X shape:', X.shape, '| y shape:', y.shape)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
LATENT_DIMS      = [16, 32, 64]
BATCH_SIZE       = 128
SAMPLES_PER_DIST = 1000

N_CLUSTERS = len(np.unique(y))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE, '| N_clusters:', N_CLUSTERS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def l2_normalize(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-10, None)

def clustering_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for p, t in zip(y_pred, y_true):
        w[p, t] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return w[r, c].sum() / len(y_true)

def sample_indices_from_distribution(targets, distribution, total_samples=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    dist = np.asarray(distribution, dtype=np.float64)
    dist = dist / dist.sum()

    counts = np.floor(dist * total_samples).astype(int)
    remainder = total_samples - counts.sum()
    if remainder > 0:
        frac = dist * total_samples - counts
        counts[np.argsort(-frac)[:remainder]] += 1

    indices = []
    num_classes = len(distribution)
    for cls in range(num_classes):
        n = counts[cls]
        if n == 0:
            continue
        actual_cls = cls % len(np.unique(targets))
        cls_indices = np.where(targets == actual_cls)[0]
        selected = rng.choice(cls_indices, size=n, replace=(n > len(cls_indices)))
        indices.extend(selected.tolist())

    rng.shuffle(indices)
    return indices

# ---------------------------------------------------------------------------
# Beta-VAE Model Definition
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

# ---------------------------------------------------------------------------
# Load Dataset (test set only for inference)
# ---------------------------------------------------------------------------
def load_test_dataset(data_name):
    if data_name == "cifar10":
        tfm = transforms.ToTensor()
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tfm)
        targets_test = np.array(test_dataset.targets)
        return test_dataset, targets_test

    elif data_name == "agnew":
        from datasets import load_dataset
        hf_dataset = load_dataset("ag_news")
        
        class AGNewsImageDataset(torch.utils.data.Dataset):
            def __init__(self, hf_dataset_split):
                self.items = list(hf_dataset_split)
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                item = self.items[idx]
                text = item['text']
                label = item['label']
                arr = np.zeros(3072, dtype=np.float32)
                for i, char in enumerate(text[:3072]):
                    arr[i] = ord(char) / 255.0
                img_tensor = torch.tensor(arr.reshape(3, 32, 32), dtype=torch.float32)
                return img_tensor, label

        test_dataset = AGNewsImageDataset(hf_dataset['test'])
        targets_test = np.array([item['label'] for item in hf_dataset['test']])
        return test_dataset, targets_test

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
        
        base_test = ad.SPEECHCOMMANDS(root='./data', download=True, subset='testing')
        
        SELECTED_SPEECH_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        available_labels = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in base_test._walker])))
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
                
                val = F.interpolate(waveform.unsqueeze(0), size=3072, mode='linear', align_corners=False).squeeze()
                val = (val - val.min()) / torch.clamp(val.max() - val.min(), min=1e-5)
                img_tensor = val.view(3, 32, 32)
                return img_tensor, self.labels[idx]

        test_dataset = SpeechCommandsImageDataset(base_test, label_to_idx)
        targets_test = np.array(test_dataset.labels)
        return test_dataset, targets_test

    else:
        raise ValueError(f"Unknown data_name: {data_name}")


print("Loading test dataset...")
test_dataset, targets_test = load_test_dataset(data_name)
print(f"Loaded test_dataset: {len(test_dataset)} samples")

results = {}
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Inference Loop
# ---------------------------------------------------------------------------
for LATENT_DIM in LATENT_DIMS:
    MODEL_PATH = f'beta_vae_{data_name}_latent{LATENT_DIM}.pth'
    model = BetaVAE(latent_dim=LATENT_DIM).to(DEVICE)

    assert os.path.exists(MODEL_PATH), f"Model path not found: {MODEL_PATH}"
    print(f'\n[{LATENT_DIM}d] [OK] Load pretrained: {MODEL_PATH}')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    Z_vae = []
    print(f'[{LATENT_DIM}d] Inference {N_DISTRIBUTIONS} distributions x {SAMPLES_PER_DIST} samples...')
    for i in range(N_DISTRIBUTIONS):
        if (i + 1) % 100 == 0 or i == 0:
            print(f'  {i+1}/{N_DISTRIBUTIONS}')

        # Sample SAMPLES_PER_DIST ảnh theo phân phối X[i]
        sampled_indices = sample_indices_from_distribution(
            targets_test, X[i], total_samples=SAMPLES_PER_DIST, rng=rng
        )
        sampled_ds     = Subset(test_dataset, sampled_indices)
        eval_loader    = DataLoader(sampled_ds, batch_size=BATCH_SIZE, shuffle=False)

        mu_all = []
        with torch.no_grad():
            for xb, _ in eval_loader:
                mu, _ = model.encode(xb.to(DEVICE))
                mu_all.append(mu.cpu().numpy())

        # Mean của tất cả mu -> 1 vector đại diện cho distribution i
        Z_vae.append(np.vstack(mu_all).mean(axis=0))

    Z_vae = np.asarray(Z_vae, dtype=np.float32)  # (N_DISTRIBUTIONS, LATENT_DIM)
    Z_norm = l2_normalize(Z_vae)

    # ---- K-Means trên N_DISTRIBUTIONS điểm ----
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10).fit(Z_norm)
    labels = km.labels_

    nmi = normalized_mutual_info_score(y, labels)
    acc = clustering_accuracy(y, labels)
    print(f'  NMI={nmi:.4f}  ACC={acc:.4f}')

    # ---- PCA 2D ----
    pca  = PCA(n_components=2, random_state=42)
    Z_2d = pca.fit_transform(Z_norm)

    results[LATENT_DIM] = dict(Z_2d=Z_2d, labels=labels, nmi=nmi, acc=acc)

# Save intermediate inference results for gen_plot.py
with open("results_inferred.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nDone Inference! Intermediate results saved to results_inferred.pkl")