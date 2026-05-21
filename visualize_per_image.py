# ===========================================================================
# VAE 32D Latent Space Visualization Per Image
# ===========================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# 1. Model Definition (Same as training.py & inference.py)
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

def beta_vae_loss(x_hat, x, mu, logvar, beta=3.0):
    recon = F.mse_loss(x_hat, x, reduction='mean')
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl

# ---------------------------------------------------------------------------
# 2. Dynamic Data Loading Helper
# ---------------------------------------------------------------------------
def load_dataset_split(data_name):
    """Loads the dataset split and returns (dataset, targets, class_names)"""
    if data_name == "cifar10":
        tfm = transforms.ToTensor()
        ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=tfm)
        targets = np.array(ds.targets)
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        return ds, targets, class_names

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

        ds = AGNewsImageDataset(hf_dataset['test'])
        targets = np.array([item['label'] for item in hf_dataset['test']])
        class_names = ["World", "Sports", "Business", "Sci/Tech"]
        return ds, targets, class_names

    elif data_name == "speechcommand":
        import scipy.io.wavfile as wavfile
        
        # Mock torchaudio.load for Windows compatibility
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
            
        import torchaudio
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

        ds = SpeechCommandsImageDataset(base_test, label_to_idx)
        targets = np.array(ds.labels)
        return ds, targets, selected_labels

    else:
        raise ValueError(f"Unknown data_name: {data_name}")

# ---------------------------------------------------------------------------
# 3. Main Logic
# ---------------------------------------------------------------------------
def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    datasets_to_visualize = ["agnew", "cifar10", "speechcommand"]
    
    for data_name in datasets_to_visualize:
        print(f"\n=============================================================")
        print(f"Processing Dataset: {data_name}")
        print(f"=============================================================")
        
        # Load data
        ds, targets, class_names = load_dataset_split(data_name)
        num_classes = len(class_names)
        print(f"Loaded {len(ds)} test samples with {num_classes} classes: {class_names}")
        
        # Select exactly 50 samples per label
        selected_indices = []
        selected_labels = []
        for cls_idx in range(num_classes):
            cls_indices = np.where(targets == cls_idx)[0]
            if len(cls_indices) < 50:
                print(f"[WARNING] Class {class_names[cls_idx]} has only {len(cls_indices)} test samples. Using all available.")
                chosen = cls_indices
            else:
                chosen = cls_indices[:50]
            selected_indices.extend(chosen)
            selected_labels.extend([cls_idx] * len(chosen))
            
        print(f"Selected total {len(selected_indices)} samples (target: 50 per label)")
        
        # Build Subset and DataLoader
        sub_ds = Subset(ds, selected_indices)
        loader = DataLoader(sub_ds, batch_size=64, shuffle=False)
        
        # Model path
        model_path = f"beta_vae_{data_name}_latent32.pth"
        model = BetaVAE(latent_dim=32).to(DEVICE)
        
        # Load weights, if missing train a quick VAE model on the fly to avoid crashing
        if os.path.exists(model_path):
            print(f"Loading pretrained weights from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            print(f"[WARNING] Pretrained model {model_path} not found! Training a quick VAE on the fly...")
            # We will train for 3 epochs using the subset of data to avoid taking too much time
            quick_train_loader = DataLoader(ds, batch_size=128, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            for epoch in range(3):
                total_loss = 0.0
                # Limit to 50 batches for fast CPU training if file is missing
                batch_count = 0
                for xb, _ in quick_train_loader:
                    xb = xb.to(DEVICE)
                    optimizer.zero_grad()
                    xh, mu, lv, _ = model(xb)
                    loss, _, _ = beta_vae_loss(xh, xb, mu, lv, beta=3.0)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_count += 1
                    if batch_count >= 50: 
                        break
                print(f"  Epoch {epoch+1}/3 Loss: {total_loss/batch_count:.4f}")
            torch.save(model.state_dict(), model_path)
            print(f"Saved quick model to: {model_path}")
            
        # Inference
        model.eval()
        latent_features = []
        with torch.no_grad():
            for xb, _ in loader:
                mu, _ = model.encode(xb.to(DEVICE))
                latent_features.append(mu.cpu().numpy())
                
        latent_features = np.vstack(latent_features)
        print(f"Extracted VAE Latent Features shape: {latent_features.shape}")
        
        # PCA projection to 2D
        pca = PCA(n_components=2, random_state=42)
        latent_2d = pca.fit_transform(latent_features)
        var_explained = pca.explained_variance_ratio_.sum() * 100
        
        # ---------------------------------------------------------------------------
        # Beautiful Scatter Plot Visualization
        # ---------------------------------------------------------------------------
        plt.figure(figsize=(8, 6))
        
        # Custom harmonic palettes
        if num_classes == 4:
            colors = ["#1D9E75", "#7F77DD", "#D85A30", "#3C3489"]
            markers = ["o", "s", "^", "D"]
        else:
            colors = [
                "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"
            ]
            markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
            
        for cls_idx in range(num_classes):
            mask = np.array(selected_labels) == cls_idx
            plt.scatter(
                latent_2d[mask, 0],
                latent_2d[mask, 1],
                c=colors[cls_idx % len(colors)],
                marker=markers[cls_idx % len(markers)],
                s=50,
                alpha=0.8,
                edgecolors='w',
                linewidths=0.5,
                label=class_names[cls_idx]
            )
            
        plt.xlabel("Latent PC1", fontsize=10)
        plt.ylabel("Latent PC2", fontsize=10)
        
        # Sleek Legend (only labels, no title)
        plt.legend(
            fontsize='9', 
            loc='best',
            frameon=True, 
            shadow=False, 
            facecolor='#F8F9FA', 
            edgecolor='#E5E5E5'
        )
        
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        
        # Create folder if not exists
        os.makedirs(data_name, exist_ok=True)
        save_path = os.path.join(data_name, "visualize_latent32.pdf")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        
        print(f"Success! Saved visualization to: {save_path}")

if __name__ == "__main__":
    main()
