# ===========================================================================
# Beta-VAE Model Training (Targeted 32D Latent Representation)
# ===========================================================================

import os
import ast
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 0. Load Configuration
# ---------------------------------------------------------------------------
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
data_name = config.get("data_name", "cifar10")
print(f"Using dataset: {data_name}")

# ---------------------------------------------------------------------------
# 1. Load Distribution Files (Required for downstream inference & plotting)
# ---------------------------------------------------------------------------
def load_cluster_file(filename):
    with open(filename, "r") as f:
        content = f.read().strip()
    if content.endswith(","):
        content = content[:-1]
    return np.array(ast.literal_eval("[" + content + "]"), dtype=np.float32)

X0 = load_cluster_file(os.path.join(data_name, "cluster0.txt"))
X1 = load_cluster_file(os.path.join(data_name, "cluster1.txt"))
X2 = load_cluster_file(os.path.join(data_name, "cluster2.txt"))

X = np.vstack([X0, X1, X2])
y = np.concatenate([
    np.zeros(len(X0), dtype=int),
    np.ones(len(X1), dtype=int),
    np.full(len(X2), 2, dtype=int)
])

print("X shape:", X.shape, "| y shape:", y.shape)
np.save("X_loaded.npy", X)
np.save("y_loaded.npy", y)

# ---------------------------------------------------------------------------
# 2. VAE Model & Loss Parameters
# ---------------------------------------------------------------------------
LATENT_DIM      = 32
PRETRAIN_EPOCHS  = 3
BATCH_SIZE       = 128
LEARNING_RATE    = 1e-3
BETA             = 3.0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE, '| Latent dimension:', LATENT_DIM)

# ---------------------------------------------------------------------------
# 3. Beta-VAE Model Definition
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
# 4. Dynamic Data Loading Helper
# ---------------------------------------------------------------------------
def load_training_data(data_name):
    if data_name == "cifar10":
        tfm = transforms.ToTensor()
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tfm)
        return train_dataset

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
                arr = np.zeros(3072, dtype=np.float32)
                for i, char in enumerate(text[:3072]):
                    arr[i] = ord(char) / 255.0
                img_tensor = torch.tensor(arr.reshape(3, 32, 32), dtype=torch.float32)
                return img_tensor, item['label']

        return AGNewsImageDataset(hf_dataset['train'])

    elif data_name == "speechcommand":
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
            
        import torchaudio
        torchaudio.load = mock_torchaudio_load
        import torchaudio.datasets as ad
        
        base_train = ad.SPEECHCOMMANDS(root='./data', download=True, subset='training')
        
        SELECTED_SPEECH_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
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
                for idx in range(len(base_dataset)):
                    file_path = base_dataset._walker[idx]
                    lbl = os.path.basename(os.path.dirname(file_path))
                    if lbl in label_to_idx:
                        self.indices.append((idx, label_to_idx[lbl]))
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                base_idx, label = self.indices[idx]
                waveform, sr, _, _, _ = self.dataset[base_idx]
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                val = F.interpolate(waveform.unsqueeze(0), size=3072, mode='linear', align_corners=False).squeeze()
                val = (val - val.min()) / torch.clamp(val.max() - val.min(), min=1e-5)
                img_tensor = val.view(3, 32, 32)
                return img_tensor, label

        return SpeechCommandsImageDataset(base_train, label_to_idx)
    else:
        raise ValueError(f"Unknown data_name: {data_name}")

# ---------------------------------------------------------------------------
# 5. Training Execution
# ---------------------------------------------------------------------------
print("Loading training dataset...")
train_dataset = load_training_data(data_name)
print(f"Loaded train_dataset: {len(train_dataset)} samples")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE == 'cuda'))

MODEL_PATH = f'beta_vae_{data_name}_latent{LATENT_DIM}.pth'
model = BetaVAE(latent_dim=LATENT_DIM).to(DEVICE)

if os.path.exists(MODEL_PATH):
    print(f'\n[OK] Pretrained model already exists: {MODEL_PATH}')
else:
    print(f'\n[WARNING] Pretrained model not found -> training {PRETRAIN_EPOCHS} epochs...')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        total = 0.0
        for xb, _ in train_loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            xh, mu, lv, _ = model(xb)
            loss, _, _ = beta_vae_loss(xh, xb, mu, lv, beta=BETA)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'  Epoch {epoch+1}/{PRETRAIN_EPOCHS} | Average loss={total/len(train_loader):.4f}')
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved successfully -> {MODEL_PATH}')

print("\nDone VAE Training!")