import os
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torchaudio

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

# Mock torchaudio load
torchaudio.load = mock_torchaudio_load

# Test loading an item from SPEECHCOMMANDS
base_test = torchaudio.datasets.SPEECHCOMMANDS(root='./data', download=False, subset='testing')
print("Successfully loaded SPEECHCOMMANDS dataset object")
item = base_test[0]
print("Successfully loaded item 0:", item[0].shape, item[1], item[2])
