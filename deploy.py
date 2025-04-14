import os
import torch
import yaml
import torchaudio
import torch.nn.functional as F
from types import SimpleNamespace
from omegaconf import DictConfig  # pip install omegaconf
from torch.serialization import safe_globals

# -----------------------------------------------------------------------------
# Helper function to convert nested dictionaries to SimpleNamespace objects
# for easier attribute access
# -----------------------------------------------------------------------------
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

# =============================================================================
# 1. Configuration and Model Initialization
# =============================================================================

# Load network configuration from YAML file
# yaml_config_path = r"configs/paper_1912_unet_cqt_oct_attention_44k_2 .yaml"
yaml_config_path = r"configs/paper_1912_unet_cqt_oct_attention_adaLN_2.yaml"
with open(yaml_config_path, "r") as f:
    network_config = yaml.safe_load(f)

# Define experimental parameters (4-second audio at 44.1 kHz)
exp_config = {
    "sample_rate": 44100,
    "audio_len": 44100 * 4  # 176400 samples for 4 seconds
}

# Combine configurations and convert to namespace for attribute-style access
args_dict = {"network": network_config, "exp": exp_config}
args = dict_to_namespace(args_dict)
print("Configuration loaded:")
print(args)

# Import model definition
from architecture.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention

# Set up device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Instantiate model and move to appropriate device
model = Unet_CQT_oct_with_attention(args, device).to(device)
print("Model architecture:")
print(model)

# =============================================================================
# 2. Checkpoint Loading
# =============================================================================
# checkpoint_path = "musicnet_44k_4s-560000.pt"
checkpoint_path = "maestro_22k_8s-750000.pt"
with safe_globals([DictConfig]):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("Loaded state dict from 'state_dict' key.")
    elif "network" in checkpoint:
        state_dict = checkpoint["network"]
        print("Loaded state dict from 'network' key.")
    else:
        state_dict = checkpoint
        print("Loaded checkpoint as state dict.")
else:
    state_dict = checkpoint
    print("Checkpoint loaded as a full model instance.")

# Load state dictionary with flexible matching
load_result = model.load_state_dict(state_dict, strict=False)
print("Missing keys after loading:", load_result.missing_keys)
print("Unexpected keys after loading:", load_result.unexpected_keys)
model.eval()  # Set model to evaluation mode

# =============================================================================
# 3. Audio Input Processing
# =============================================================================
audio_path = r"1727.wav"
waveform, sr = torchaudio.load(audio_path)  # waveform shape: [channels, num_samples]
print(f"Loaded audio with shape {waveform.shape} and sample rate {sr}")

# Resample if necessary
if sr != args.exp.sample_rate:
    resampler = torchaudio.transforms.Resample(sr, args.exp.sample_rate)
    waveform = resampler(waveform)

# Convert stereo to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Ensure consistent audio length by truncating or padding
expected_length = args.exp.audio_len
num_samples = waveform.shape[1]
if num_samples > expected_length:
    waveform = waveform[:, :expected_length]
elif num_samples < expected_length:
    pad_length = expected_length - num_samples
    waveform = F.pad(waveform, (0, pad_length))

waveform = waveform.to(device)
audio_input = waveform.squeeze(0).unsqueeze(0)  # shape: [1, expected_length]

# =============================================================================
# 4. Spectrogram Generation and Masking
# =============================================================================
# Compute CQT spectrogram (returns a list of octave tensors)
spectrogram_list = model.CQTransform.fwd(audio_input.unsqueeze(1))
print("Number of octaves in spectrogram:", len(spectrogram_list))

# Select one octave (the last one) for inpainting demonstration
spectrogram = spectrogram_list[-1]
print(f"Chosen spectrogram shape: {spectrogram.shape}")

# Create a mask to simulate missing audio segments (middle third removed)
batch, ch, freq, time_steps = spectrogram.shape
mask = torch.ones_like(spectrogram)
t_start = time_steps // 3
t_end = 2 * time_steps // 3
mask[..., t_start:t_end] = 0
masked_spectrogram = spectrogram * mask

# =============================================================================
# 5. Converting Masked Spectrogram to Time Domain
# =============================================================================
# Create a copy of the original spectrogram list and replace the masked octave
masked_spectrogram_list = spectrogram_list.copy()
masked_spectrogram_list[-1] = masked_spectrogram

# Convert back to time domain using the inverse transformation
masked_audio = model.CQTransform.bwd(masked_spectrogram_list)
# Expected shape: [batch, 1, audio_length]
masked_audio = masked_audio.squeeze(1)  # shape: [batch, audio_length]

# =============================================================================
# 6. Audio Inpainting
# =============================================================================
# Set noise level for the model
sigma = torch.tensor([[0.5]], dtype=torch.float32).to(device)
# Perform inpainting inference
with torch.no_grad():
    inpainted_audio = model(masked_audio, sigma)
print("Inpainting forward pass completed.")
print("Reconstructed audio shape:", inpainted_audio.shape)

# =============================================================================
# 7. Save Results
# =============================================================================
# Convert tensor to CPU and prepare for saving
reconstructed_waveform = inpainted_audio.squeeze(0).cpu()
output_path = "inpainted_audio.wav"
torchaudio.save(output_path, reconstructed_waveform.unsqueeze(0), args.exp.sample_rate)
print(f"Inpainted audio saved to {output_path}")