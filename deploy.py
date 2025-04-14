import os
import torch
import yaml
import torchaudio
import torch.nn.functional as F
import streamlit as st
from types import SimpleNamespace
from omegaconf import DictConfig  # pip install omegaconf
from torch.serialization import safe_globals
from io import BytesIO

# -------------------------------------------------------------------
# Helper function to convert nested dictionaries into a SimpleNamespace 
# for easier attribute access.
# -------------------------------------------------------------------
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

# -------------------------------------------------------------------
# Main function integrating the inpainting workflow into Streamlit
# -------------------------------------------------------------------
def main():
    st.title("Music Inpainting Demo")
    
    # Sidebar: Checkpoint selection (config is determined automatically)
    st.sidebar.header("Model & Configuration")
    checkpoint_option = st.sidebar.selectbox(
        "Select Checkpoint",
        options=["musicnet_44k_4s-560000.pt", "maestro_22k_8s-750000.pt"]
    )
    # Automatically determine the configuration file based on the checkpoint
    if "musicnet" in checkpoint_option:
        config_option = "musicnet.yaml"
    elif "maestro" in checkpoint_option:
        config_option = "maestro.yaml"
    else:
        config_option = None
    st.sidebar.write("Using configuration file:", config_option)
    
    # Noise level parameter for inpainting
    sigma_value = st.sidebar.slider("Noise Level (sigma)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Audio Input Section
    st.header("Audio Input")
    uploaded_file = st.file_uploader("Upload a WAV file for inpainting", type=["wav"])
    if uploaded_file is not None:
        input_audio_path = "input_temp.wav"
        with open(input_audio_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Audio file uploaded successfully.")
    else:
        input_audio_path = os.path.join("defaults", "1727.wav")
        st.info(f"Using default audio file: {input_audio_path}")
    
    # Run processing when the button is clicked
    if st.button("Run Inpainting"):
        with st.spinner("Processing inpainting..."):
            # 1. Configuration and model initialization
            yaml_config_path = os.path.join("configs", config_option)
            with open(yaml_config_path, "r") as f:
                network_config = yaml.safe_load(f)
            exp_config = {"sample_rate": 44100, "audio_len": 44100 * 4}  # 4 seconds audio
            args = dict_to_namespace({"network": network_config, "exp": exp_config})
            
            # Import the model definition (ensure the architecture file is accessible)
            from architecture.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Unet_CQT_oct_with_attention(args, device).to(device)
            
            # 2. Checkpoint Loading
            checkpoint_path = os.path.join("models", checkpoint_option)
            with safe_globals([DictConfig]):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "network" in checkpoint:
                    state_dict = checkpoint["network"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # 3. Audio Input Processing
            waveform, sr = torchaudio.load(input_audio_path)
            if sr != args.exp.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, args.exp.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            expected_length = args.exp.audio_len
            num_samples = waveform.shape[1]
            if num_samples > expected_length:
                waveform = waveform[:, :expected_length]
            elif num_samples < expected_length:
                waveform = F.pad(waveform, (0, expected_length - num_samples))
            waveform = waveform.to(device)
            audio_input = waveform.squeeze(0).unsqueeze(0)
            
            # 4. Spectrogram Generation and Masking
            spectrogram_list = model.CQTransform.fwd(audio_input.unsqueeze(1))
            spectrogram = spectrogram_list[-1]
            batch, ch, freq, time_steps = spectrogram.shape
            mask = torch.ones_like(spectrogram)
            mask[..., time_steps // 3:2 * time_steps // 3] = 0
            masked_spectrogram = spectrogram * mask
            masked_spectrogram_list = spectrogram_list.copy()
            masked_spectrogram_list[-1] = masked_spectrogram
            masked_audio = model.CQTransform.bwd(masked_spectrogram_list).squeeze(1)
            
            # 5. Audio Inpainting
            sigma = torch.tensor([[sigma_value]], dtype=torch.float32).to(device)
            with torch.no_grad():
                inpainted_audio = model(masked_audio, sigma)
            
            # 6. Prepare Inpainted Audio for Streaming/Download (without saving to disk)
            reconstructed_waveform = inpainted_audio.squeeze(0).cpu()
            try:
                import soundfile as sf  # pip install soundfile
            except ImportError:
                st.error("Please install 'soundfile' library (pip install soundfile)")
                return
            mem_file = BytesIO()
            audio_np = reconstructed_waveform.numpy()
            # Transpose to (samples, channels) if stereo (soundfile expects this shape)
            if audio_np.ndim == 2:
                audio_np = audio_np.T
            sf.write(mem_file, audio_np, args.exp.sample_rate, format="WAV")
            mem_file.seek(0)
            audio_bytes = mem_file.read()
            
        st.success("Inpainting completed.")
        st.audio(audio_bytes, format="audio/wav")
        st.download_button(
            label="Download Inpainted Audio",
            data=audio_bytes,
            file_name="inpainted_audio.wav",
            mime="audio/wav"
        )

if __name__ == "__main__":
    main()
