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
# Helper function to convert nested dictionaries into a
# SimpleNamespace for easier attribute access.
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
    
    # ------------------------------
    # Sidebar: Model and Config Selection
    # ------------------------------
    st.sidebar.header("Model & Configuration")
    
    # Select between two YAML configuration files
    config_option = st.sidebar.selectbox(
        "Select Model Configuration",
        options=[
            "musicnet.yaml",
            "maestro.yaml"
        ]
    )
    
    # Select the checkpoint to use
    checkpoint_option = st.sidebar.selectbox(
        "Select Checkpoint",
        options=[
            "musicnet_44k_4s-560000.pt",
            "maestro_22k_8s-750000.pt"
        ]
    )
    
    # Control noise level (sigma) for the inpainting process
    sigma_value = st.sidebar.slider("Noise Level (sigma)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # ------------------------------
    # Audio Input Section
    # ------------------------------
    st.header("Audio Input")
    uploaded_file = st.file_uploader("Upload a WAV file for inpainting", type=["wav"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        input_audio_path = "input_temp.wav"
        with open(input_audio_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Audio file uploaded successfully.")
    else:
        # Use a default audio file if none is uploaded
        input_audio_path = "defaults/1727.wav"
        st.info(f"Using default audio file: {input_audio_path}")
    
    # ------------------------------
    # Run Processing When Button is Clicked
    # ------------------------------
    if st.button("Run Inpainting"):
        # ---------------------------------------------------------------------
        # 1. Configuration and Model Initialization
        # ---------------------------------------------------------------------
        st.info("Loading configuration and initializing model...")
        # Build the full path to the selected YAML config
        yaml_config_path = os.path.join("configs", config_option)
        with open(yaml_config_path, "r") as f:
            network_config = yaml.safe_load(f)
        
        # Define experimental parameters (4-second audio at 44.1 kHz)
        exp_config = {
            "sample_rate": 44100,
            "audio_len": 44100 * 4  # 4 seconds = 176400 samples
        }
        
        # Combine configurations and convert to namespace for attribute-style access
        args_dict = {"network": network_config, "exp": exp_config}
        args = dict_to_namespace(args_dict)
        st.write("Configuration loaded:")
        st.json({
            "network": network_config,
            "exp": exp_config
        })
        
        # Import the model definition (make sure the architecture file is accessible)
        from architecture.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention
        
        # Set up the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Using device: {device}")
        
        # Instantiate and move model to the proper device
        model = Unet_CQT_oct_with_attention(args, device).to(device)
        st.write("Model architecture:")
        st.text(str(model))
        
        # ---------------------------------------------------------------------
        # 2. Checkpoint Loading
        # ---------------------------------------------------------------------
        st.info("Loading checkpoint...")
        checkpoint_path = os.path.join( "models", checkpoint_option)  # Adjust path if necessary
        with safe_globals([DictConfig]):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Determine the proper key for the state dictionary
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                st.write("Loaded state dict from 'state_dict' key.")
            elif "network" in checkpoint:
                state_dict = checkpoint["network"]
                st.write("Loaded state dict from 'network' key.")
            else:
                state_dict = checkpoint
                st.write("Loaded checkpoint as state dict.")
        else:
            state_dict = checkpoint
            st.write("Checkpoint loaded as a full model instance.")
        
        load_result = model.load_state_dict(state_dict, strict=False)
        st.write("Missing keys after loading:", load_result.missing_keys)
        st.write("Unexpected keys after loading:", load_result.unexpected_keys)
        model.eval()  # Set the model to evaluation mode
        
        # ---------------------------------------------------------------------
        # 3. Audio Input Processing
        # ---------------------------------------------------------------------
        st.info("Processing audio input...")
        waveform, sr = torchaudio.load(input_audio_path)  # waveform: [channels, num_samples]
        st.write(f"Loaded audio shape: {waveform.shape} at sample rate {sr}")
        
        # Resample if necessary
        if sr != args.exp.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, args.exp.sample_rate)
            waveform = resampler(waveform)
        
        # Convert multi-channel to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Ensure audio length consistency: truncate or pad to the expected length
        expected_length = args.exp.audio_len
        num_samples = waveform.shape[1]
        if num_samples > expected_length:
            waveform = waveform[:, :expected_length]
        elif num_samples < expected_length:
            pad_length = expected_length - num_samples
            waveform = F.pad(waveform, (0, pad_length))
        
        waveform = waveform.to(device)
        audio_input = waveform.squeeze(0).unsqueeze(0)  # shape: [1, expected_length]
        
        # ---------------------------------------------------------------------
        # 4. Spectrogram Generation and Masking
        # ---------------------------------------------------------------------
        st.info("Generating spectrogram and applying mask...")
        # Compute CQT spectrogram (returns a list of octave tensors)
        spectrogram_list = model.CQTransform.fwd(audio_input.unsqueeze(1))
        st.write("Number of octaves in spectrogram:", len(spectrogram_list))
        
        # Use the last octave for inpainting demonstration
        spectrogram = spectrogram_list[-1]
        st.write(f"Chosen spectrogram shape: {spectrogram.shape}")
        
        # Create a mask: zero out the middle third along the time dimension
        batch, ch, freq, time_steps = spectrogram.shape
        mask = torch.ones_like(spectrogram)
        t_start = time_steps // 3
        t_end = 2 * time_steps // 3
        mask[..., t_start:t_end] = 0
        masked_spectrogram = spectrogram * mask
        
        # Replace the original octave with the masked one
        masked_spectrogram_list = spectrogram_list.copy()
        masked_spectrogram_list[-1] = masked_spectrogram
        
        # Convert the masked spectrogram back to the time domain
        masked_audio = model.CQTransform.bwd(masked_spectrogram_list)
        masked_audio = masked_audio.squeeze(1)  # Expected shape: [batch, audio_length]
        
        # ---------------------------------------------------------------------
        # 5. Audio Inpainting
        # ---------------------------------------------------------------------
        st.info("Performing audio inpainting...")
        sigma = torch.tensor([[sigma_value]], dtype=torch.float32).to(device)
        with torch.no_grad():
            inpainted_audio = model(masked_audio, sigma)
        st.write("Inpainting forward pass completed.")
        st.write("Reconstructed audio shape:", inpainted_audio.shape)
        
        # ---------------------------------------------------------------------
        # 6. Save the Inpainted Audio and Stream on UI
        # ---------------------------------------------------------------------
        st.info("Saving inpainted audio...")
        # Convert tensor to CPU and prepare the waveform for saving
        reconstructed_waveform = inpainted_audio.squeeze(0).cpu()
        output_path = "inpainted_audio.wav"
        torchaudio.save(output_path, reconstructed_waveform.unsqueeze(0), args.exp.sample_rate)
        st.success(f"Inpainted audio saved to {output_path}")
        
        # Display the audio on the UI using Streamlit's audio player.
        st.audio(output_path)
        
        # Provide a download button for the inpainted audio.
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        st.download_button(
            label="Download Inpainted Audio",
            data=audio_bytes,
            file_name=output_path,
            mime="audio/wav"
        )

if __name__ == "__main__":
    main()