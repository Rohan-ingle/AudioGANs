import os
import torch
import yaml
import torchaudio
import torch.nn.functional as F
from types import SimpleNamespace
from torch.serialization import safe_globals
from io import BytesIO
import soundfile as sf

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv

# Load token from .env file
load_dotenv()
TOKEN = os.getenv("TOKEN")
if not TOKEN:
    raise ValueError("TOKEN not found in environment variables.")

##############################################
# Helper function: convert dictionary to namespace.
##############################################
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

##############################################
# Function: run_inpainting
#
# input_audio_path: path to the WAV file to inpaint.
# checkpoint_option: string; e.g. "musicnet_44k_4s-560000.pt" or "maestro_22k_8s-750000.pt"
# sigma_value: float noise level parameter.
#
# Returns: inpainted audio as bytes (WAV format).
##############################################
def run_inpainting(input_audio_path, checkpoint_option="musicnet_44k_4s-560000.pt", sigma_value=0.5):
    # Determine configuration file based on checkpoint
    if "musicnet" in checkpoint_option:
        config_option = "musicnet.yaml"
    elif "maestro" in checkpoint_option:
        config_option = "maestro.yaml"
    else:
        raise ValueError("Unknown checkpoint option.")
    
    # Load network configuration from YAML
    yaml_config_path = os.path.join("configs", config_option)
    with open(yaml_config_path, "r") as f:
        network_config = yaml.safe_load(f)
    
    # Define experiment configuration. (Adjust sample_rate/audio_len as needed.)
    exp_config = {"sample_rate": 44100, "audio_len": 44100 * 4}  # 4 seconds of audio
    args = dict_to_namespace({"network": network_config, "exp": exp_config})
    
    # Import the model architecture. Ensure this file exists in architecture/
    from architecture.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet_CQT_oct_with_attention(args, device).to(device)
    
    # Load checkpoint (model file from the models/ directory)
    checkpoint_path = os.path.join("models", checkpoint_option)
    with safe_globals([torch.nn.Module, SimpleNamespace]):
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
    
    # Load and preprocess audio input using torchaudio
    waveform, sr = torchaudio.load(input_audio_path)
    if sr != args.exp.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, args.exp.sample_rate)
        waveform = resampler(waveform)
    # Ensure mono-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    expected_length = args.exp.audio_len
    num_samples = waveform.shape[1]
    if num_samples > expected_length:
        waveform = waveform[:, :expected_length]
    elif num_samples < expected_length:
        waveform = F.pad(waveform, (0, expected_length - num_samples))
    waveform = waveform.to(device)
    audio_input = waveform.unsqueeze(0)  # shape: (1, 1, num_samples)
    
    # Generate spectrogram and apply masking
    spectrogram_list = model.CQTransform.fwd(audio_input.unsqueeze(1))
    spectrogram = spectrogram_list[-1]
    batch, ch, freq, time_steps = spectrogram.shape
    mask = torch.ones_like(spectrogram)
    mask[..., time_steps // 3:2 * time_steps // 3] = 0
    masked_spectrogram = spectrogram * mask
    masked_spectrogram_list = spectrogram_list.copy()
    masked_spectrogram_list[-1] = masked_spectrogram
    masked_audio = model.CQTransform.bwd(masked_spectrogram_list).squeeze(1)
    
    # Perform inpainting using the model; sigma controls the noise level.
    sigma = torch.tensor([[sigma_value]], dtype=torch.float32).to(device)
    with torch.no_grad():
        inpainted_audio = model(masked_audio, sigma)
    
    # Prepare the inpainted audio for sending via Telegram (without writing to disk)
    reconstructed_waveform = inpainted_audio.squeeze(0).cpu()
    mem_file = BytesIO()
    audio_np = reconstructed_waveform.numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # soundfile expects shape (samples, channels)
    sf.write(mem_file, audio_np, args.exp.sample_rate, format="WAV")
    mem_file.seek(0)
    audio_bytes = mem_file.read()
    
    return audio_bytes

##############################################
# Telegram Bot Handlers
##############################################
from telegram.ext import CallbackContext

# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = (
        "Welcome to the Audio Inpainting Bot!\n\n"
        "Send me a WAV file and I will restore the missing sections using our cutting-edge inpainting model.\n"
        "You can optionally specify a checkpoint and noise level by sending a message like:\n"
        "`/inpaint musicnet 0.5`\n"
        "If you do not specify parameters, default values will be used."
    )
    await update.message.reply_text(welcome_message)

# /inpaint command handler to allow optional parameter specification
async def inpaint_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Expected format: /inpaint [checkpoint] [sigma]
    # For instance: /inpaint maestro 0.7 OR /inpaint musicnet 0.5
    args = context.args
    # Set defaults:
    checkpoint_option = "musicnet_44k_4s-560000.pt"
    sigma_value = 0.5
    if len(args) >= 1:
        if args[0].lower() in ["musicnet", "maestro"]:
            if args[0].lower() == "maestro":
                checkpoint_option = "maestro_22k_8s-750000.pt"
            else:
                checkpoint_option = "musicnet_44k_4s-560000.pt"
    if len(args) >= 2:
        try:
            sigma_value = float(args[1])
        except ValueError:
            await update.message.reply_text("Invalid sigma value. Using default sigma=0.5")
            sigma_value = 0.5

    # Tell the user to send an audio file
    reply_text = (
        f"Using checkpoint: {checkpoint_option}\n"
        f"Using sigma value: {sigma_value}\n"
        "Now please send me a WAV audio file to process."
    )
    context.user_data["checkpoint_option"] = checkpoint_option
    context.user_data["sigma_value"] = sigma_value
    await update.message.reply_text(reply_text)

# Message handler for processing incoming audio files (document or voice)
async def process_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Get checkpoint and sigma from user data; use defaults if not set.
    checkpoint_option = context.user_data.get("checkpoint_option", "musicnet_44k_4s-560000.pt")
    sigma_value = context.user_data.get("sigma_value", 0.5)

    # Check if the incoming message has a document (and ensure it is .wav) or a voice message
    file_id = None
    if update.message.document:
        file_name = update.message.document.file_name
        if not file_name.lower().endswith(".wav"):
            await update.message.reply_text("Please send a WAV file.")
            return
        file_id = update.message.document.file_id
    elif update.message.voice:
        file_id = update.message.voice.file_id
    else:
        await update.message.reply_text("No audio file found. Please send a WAV file.")
        return

    # Download the file to a temporary location
    file = await context.bot.get_file(file_id)
    temp_input = "temp_input.wav"
    await file.download_to_drive(custom_path=temp_input)
    await update.message.reply_text("Audio file received. Processing inpainting, please wait...")

    # Run the inpainting process (this may take a while)
    try:
        result_audio_bytes = run_inpainting(temp_input, checkpoint_option, sigma_value)
    except Exception as e:
        await update.message.reply_text(f"Error during inpainting: {e}")
        return

    # Send the resulting audio back to the user as a document (audio file)
    await update.message.reply_document(document=result_audio_bytes, filename="inpainted_audio.wav", caption="Here is your inpainted audio!")
    
    # Optionally, remove the temporary file
    try:
        os.remove(temp_input)
    except Exception:
        pass

##############################################
# Main function: Telegram Bot Application
##############################################
def main():
    app = Application.builder().token(TOKEN).build()

    # Command handlers: /start and /inpaint
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("inpaint", inpaint_command))

    # Audio message handler: listens for documents or voice messages with audio
    app.add_handler(MessageHandler(
        filters.Document.FileExtension("wav") | filters.VOICE,
        process_audio
    ))

    print("Audio Inpainting Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()