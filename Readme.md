# AudioGANs

AudioGANs is a modular and configurable framework for training Generative Adversarial Networks (GANs) on audio data. It supports various GAN architectures and is designed for flexibility, ease of experimentation, and interactive demos. This repository now includes a **Music Inpainting Demo** powered by Streamlit and a **Telegram Bot Interface**.

## Features

- **Modular Architecture**: Easily switch between different GAN models.
- **Configurable Training**: Customize training parameters via YAML configuration files.
- **Interactive Demo**: Run a Music Inpainting demo using Streamlit, allowing interactive checkpoint selection, noise level adjustment, and audio upload.
- **Telegram Bot**: Interact with the inpainting model via Telegram by sending WAV files.
- **Deployment & Inference**: Scripts available for deploying trained models on audio inputs.
- **Dependency Management**: A script to install required libraries and download pretrained weights from Hugging Face.

## Repository Structure

```
AudioGANs/
├── architecture/             # Contains different GAN model architectures
├── configs/                  # YAML configuration files
├── defaults/                 # Default assets (e.g., sample WAV files)
├── dependencies.sh           # Install Python dependencies and download weights
├── run.sh                    # Launches Streamlit demo
├── deploy.py                 # Streamlit Music Inpainting demo
├── telegram_bot.py           # Telegram bot implementation
├── requirements.txt          # Python dependencies list
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.7 or higher
- Additional dependencies listed in `requirements.txt`
- [Streamlit](https://streamlit.io/) (for demo)
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) (for bot)

### Installation

```bash
git clone https://github.com/Rohan-ingle/AudioGANs.git
cd AudioGANs
bash dependencies.sh
```

### Verify Installation

```bash
bash verify.sh
```

## Running the Music Inpainting Demo (Streamlit)

```bash
bash run.sh
```

This opens the demo in a browser where you can:

- Choose a checkpoint (e.g., `musicnet_44k_4s-560000.pt`)
- Set noise level (sigma)
- Upload a WAV file or use the default
- Click **Run Inpainting** to see results

## Using the Telegram Bot

### Setup

1. Create a `.env` file in the root directory:

```dotenv
TOKEN=your_telegram_bot_token_here
```

2. Run the bot:

```bash
python telegram_bot.py
```

### Interacting with the Bot

- Start with `/start` for instructions
- Use `/inpaint [checkpoint] [sigma]` (e.g., `/inpaint musicnet 0.5`)
- Send a `.wav` file or voice message
- Bot replies with the inpainted audio

## Bot Code Highlights

- Loads token securely from `.env`
- Uses `torchaudio`, `soundfile`, and PyTorch
- Supports `musicnet` and `maestro` checkpoints
- Auto-pads or trims audio to model's required length

## Model Architecture Support

To add new models:

1. Place model file in `architecture/`
2. Add YAML config to `configs/`
3. Adjust `run_inpainting()` to use new architecture

## License

MIT License. See the [LICENSE](LICENSE) file.

## Acknowledgments

Inspired by state-of-the-art GAN audio models. Paper: [Arxiv](https://arxiv.org/abs/2305.15266).
