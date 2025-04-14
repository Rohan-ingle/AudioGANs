# AudioGANs

AudioGANs is a modular and configurable framework for training Generative Adversarial Networks (GANs) on audio data. It supports various GAN architectures and is designed for flexibility, ease of experimentation, and interactive demos. This repository now includes a **Music Inpainting Demo** powered by Streamlit.

## Features

- **Modular Architecture**: Easily switch between different GAN models.
- **Configurable Training**: Customize training parameters via YAML configuration files.
- **Interactive Demo**: Run a Music Inpainting demo using Streamlit, allowing interactive checkpoint selection, noise level adjustment, and audio upload.
- **Deployment & Inference**: Scripts available for deploying trained models on audio inputs.
- **Dependency Management**: A script to install required libraries and download pretrained weights from Hugging Face.

## Repository Structure

```
AudioGANs/
├── architecture/             # Contains different GAN model architectures (e.g., Unet_CQT_oct_with_attention)
├── configs/                  # YAML configuration files for various checkpoints (e.g., musicnet.yaml, maestro.yaml)
├── defaults/                 # Default assets (e.g., default WAV files used when no input is provided)
├── dependencies.sh           # Installs Python dependencies and downloads pretrained weights
├── run.sh                    # Shell script to launch the Streamlit demo (runs deploy.py)
├── deploy.py                 # Streamlit application for Music Inpainting demo
├── requirements.txt          # List of required Python packages
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.7 or higher
- Additional dependencies listed in `requirements.txt`
- [Streamlit](https://streamlit.io/) (for running the demo)

### Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/Rohan-ingle/AudioGANs.git
    cd AudioGANs
    ```

2. **Install Dependencies and Download Pretrained Weights**:

    Run the `dependencies.sh` script to install all required Python libraries and download the necessary pretrained weights from Hugging Face.

    ```bash
    bash dependancies.sh
    ```

## Usage

### Running the Music Inpainting Demo

The repository includes a demo that lets you inpaint (restore masked sections of) audio files using a trained GAN model. To run the demo:

1. **Launch the Streamlit Application using `run.sh`**:

    ```bash
    bash run.sh
    ```

    This script runs the Streamlit application (via `deploy.py`), which will open a new browser window displaying the Music Inpainting Demo.

2. **Demo Interface Overview**:

    - **Model & Configuration Selection**:  
      Use the sidebar to select a checkpoint (e.g., `musicnet_44k_4s-560000.pt` or `maestro_22k_8s-750000.pt`). The corresponding configuration file (either `musicnet.yaml` or `maestro.yaml`) is chosen automatically.

    - **Noise Level Adjustment**:  
      Adjust the noise level (sigma) for the inpainting process with a slider.

    - **Audio Input**:  
      Upload a WAV file or use the default audio sample (located in the `defaults` folder).

    - **Inpainting Process**:  
      Click the "Run Inpainting" button. The demo will:
        - Load and preprocess the input audio.
        - Generate a spectrogram and mask its central portion.
        - Perform inpainting on the masked spectrogram using the GAN model.
        - Reconstruct and output the inpainted audio, with options to play and download the result.

### Training a Model

If you wish to train a model, modify the YAML configuration files in the `configs/` directory to adjust training parameters (e.g., learning rate, batch size, number of epochs). (Note: The current focus of the repository is on the inpainting demo. Training scripts and details will be provided separately.)

### Deploying a Trained Model

For non-interactive deployment, you can also run the deployment script directly:

```bash
python deploy.py --model_path path_to_trained_model --input_audio path_to_input_audio --output_path path_to_save_output
```

This processes the input audio using a trained GAN model and saves the output at the designated location.

## Code Walkthrough

The main functionalities in the Streamlit demo (inside `deploy.py`) include:

- **Configuration Loading**:  
  The app reads YAML configuration files from `configs/` and converts nested dictionaries into `SimpleNamespace` objects for simpler attribute access.

- **Model Initialization**:  
  Based on the selected checkpoint, the relevant model architecture (for example, located in `architecture/unet_cqt_oct_with_projattention_adaLN_2.py`) is imported, and the model is loaded with pretrained weights.

- **Audio Preprocessing**:  
  Uses `torchaudio` to load and resample the audio if needed, ensuring the input is mono and matches the expected length through trimming or padding.

- **Spectrogram Masking and Inpainting**:  
  The app generates a spectrogram from the audio, masks its central portion, and then performs inpainting using the GAN model, controlled by a user-specified noise level (sigma).

- **Result Generation**:  
  The inpainted audio is reconstructed and provided through the Streamlit interface, with options for playback and download.

## Customization

### Adding a New GAN Architecture

1. **Create a New Model File**:  
   Place your new model architecture as a Python file within the `architecture/` directory.

2. **Update Configuration**:  
   Provide a corresponding YAML configuration file in the `configs/` directory with model-specific parameters.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

AudioGANs is inspired by various GAN implementations and aims to offer a flexible platform for audio-based GAN research and development.

---

For additional details and updates, visit the [AudioGANs GitHub repository](https://github.com/Rohan-ingle/AudioGANs).
