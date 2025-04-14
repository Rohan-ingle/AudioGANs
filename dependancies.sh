#!/bin/bash
# download_models.sh: Download Maestro and/or Musicnet models and install Python dependencies.

# URLs for the models from Hugging Face
MAESTRO_URL="https://huggingface.co/Eloimoliner/audio-inpainting-diffusion/resolve/main/maestro_22k_8s-750000.pt"
MUSICNET_URL="https://huggingface.co/Eloimoliner/audio-inpainting-diffusion/resolve/main/musicnet_44k_4s-560000.pt"

# Directory where the models will be saved
DOWNLOAD_DIR="./models"

# Function to install Python dependencies
install_requirements() {
    echo "Installing Python dependencies from requirements.txt..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        if [ $? -eq 0 ]; then
            echo "Dependencies installed successfully."
        else
            echo "Error installing dependencies."
        fi
    else
        echo "requirements.txt not found in the current directory."
    fi
}

# Function to download the Maestro model
download_maestro() {
    echo "Downloading Maestro model..."
    wget -O "${DOWNLOAD_DIR}/maestro_22k_8s-750000.pt" "$MAESTRO_URL"
    if [ $? -eq 0 ]; then
        echo "Maestro model downloaded successfully."
    else
        echo "Error downloading Maestro model."
    fi
}

# Function to download the Musicnet model
download_musicnet() {
    echo "Downloading Musicnet model..."
    wget -O "${DOWNLOAD_DIR}/musicnet_44k_4s-560000.pt" "$MUSICNET_URL"
    if [ $? -eq 0 ]; then
        echo "Musicnet model downloaded successfully."
    else
        echo "Error downloading Musicnet model."
    fi
}

# Function to display usage
usage() {
    echo "Usage: $0 [--maestro | --musicnet | --both]"
    exit 1
}

# Ensure the models directory exists
mkdir -p "$DOWNLOAD_DIR"

# Install Python dependencies
install_requirements

# Check if a command line argument is provided
if [ "$#" -eq 1 ]; then
    case "$1" in
        --maestro)
            download_maestro
            ;;
        --musicnet)
            download_musicnet
            ;;
        --both)
            download_maestro
            download_musicnet
            ;;
        *)
            usage
            ;;
    esac
else
    # No arguments: interactive mode
    echo "Select which model to download:"
    echo "1) Maestro"
    echo "2) Musicnet"
    echo "3) Both"
    read -rp "Enter choice [1-3]: " choice

    case "$choice" in
        1)
            download_maestro
            ;;
        2)
            download_musicnet
            ;;
        3)
            download_maestro
            download_musicnet
            ;;
        *)
            echo "Invalid option. Exiting."
            usage
            ;;
    esac
fi