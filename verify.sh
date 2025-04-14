#!/usr/bin/env bash
# verify.sh

########################################
# 1. Check for deploy.py
########################################
if [ -f "deploy.py" ]; then
    echo "✓ 'deploy.py' found."
else
    echo "✗ ERROR: 'deploy.py' is missing!"
fi

########################################
# 2. Check for dependencies.sh
########################################
if [ -f "dependencies.sh" ]; then
    echo "✓ 'dependencies.sh' found."
else
    echo "✗ ERROR: 'dependencies.sh' is missing!"
fi

########################################
# 3. Check architecture/ and its file
########################################
if [ -d "architecture" ]; then
    echo "✓ 'architecture/' directory exists."
    if [ -f "architecture/unet_cqt_oct_with_projattention_adaLN_2.py" ]; then
        echo "✓ 'unet_cqt_oct_with_projattention_adaLN_2.py' found in 'architecture/'."
    else
        echo "✗ ERROR: 'unet_cqt_oct_with_projattention_adaLN_2.py' not found in 'architecture/'."
    fi
else
    echo "✗ ERROR: 'architecture/' directory does NOT exist!"
fi

########################################
# 4. Check configs/ and its files
########################################
if [ -d "configs" ]; then
    echo "✓ 'configs/' directory exists."
    if [ -f "configs/maestro.yaml" ]; then
        echo "✓ 'maestro.yaml' found in 'configs/'."
    else
        echo "✗ ERROR: 'maestro.yaml' not found in 'configs/'."
    fi
    if [ -f "configs/musicnet.yaml" ]; then
        echo "✓ 'musicnet.yaml' found in 'configs/'."
    else
        echo "✗ ERROR: 'musicnet.yaml' not found in 'configs/'."
    fi
else
    echo "✗ ERROR: 'configs/' directory does NOT exist!"
fi

########################################
# 5. Check defaults/ and 1727.wav
########################################
if [ -d "defaults" ]; then
    echo "✓ 'defaults/' directory exists."
    if [ -f "defaults/1727.wav" ]; then
        echo "✓ '1727.wav' found in 'defaults/'."
    else
        echo "✗ ERROR: '1727.wav' not found in 'defaults/'."
    fi
else
    echo "✗ ERROR: 'defaults/' directory does NOT exist!"
fi

########################################
# 6. Check models/ directory and model files
#    If missing or incomplete, auto-run dependencies.sh --both
########################################

MODELS_DIR="models"
NEED_DOWNLOAD=false

check_and_download_models () {
    echo "INFO: Attempting to download both Maestro and Musicnet models..."
    if [ -f "./dependencies.sh" ]; then
        chmod +x ./dependencies.sh
        ./dependencies.sh --both
        echo "INFO: Download script completed."
    else
        echo "✗ ERROR: 'dependencies.sh' not found, cannot download models!"
    fi
}

# Confirm models/ directory
if [ ! -d "$MODELS_DIR" ]; then
    echo "✗ WARNING: '$MODELS_DIR/' directory does not exist. Creating it..."
    mkdir -p "$MODELS_DIR"
    NEED_DOWNLOAD=true
else
    echo "✓ '$MODELS_DIR/' directory exists."
fi

# Check for the two model files
MAESTRO_PT="$MODELS_DIR/maestro_22k_8s-750000.pt"
MUSICNET_PT="$MODELS_DIR/musicnet_44k_4s-560000.pt"

if [ -f "$MAESTRO_PT" ]; then
    echo "✓ 'maestro_22k_8s-750000.pt' found in '$MODELS_DIR/'."
else
    echo "✗ WARNING: 'maestro_22k_8s-750000.pt' not found in '$MODELS_DIR/'."
    NEED_DOWNLOAD=true
fi

if [ -f "$MUSICNET_PT" ]; then
    echo "✓ 'musicnet_44k_4s-560000.pt' found in '$MODELS_DIR/'."
else
    echo "✗ WARNING: 'musicnet_44k_4s-560000.pt' not found in '$MODELS_DIR/'."
    NEED_DOWNLOAD=true
fi

# Download if needed
if [ "$NEED_DOWNLOAD" = true ]; then
    check_and_download_models
fi

echo "Verification complete."
