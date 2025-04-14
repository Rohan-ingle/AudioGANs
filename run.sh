#!/bin/bash
# run_deploy.sh - Script to launch the Streamlit app

# Check if deploy.py exists in the current directory
if [ ! -f "deploy.py" ]; then
  echo "Error: deploy.py not found in the current directory."
  exit 1
fi

# Print a starting message
echo "Starting Streamlit application..."

# Run the Streamlit app
streamlit run deploy.py
