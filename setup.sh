#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install jupyter
pip install numpy
pip install pandas
pip install nibabel
pip install matplotlib
pip install scikit-image
pip install scipy

# Create Jupyter kernel for this environment
python -m ipykernel install --user --name=mri_preprocessing

echo "Setup complete! You can now activate the environment with:"
echo "source venv/bin/activate"
echo "And start Jupyter with:"
echo "jupyter notebook" 