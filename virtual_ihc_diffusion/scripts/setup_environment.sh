#!/bin/bash

# Setup script for Virtual IHC Diffusion project
# Run this script once to set up the environment on MIT ORCD

echo "=================================="
echo "Virtual IHC Staining - Environment Setup"
echo "=================================="

# Create virtual environment
echo "Creating virtual environment..."
python -m venv ~/venv_ihc
source ~/venv_ihc/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify installations
echo "=================================="
echo "Verifying installations..."
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo ""
echo "MONAI version:"
python -c "import monai; print(monai.__version__)"
echo "=================================="

# Create necessary directories
echo "Creating project directories..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs

echo "=================================="
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source ~/venv_ihc/bin/activate"
echo ""
echo "To train the model:"
echo "  sbatch scripts/submit_job.sh"
echo "=================================="
