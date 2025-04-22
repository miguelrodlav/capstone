#!/bin/bash
# Setup script for Mac users (especially Apple Silicon M-series)

echo "Setting up Python environment for Mac..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install PyTorch for Mac (with MPS support)
echo "Installing PyTorch with MPS support for Apple Silicon..."
pip install torch torchvision

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify MPS availability
echo "Verifying MPS (Metal) availability..."
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

echo ""
echo "Setup complete! You can now run the server with:"
echo "source venv/bin/activate  # If not already activated"
echo "python app.py" 