# Installation Guide

This guide provides detailed instructions for installing and setting up the Smart Contract Decompilation system.

## Prerequisites

Before installing, ensure your system meets these requirements:

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with 8GB VRAM | NVIDIA A100 (40GB) or equivalent |
| **System RAM** | 16GB | 32GB+ |
| **Storage** | 50GB free space | 200GB+ (for full dataset) |
| **CPU** | 4 cores | 16+ cores for data processing |

### Software Requirements

- **Python**: Version 3.8 or higher
- **CUDA**: Compatible version for your GPU (11.8+ recommended)
- **Git**: For cloning the repository
- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### API Access

You'll need API keys for:

- **Etherscan API**: For collecting verified contracts ([Get API key](https://etherscan.io/apis))
- **Hugging Face**: For accessing Llama 3.2 3B model ([Get token](https://huggingface.co/settings/tokens))

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd A1.5_Smart_Contract_Bytecode_To_Code_Generator
```

### Step 2: Set Up Python Environment

#### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Option B: Using conda

```bash
# Create conda environment
conda create -n smart-contract-decompiler python=3.10
conda activate smart-contract-decompiler
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

#### Dependencies Overview

The main dependencies include:

- **transformers**: Hugging Face transformers library
- **torch**: PyTorch for deep learning
- **peft**: Parameter-Efficient Fine-Tuning (LoRA)
- **bitsandbytes**: 4-bit quantization support
- **sentence-transformers**: For semantic similarity evaluation
- **web3**: Ethereum interaction
- **requests**: API communication

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env
```

Add the following configuration:

```bash
# Etherscan API Configuration
ETHERSCAN_API_KEY=your_etherscan_api_key_here

# Hugging Face Configuration
HF_TOKEN=your_huggingface_token_here

# Optional: Model Configuration
MODEL_NAME=meta-llama/Llama-3.2-3B
MODEL_CACHE_DIR=./models/cache

# Optional: Data Configuration
DATA_DIR=./data
OUTPUT_DIR=./output
```

#### Setting Environment Variables (Alternative)

If you prefer not to use a `.env` file:

**Linux/macOS:**

```bash
export ETHERSCAN_API_KEY="your_etherscan_api_key"
export HF_TOKEN="your_huggingface_token"
```

**Windows (Command Prompt):**

```cmd
set ETHERSCAN_API_KEY=your_etherscan_api_key
set HF_TOKEN=your_huggingface_token
```

**Windows (PowerShell):**

```powershell
$env:ETHERSCAN_API_KEY="your_etherscan_api_key"
$env:HF_TOKEN="your_huggingface_token"
```

### Step 5: Verify Installation

Run the demonstration script to validate your installation:

```bash
python demo.py
```

Expected output:

```text
=== Smart Contract Decompilation Demo ===
Testing bytecode analysis...
✓ Bytecode analysis successful
Testing TAC generation...
✓ TAC generation successful
Testing dataset format...
✓ Dataset format valid
All tests passed!
```

## GPU Setup

### CUDA Installation

#### Linux (Ubuntu/Debian)

```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### Verify CUDA Installation

```bash
nvidia-smi
nvcc --version
```

### PyTorch GPU Support

Verify PyTorch can access your GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Directory Structure Setup

The system will automatically create required directories, but you can set them up manually:

```bash
# Create data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/train
mkdir -p data/val
mkdir -p data/test

# Create model directories
mkdir -p models/checkpoints
mkdir -p models/final
mkdir -p models/cache

# Create output directories
mkdir -p output/logs
mkdir -p output/evaluation
```

## Obtaining API Keys

### Etherscan API Key

1. Visit [Etherscan](https://etherscan.io)
2. Create an account or log in
3. Navigate to [API Keys](https://etherscan.io/myapikey)
4. Create a new API key
5. Copy the key and add to your `.env` file

**Rate Limits**: Free tier allows 5 calls/second, 100,000 calls/day

### Hugging Face Token

1. Visit [Hugging Face](https://huggingface.co)
2. Create an account or log in
3. Navigate to [Access Tokens](https://huggingface.co/settings/tokens)
4. Create a new token with read access
5. Accept Llama 3.2 model license at [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
6. Copy the token and add to your `.env` file

## Troubleshooting Installation Issues

### Issue: CUDA Out of Memory

**Solution:**

```bash
# Enable 4-bit quantization (requires less VRAM)
# This is configured by default in the training scripts
```

### Issue: PyTorch Installation Fails

**Solution:**

```bash
# Install specific PyTorch version for your CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Transformers Library Error

**Solution:**

```bash
# Update to latest version
pip install --upgrade transformers accelerate
```

### Issue: Permission Denied for Model Download

**Solution:**

```bash
# Ensure HF_TOKEN is set correctly
huggingface-cli login
# Enter your token when prompted
```

### Issue: Etherscan API Rate Limiting

**Solution:**

- Reduce parallel workers in data collection
- Add delays between requests
- Consider Etherscan Pro subscription for higher limits

## Updating the Installation

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Re-run verification
python demo.py
```

## Uninstallation

To completely remove the installation:

```bash
# Deactivate virtual environment
deactivate  # or conda deactivate

# Remove virtual environment
rm -rf venv  # or conda env remove -n smart-contract-decompiler

# Remove data and models (optional)
rm -rf data/
rm -rf models/
rm -rf output/

# Remove repository
cd ..
rm -rf A1.5_Smart_Contract_Bytecode_To_Code_Generator
```

## Next Steps

After successful installation:

1. Review the [Usage Guide](usage.md) for basic usage examples
2. Check the [Training Pipeline](training-pipeline.md) for model training
3. Explore [Data Format](data-format.md) for dataset specifications
4. See [Architecture](architecture.md) for system design details

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Etherscan API Documentation](https://docs.etherscan.io/)
