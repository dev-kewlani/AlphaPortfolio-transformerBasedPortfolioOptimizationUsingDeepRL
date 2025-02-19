# AlphaPortfolio-transformerBasedPortfolioOptimizationUsingDeepRL

## Project Overview
This repository implements a reinforcement learning framework for portfolio management using a Transformer-based neural architecture. The model processes historical financial data (prices, returns, volume, market cap) to generate long/short portfolio weights, optimizing for the Sharpe ratio across sequential rebalancing periods. Key components include:

- **Temporal Processing**: Transformer encoder for time-series dependencies
- **Cross-Asset Attention**: CAAN module for inter-asset relationships
- **Sequential Training**: Episode-based RL with delayed Sharpe ratio reward
- **Data Pipeline**: Integrated WRDS data fetching and preprocessing

## Repository Structure


## Key Files Description

### 1. Main Components
- **main.py**: Orchestrates data loading, model initialization, and training
- **model_architecture.py**: Contains `AlphaPortfolioModel` class with:
  - Input projection layer
  - Transformer encoder for temporal patterns
  - Cross-Asset Attention Network (CAAN)
  - Portfolio generator with long/short selection

- **data_pipeline.py**: 
  - `AlphaPortfolioData` class handling:
  - WRDS data fetching from CRSP/Compustat
  - Feature engineering (returns, market cap, etc.)
  - Sequential episode generation with lookback windows

- **training_model.py**:
  - `train_model_sequential()` implementing:
  - Episode-based RL training loop
  - Sharpe ratio calculation & optimization
  - Gradient updates with Adam optimizer

### 2. Support Components
- **plotting_functions_for_convergence.py**:
  - Visualizes training progress through:
  - `plot_epoch_sharpe()`: Epoch-wise Sharpe convergence
  - `plot_episode_sharpe()`: Intra-epoch performance

- **imports.py**:
  - Centralized dependency management
  - WRDS connection setup
  - Logging configuration (file + console)

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- WRDS account (University/Institutional license)
  
### 2. Installation
```bash
# Clone repository
git clone https://github.com/yourusername/AlphaPortfolio.git
cd AlphaPortfolio

# Create conda environment
conda create -n alphaportfolio python=3.8
conda activate alphaportfolio

# Install dependencies
pip install -r requirements.txt

#WRDS Configuration
#Create ~/.wrds_profile with your credentials:
[wrds]
username = your_wrds_id
password = your_wrds_password

#Test Connection
import wrds
db = wrds.Connection()  # Should connect successfully

#Modify hyperparameters in main.py
# Core parameters
lookback = 12          # Historical window (months)
T = 12                 # Rebalancing steps per episode
model_G = 10           # Assets to long/short
num_epochs = 15        # Training epochs

#Start Training
python main.py
```

## Outputs:

- Training logs: training.log
- Convergence plots: plots/ directory
- Model checkpoints: Saved automatically (add checkpointing code as needed)
