# AlphaPortfolio-transformerBasedPortfolioOptimizationUsingDeepRL

## Project Overview
Alpha Portfolio is a deep learning-based portfolio management system that uses transformer architecture to optimize investment strategies. The system processes historical market data to generate dynamic portfolio weights for long/short equity positions, aiming to maximize risk-adjusted returns measured by the Sharpe ratio.

### Key Features
- Deep learning model combining Transformer encoders and Cross-Asset Attention Network (CAAN)
- Sequential portfolio optimization with reinforcement learning principles
- Integration with WRDS (Wharton Research Data Services) for financial data
- Automated portfolio weight generation for long/short positions
- Performance visualization and convergence tracking

## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
```
alpha-portfolio/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── src/
│   ├── imports.py
│   ├── data_pipeline.py
│   ├── model_architecture.py
│   ├── training_model.py
│   └── plotting_functions_for_convergence.py
├── tests/
│   └── __init__.py
├── logs/
│   └── .gitkeep
└── plots/
    └── .gitkeep
```

### File Descriptions
- `imports.py`: Central file for all required dependencies and logging setup
- `data_pipeline.py`: Handles data loading and preprocessing from WRDS
- `model_architecture.py`: Implements the deep learning model architecture
- `training_model.py`: Contains the training loop and optimization logic
- `plotting_functions_for_convergence.py`: Visualization utilities for model convergence

## Prerequisites
- Python 3.8+
- WRDS account with access to CRSP and Compustat databases
- PyTorch 1.9+
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AlphaPortfolio-transformerBasedPortfolioOptimizationUsingDeepRL.git
cd AlphaPortfolio-transformerBasedPortfolioOptimizationUsingDeepRL
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up WRDS credentials:
```bash
export WRDS_USERNAME=your_username
export WRDS_PASSWORD=your_password
```

## Usage

1. Configure your parameters in **main.py** file:
```bash
training:
  lookback: 12
  start_year: 2017
  final_year: 2018
  end_year: 2020
  batch_size: 1
  num_epochs: 15
  learning_rate: 1e-4

model:
  d_model: 16
  nhead: 2
  num_encoder_layers: 1
  d_attn: 8
  G: 10
```

2. Run the script:
```bash
python src/main.py
```

## Data Pipeline

The data pipeline (`data_pipeline.py`) handles:
- Fetching historical market data from WRDS
- Creating sequential episodes for reinforcement learning
- Processing features including returns, prices, volume, market cap, and sales
- Implementing data masks for handling missing values

## Model Architecture

The model (`model_architecture.py`) consists of:
1. Input projection layer
2. Transformer encoder for temporal dependencies
3. Cross-Asset Attention Network (CAAN)
4. Portfolio generator for selecting top/bottom assets

## Training Process

The training process (`training_model.py`):
1. Processes sequential episodes with T rebalancing steps
2. Computes portfolio weights for each time step
3. Calculates forward returns and portfolio performance
4. Optimizes for maximum Sharpe ratio

## Configuration

Key configuration parameters:
- `lookback`: Historical window length (default: 12 months)
- `T`: Number of rebalancing steps per episode
- `model_G`: Number of assets selected for long/short positions
- `d_model`: Dimension of the model's hidden states
- `nhead`: Number of attention heads in transformer
- `num_encoder_layers`: Number of transformer encoder layers

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/SomeFeatures`)
3. Commit your changes (`git commit -m 'Add some SomeFeatures'`)
4. Push to the branch (`git push origin feature/SomeFeatures`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
