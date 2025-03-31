# AlphaPortfolio: Deep Reinforcement Learning for Dynamic Portfolio Optimization

![Portfolio Optimization](https://img.shields.io/badge/Portfolio-Optimization-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Transformer-green)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-Finance-orange)

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Results Interpretation](#results-interpretation)
- [Visualization Features](#visualization-features)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

AlphaPortfolio is a sophisticated deep learning framework for dynamic portfolio optimization using transformer-based architectures and reinforcement learning principles. The model is designed to learn optimal portfolio allocation strategies directly from financial data, adapting to changing market conditions while maximizing risk-adjusted returns.

This implementation provides a comprehensive pipeline for training and evaluating portfolio optimization models, including:

- Data preprocessing and normalization
- Episodic training approach for reinforcement learning
- Both episodic and sequential monthly testing methodologies
- Extensive visualization tools for performance analysis
- Transaction cost modeling
- Benchmark comparison

The framework is built on PyTorch and designed for flexibility, allowing researchers and practitioners to experiment with different model architectures, hyperparameters, and evaluation methodologies.

## Methodology

### Theoretical Background

AlphaPortfolio combines several key innovations:

1. **Sequence Representation Extraction Model (SREM)**: Uses transformer encoders to process temporal patterns in asset features.
2. **Cross-Asset Attention Network (CAAN)**: Models interactions between assets through attention mechanisms.
3. **Reinforcement Learning**: Optimizes for long-term Sharpe ratio rather than one-step returns.
4. **Portfolio Generator**: Constructs portfolios by selecting top and bottom assets based on model scores.

The model processes historical financial features for each asset, produces allocation weights, and optimizes these weights to maximize risk-adjusted returns.

### Training Approach

The training methodology employs an episodic reinforcement learning approach:

1. **Episodes**: Each episode consists of T consecutive time steps
2. **States**: Each state contains lookback periods of financial features for all assets
3. **Actions**: Portfolio weights for each asset
4. **Reward**: Sharpe ratio of portfolio returns over the episode

This approach allows the model to learn long-term dependencies and optimize for risk-adjusted returns rather than just maximizing returns.

### Evaluation Methodologies

Two distinct evaluation approaches are implemented:

1. **Episodic Testing**: Evaluates the model on episodes similar to training, useful for validating learning.
2. **Monthly Sequential Testing**: A more realistic evaluation that tests the model in sequential month-by-month fashion with transaction costs, replicating real-world portfolio management.

## Project Structure

```
Complete AlphaPortfolio Pipeline/
├── config.yaml                    # Configuration file
├── config.py                      # Configuration loading module
├── data.py                        # Data loading and preprocessing
├── model.py                       # Neural network model implementation
├── main.py                        # Main execution script
├── enhanced_training.py           # Training pipeline implementation
├── enhanced_visualization.py      # Visualization utilities
├── monthly_test_evaluator.py      # Sequential monthly testing
├── sequential_data_loader.py      # Sequential data loading utilities
├── test.py                        # Episodic testing implementation
├── Results/                       # Directory for results
│   ├── logs/                      # Log files
│   ├── models/                    # Saved models
│   ├── output/                    # Output files
│   ├── plots/                     # Plot files
│   └── training_data.csv          # Training data
└── README.md                      # This file
```

## Implementation Details

### Core Components

#### 1. Model Architecture (`model.py`)

- **AlphaPortfolio**: Main model class integrating all components
- **SequenceRepresentationExtractionModel (SREM)**: Transformer-based module for processing temporal patterns
- **CrossAssetAttentionNetwork (CAAN)**: Attention mechanism for modeling asset interactions
- **PortfolioGenerator**: Constructs portfolios from model scores

The architecture follows the design described in the AlphaPortfolio paper, with enhancements for stability and performance.

#### 2. Data Processing (`data.py`)

- **AlphaPortfolioData**: Dataset class for loading and preprocessing financial data
- **FeatureScaler**: Normalization of financial features
- **Sequence creation**: Generates episodes with lookback periods for training

The data pipeline handles caching for efficiency and supports both episodic and sequential data retrieval.

#### 3. Training (`enhanced_training.py`)

- **RLTrainer**: Reinforcement learning trainer
- **TrainingManager**: Manages training process, checkpointing, visualization
- **Batch and Episode Tracking**: Monitors performance metrics during training

The training system optimizes the model to maximize risk-adjusted returns using a reinforcement learning approach.

#### 4. Testing

- **TestEvaluator** (`test.py`): Episodic testing similar to training setup
- **MonthlyTestEvaluator** (`monthly_test_evaluator.py`): Sequential monthly rebalancing evaluation

Both evaluation methods provide comprehensive performance metrics and visualizations.

#### 5. Visualization

- **VisualizationManager**: Creates visualizations of model performance
- Comprehensive visualizations at episode, batch, and epoch levels
- Enhanced date handling for clear timeline representation

### Key Files Explained

- **config.yaml**: Central configuration file with all parameters
- **config.py**: Loads and validates configuration
- **data.py**: Data loading, normalization, and episode creation
- **model.py**: Neural network model implementation
- **main.py**: Main execution script with command-line interface
- **enhanced_training.py**: Training pipeline implementation
- **enhanced_visualization.py**: Visualization utilities
- **monthly_test_evaluator.py**: Sequential monthly testing implementation
- **sequential_data_loader.py**: Utilities for sequential data loading
- **test.py**: Episodic testing implementation

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AlphaPortfolio.git
   cd AlphaPortfolio
   ```

2. Create a virtual environment:
   ```bash
   # Using conda
   conda create -n alphaportfolio python=3.10
   conda activate alphaportfolio
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If requirements.txt is not available, install these core packages:
   ```bash
   pip install torch numpy pandas matplotlib seaborn tqdm pyyaml scikit-learn
   ```

4. Set up data directories:
   ```bash
   mkdir -p Results/logs Results/models Results/output Results/plots
   ```

5. Prepare your data:
   - Place your financial data CSV in the `Results/` directory as `training_data.csv` (I have uploaded sample cleaned data from 1975-2000)
   - Ensure your data has columns for asset identifiers (permno), date, and features

## Usage

### Configuration

Edit `config.yaml` to set parameters:

```yaml
# Basic experiment settings
experiment_id: alpha_portfolio_enhanced
description: AlphaPortfolio with Transformer and RL

# Paths
paths:
  output_dir: ./Results/output
  model_dir: ./Results/models
  log_dir: ./Results/logs
  plot_dir: ./Results/plots
  data_path: ./Results/training_data.csv

# Model parameters
model:
  T: 12                # Number of time steps per episode
  lookback: 12         # Lookback period for historical data
  d_model: 128         # Model dimension
  nhead: 4             # Number of attention heads
  num_layers: 2        # Number of transformer layers
  G: 15                # Number of assets to include in long/short portfolios

# Training parameters
training:
  num_epochs: 10
  batch_size: 2
  learning_rate: 1e-4
  weight_decay: 5e-5
  
# Testing parameters
test:
  transaction_cost_rate: 0.001
  compare_benchmark: true
  
# Training cycles
cycles:
  - cycle_idx: 0
    train_start: "1975-01-01"
    train_end: "2000-12-31"
    test_start: "2001-01-01"
    test_end: "2010-12-31"
```

### Training

To train the model:

```bash
python main.py --mode train --config config.yaml --seed 42
```

This will:
1. Load data from the specified path
2. Preprocess and normalize the data
3. Train the model using reinforcement learning
4. Save checkpoints and the best model
5. Generate training visualizations

### Testing

#### Episodic Testing

To test using the episodic approach:

```bash
python main.py --mode test --config config.yaml --seed 42
```

#### Monthly Sequential Testing

For more realistic sequential monthly rebalancing evaluation:

```bash
python main.py --mode monthly_test --config config.yaml --seed 42
```

#### Run Everything

To run training and both testing approaches:

```bash
python main.py --mode full --config config.yaml --seed 42
```

### Command-Line Arguments

- `--config`: Path to configuration file (default: `config.yaml`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--mode`: Operation mode: `train`, `test`, `monthly_test`, or `full` (default: `full`)
- `--cycle`: Cycle index to run (default: `0`)
- `--output`: Custom output directory (optional)

## Results Interpretation

### Output Structure

```
Results/
├── logs/                           # Log files
├── models/
│   └── cycle_0/                    # Models for cycle 0
│       ├── best_train.pt           # Best model during training
│       ├── final.pt                # Final model after training
│       └── epoch_*.pt              # Checkpoints for each epoch
├── output/
│   ├── test_results/               # Episodic test results
│   │   ├── visualizations/         # Visualization plots
│   │   └── report/                 # Performance reports
│   └── monthly_test_results/       # Monthly test results
│       ├── visualizations/         # Visualization plots
│       └── report/                 # Performance reports
└── plots/
    └── cycle_0/                    # Training plots for cycle 0
        ├── epochs/                 # Plots for each epoch
        ├── batches/                # Plots for batches
        └── episodes/               # Plots for episodes
```

### Key Performance Metrics

1. **Sharpe Ratio**: Risk-adjusted return measure (higher is better)
2. **Mean Return**: Average portfolio return
3. **Cumulative Return**: Total return over test period
4. **Maximum Drawdown**: Largest peak-to-trough decline
5. **Position Counts**: Number of long and short positions
6. **Turnover**: Portfolio turnover rate

### Visualization Dashboards

1. **Performance Dashboard**: Comprehensive overview of performance metrics
2. **Monthly Returns**: Per-month return analysis
3. **Cumulative Returns**: Growth of investment over time
4. **Drawdowns**: Portfolio drawdowns analysis
5. **Portfolio Allocation**: Asset weights over time
6. **Position Analysis**: Long and short position distributions
7. **Risk Metrics**: Volatility, Sharpe ratio, and other risk measures
8. **Benchmark Comparison**: Performance against benchmark (if enabled)

## Visualization Features

The system generates extensive visualizations to help analyze model performance:

### Training Visualizations

- **Epoch-level**: Performance metrics across all episodes in an epoch
- **Batch-level**: Performance within batches
- **Episode-level**: Detailed analysis of individual episodes

### Testing Visualizations

- **Portfolio Returns**: Monthly and cumulative returns
- **Risk Metrics**: Standard deviation, drawdowns, Sharpe ratio
- **Portfolio Composition**: Asset weights and allocation changes
- **Rolling Metrics**: Rolling Sharpe ratio and other performance metrics
- **Benchmark Comparison**: Performance relative to benchmark

All visualizations include proper date formatting for clear timeline representation.

## Advanced Features

### Transaction Cost Modeling

The monthly testing methodology includes transaction cost modeling to provide realistic performance estimates:

```python
# Calculate transaction costs
weight_changes = np.abs(weights - prev_weights)
transaction_costs = np.sum(weight_changes) * self.transaction_cost_rate

# Apply to returns
net_return = portfolio_return - transaction_costs
```

### Benchmark Comparison

When enabled, the system compares portfolio performance against a benchmark:

```yaml
test:
  compare_benchmark: true
  benchmark_ticker: "SPY"  # Specify benchmark ticker
```

### Custom Data Support

You can use your own financial data by specifying it in the configuration:

```yaml
paths:
  data_path: /path/to/your/data.csv
```

The data format should include:
- `permno`: Asset identifier column
- `date`: Date column
- Feature columns for asset characteristics

## Troubleshooting

### Common Issues

#### MPS/CUDA Device Issues

Issue with PyTorch MPS (Apple Silicon) or CUDA tensors:

```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64.
```

**Solution**: Modify tensor creation to use float32:
```python
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
```

#### JSON Serialization Errors

Issue when saving metrics that contain non-serializable objects:

```
TypeError: Object of type device is not JSON serializable
```

**Solution**: Enhance the JSON encoder to handle PyTorch devices:
```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.device):
            return str(obj)
        # ...other conversions...
        return super().default(obj)
```

#### Memory Issues

If encountering memory errors during training:

**Solutions**:
- Reduce batch size in config.yaml
- Reduce model size (d_model, num_layers)
- Use data caching to improve efficiency

#### Data Loading Issues

If data loading fails:

**Solutions**:
- Check data path in config.yaml
- Ensure data format matches expected columns
- Check log files for specific error messages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings for all functions and classes
- Write unit tests for new features
- Update documentation for significant changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Note: This implementation is for research and educational purposes. Always perform your own due diligence before using any trading strategy in production.*
