# Genii Brain Weight

A machine learning project for financial market prediction and decision-making using world models, actor-critic reinforcement learning, and temporal fusion transformers.

## Overview

This repository contains trained model checkpoints and training infrastructure for a sophisticated financial AI system that combines:

- **World Models** - Learn and predict market dynamics
- **Actor-Critic Networks** - Make optimal trading decisions
- **Temporal Fusion Transformer (TFT)** - Advanced time-series forecasting
- **Financial Data** - Historical OHLCV data for multiple assets

## Project Structure

```
.
├── brain/                          # Main project structure
│   ├── src/                        # Source code directory
│   ├── colab/
│   │   └── train_brain.ipynb      # Training notebook for Colab
│   ├── data/
│   │   ├── models/                # Trained model checkpoints
│   │   │   ├── world_model_ep*.pt # World model at different epochs
│   │   │   └── world_model_best.pt
│   │   └── raw/                   # Raw market data
│   │       ├── adanient/
│   │       ├── asianpaint/
│   │       ├── axisbank/
│   │       ├── banknifty/
│   │       ├── bhartiartl/
│   │       ├── hcltech/
│   │       ├── hdfcbank/
│   │       ├── icicibank/
│   │       ├── infy/
│   │       ├── itc/
│   │       ├── kotakbank/
│   │       ├── lt/
│   │       ├── maruti/
│   │       ├── reliance/
│   │       ├── sbin/
│   │       ├── sunpharma/
│   │       ├── tatasteel/
│   │       ├── tcs/
│   │       ├── wipro/             # Indian equity markets
│   │       ├── banknifty/
│   │       ├── nifty50/           # Indian indices
│   │       ├── crude_oil/
│   │       ├── gold/
│   │       ├── silver/            # Commodities
│   │       ├── dow/
│   │       ├── nasdaq/
│   │       ├── sp500/             # US markets
│   │       ├── dxy/
│   │       ├── usdinr/
│   │       ├── us10y/             # FX and rates
│   │       └── vix/
│   ├── verify_weights.py          # Model weight verification script
│   └── requirements.txt
├── data/                          # Additional data directory
│   └── raw/                       # Mirror of market data
├── train_brain.ipynb              # Main training notebook
├── tft_model.pt                   # Trained TFT model
├── actor_critic.pt                # Trained actor-critic network
├── world_model_latest.pt          # Latest world model checkpoint
├── processed_features.npz         # Pre-processed feature data
├── training_history.json          # Training metrics
└── RESUME_INSTRUCTIONS.txt        # Instructions for resuming training
```

## Models

### World Model
- **Purpose**: Learn latent representations of market dynamics
- **Files**: 
  - `world_model_best.pt` - Best performing checkpoint
  - `world_model_latest.pt` - Most recent checkpoint
  - `world_model_ep*.pt` - Checkpoints at specific epochs

### Actor-Critic Network
- **Purpose**: Learn optimal trading policies
- **Files**:
  - `actor_critic.pt` - Trained network weights
  - `actor_critic_initial.pt` - Initial checkpoint for reference

### Temporal Fusion Transformer (TFT)
- **Purpose**: Forecast future price movements
- **Files**:
  - `tft_model.pt` - Trained model weights
  - `tft_model_initial.pt` - Initial checkpoint

## Data

Market data includes 1-day OHLCV (Open, High, Low, Close, Volume) for:

**Indian Equities**: ADANIENT, ASIANPAINT, AXISBANK, BAJFINANCE, BHARTIARTL, HCLTECH, HDFCBANK, ICICIBANK, INFY, ITC, KOTAKBANK, LT, MARUTI, RELIANCE, SBIN, SUNPHARMA, TATASTEEL, TCS, WIPRO

**Indian Indices**: BANKNIFTY, NIFTY50

**Commodities**: CRUDE_OIL, GOLD, SILVER

**US Markets**: DOW, NASDAQ, SP500

**FX & Rates**: DXY, USDINR, US10Y, VIX

## Training

### Setup
```bash
pip install -r brain/requirements.txt
```

### Training a Model
Use the provided Jupyter notebooks:
- `train_brain.ipynb` - Main training interface
- `brain/colab/train_brain.ipynb` - Colab-enabled version

### Resume Training
See `RESUME_INSTRUCTIONS.txt` for step-by-step instructions on resuming training from the latest checkpoint.

## Key Files

- `training_history.json` - Logs of training metrics and loss curves
- `processed_features.npz` - Preprocessed feature matrices
- `architecture_info.json` - Model architecture specifications
- `tft_split_data.npz` - Train-test split data for TFT

## Usage

1. **Load a trained model**: Import from the checkpoint files
2. **Make predictions**: Use the world model for environment simulation
3. **Generate trading signals**: Use the actor-critic network for decisions
4. **Forecast prices**: Use TFT for multi-step forecasting

## Notes

- All model checkpoints are stored using Git LFS
- Training data is organized by asset class for easy access
- Jupyter notebooks are available for interactive exploration
- All weights and data are included for reproducible inference

## Contents

- **87 files** total
- **360 MB** LFS objects (large model weights and data files)
- Comprehensive training history and intermediate checkpoints

---

*Last Updated: March 7, 2026*
