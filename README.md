# Part 1: Neural Network Fundamentals and Training Behavior Analysis

**BITSOM BA - Module 5 | Applied Neural Networks**

## Overview
This project builds and analyzes a feed-forward neural network to predict **customer churn** using a structured tabular dataset. The focus is on understanding how neural networks learn — through forward pass, loss calculation, backpropagation, and parameter updates.

## Dataset
- **File:** `customer_churn_nn.csv`
- **Rows:** 2000 customers
- **Target:** `churn` (1 = churned, 0 = retained)
- **Features:** tenure, monthly charges, login days, support tickets, satisfaction score, region, plan type, contract type, payment method, and more.

## Repository Structure
```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb           # Complete analysis notebook
├── requirements.txt         # Python dependencies
└── results/
    ├── model_comparison_table.csv   # Hyperparameter experiment results
    ├── model_comparison_table.png   # Visual comparison chart
    └── evaluation_outputs.png       # Training curves & confusion matrix
```

## Tasks Completed
| Task | Description | Marks |
|------|-------------|-------|
| Task 1 | Dataset understanding and exploration | 4 |
| Task 2 | Data preprocessing (encoding, scaling, split) | 4 |
| Task 3 | Feed-forward neural network model building | 5 |
| Task 4 | Training, evaluation, confusion matrix | 4 |
| Task 5 | Hyperparameter experimentation (4 configs) | 5 |
| Task 6 | Final reflection on weights, activations, LR, overfitting | 3 |

## Model Architecture (Baseline)
- Input layer → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=0.001)
- Epochs: 50, Batch Size: 32

## Hyperparameter Experiments
| Experiment | Hidden Layers | LR | Activation | Batch |
|---|---|---|---|---|
| Baseline | [64, 32] | 0.001 | ReLU | 32 |
| Deeper Network | [128, 64, 32] | 0.001 | ReLU | 32 |
| Higher LR | [64, 32] | 0.01 | ReLU | 32 |
| tanh Activation | [64, 32] | 0.001 | tanh | 64 |

## How to Run
1. Open `notebook.ipynb` in Google Colab
2. Upload `customer_churn_nn.csv` when prompted
3. Run all cells in order

## Requirements
```
tensorflow>=2.10
scikit-learn>=1.0
pandas>=1.4
numpy>=1.21
matplotlib>=3.5
seaborn>=0.11
```
