# Kernel Ridge Regression: Video Engagement Prediction

**Course**: CS771 Introduction to Machine Learning (IIT Kanpur)  
**Instructor**: Purushottam Kar  
**Academic Year**: 2024-25  
**Author**: Mainak Sarkar

---

## 📋 Problem Statement

Implement **Kernel Ridge Regression (KRR)** for a video recommendation task. The goal is to predict video engagement based on:
- **Feature x**: Video length (scalar)
- **Feature z**: [z₁, z₂] - Additional content features
- **Target y**: Engagement/recommendation score

### Key Objectives
1. Implement custom polynomial kernel: K̃((x₁,z₁),(x₂,z₂)) = x₁·x₂·K_z(z₁,z₂) + bias
2. Perform hyperparameter tuning using K-Fold cross-validation
3. Find the optimal polynomial degree for the kernel
4. Evaluate and report model performance metrics

---

## 🏗️ Project Structure

```
kernel-ridge-regression/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── solution.py                        # Main solution code
├── notebooks/
│   └── analysis.ipynb                # Jupyter notebook with detailed analysis
├── data/
│   ├── public/                       # Public test data
│   │   ├── public_x_trn.txt
│   │   ├── public_x_tst.txt
│   │   ├── public_Z_trn.txt
│   │   ├── public_Z_tst.txt
│   │   ├── public_y_trn.txt
│   │   └── public_y_tst.txt
│   └── secret/                       # Secret test data (for evaluation)
│       ├── secret_x_trn.txt
│       ├── secret_x_tst.txt
│       ├── secret_Z_trn.txt
│       ├── secret_Z_tst.txt
│       ├── secret_y_trn.txt
│       └── secret_y_tst.txt
├── results/
│   ├── model_performance.txt         # Final model metrics
│   ├── hyperparameter_tuning.txt     # Grid search results
│   └── r2_scores_by_degree.pkl       # Pickled results
└── docs/
    └── approach.md                   # Detailed approach documentation
```

---

## 🎯 Approach

### 1. **Data Preprocessing**
- Load training and test data from `.txt` files
- Standardize features using `StandardScaler`
  - X (video length) normalized independently
  - Z (z₁, z₂) normalized using training statistics
- Ensure target y is 1-D array

### 2. **Kernel Design**
Implemented a composite kernel combining linear and polynomial interactions:
```
K̃((x₁,z₁),(x₂,z₂)) = x₁·x₂·K_z(z₁,z₂) + bias
```
Where:
- `K_z(z₁,z₂) = (z₁ᵀz₂ + coef0)^degree` (polynomial kernel on z)
- Bias term added for numerical stability

### 3. **Hyperparameter Tuning**
Grid search over:
- **Polynomial degree**: [1, 2, 3, 4, 5, 6]
- **Kernel coefficient (coef0)**: [0.0, 1.0]
- **Ridge regression alpha**: [1e-2, 1e-1, 1.0]
- **Bias term**: [1.0]
- **Validation strategy**: 4-fold cross-validation

### 4. **Model Evaluation**
- **Primary metric**: R² score (average over 5 trials)
- **Secondary metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE)
- **Optimal configuration**: degree=2, coef0=1.0, alpha=0.1, bias=1.0
- **Best R² Score**: 0.921075 ± 0.000047

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/mainak9093/kernel-ridge-regression.git
cd kernel-ridge-regression

# Install dependencies
pip install -r requirements.txt
```

### Running the Solution

```bash
# Run the main solution on public data
python solution.py --dataset public

# Evaluate baseline models and degree performance
python solution.py --dataset public --run-baselines --evaluate-degrees

# This will:
# 1. Load and preprocess data
# 2. Perform hyperparameter tuning
# 3. Train the final model
# 4. Evaluate on test set
# 5. Save results to results/ directory
```

### Running the Notebook

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/analysis.ipynb for detailed step-by-step analysis
```

### Running Tests

```bash
pytest tests
```

---

## 📊 Results

### Best Model Configuration
| Parameter | Value |
|-----------|-------|
| Polynomial Degree | 2 |
| Kernel Coef (coef0) | 1.0 |
| Ridge Regularization (alpha) | 0.1 |
| Bias Term | 1.0 |

### Performance Metrics
| Metric | Value |
|--------|-------|
| **R² Score** | 0.921075 |
| **MSE** | ~0.000159 |
| **MAE** | ~0.0093 |
| **Avg Kernel Build Time** | 1.685s |
| **Avg Training Time** | 3.012s |

### Polynomial Degree Comparison
```
Degree 1: R² = 0.920982
Degree 2: R² = 0.921075  ✓ BEST
Degree 3: R² = 0.921075
Degree 4: R² = 0.921058
Degree 5: R² = 0.920889
Degree 6: R² = 0.920712
```

---

## 📚 Key Implementation Details

### Kernel Function
```python
def K_tilde(X1, Z1, X2, Z2, degree=3, coef0=1.0, bias=1.0):
    """
    Combined kernel: K̃((x1,z1),(x2,z2)) = x1*x2*Kz(z1,z2) + bias
    """
    Kz = polynomial_kernel(Z1, Z2, degree=degree, coef0=coef0)
    outer_x = np.outer(X1, X2)
    return outer_x * Kz + bias
```

### Gram Matrix Construction
```python
# Training Gram matrix
K_train_train = build_train_train_gram(X_train, Z_train, 
                                       degree=2, coef0=1.0, bias=1.0)

# Test-Train Gram matrix for predictions
K_test_train = build_test_train_gram(X_test, Z_test, X_train, Z_train,
                                     degree=2, coef0=1.0, bias=1.0)
```

### Model Training & Prediction
```python
from sklearn.kernel_ridge import KernelRidge

kr = KernelRidge(kernel='precomputed', alpha=0.1)
kr.fit(K_train_train, y_train)
y_pred = kr.predict(K_test_train)
```

---

## 🔍 Validation & Testing

The solution was validated using:
- **Google Colab** validation script (provided by course)
- **Public dataset** for intermediate testing
- **Secret dataset** for final evaluation
- **5-trial averaging** for robust performance estimates
- **Cross-validation** ensures generalization

---

## 💡 Key Insights

1. **Optimal Polynomial Degree**: A quadratic kernel (degree=2) provides the best balance between model complexity and generalization
2. **Feature Interaction**: The composite kernel captures important interactions between video length (x) and content features (z)
3. **Regularization**: Alpha=0.1 provides good regularization without over-smoothing
4. **Scalability**: The solution efficiently handles 4000 training samples and 1000 test samples

---

## 📝 Files Description

| File | Purpose |
|------|---------|
| `solution.py` | Main implementation with all functions |
| `notebooks/analysis.ipynb` | Jupyter notebook with step-by-step walkthrough |
| `data/public/*` | Public training/test data |
| `data/secret/*` | Secret data for evaluation (course provided) |
| `results/` | Output files with performance metrics |
| `docs/approach.md` | Detailed mathematical explanation |

---

## 🤝 Contributing

This is an assignment submission. For improvements or questions, feel free to open an issue or contact the author.

---

## 📄 License

Academic use only. Part of CS771 course at IIT Kanpur.

---

## 👤 Author

**Mainak** (mainak9093)  
IIT Kanpur, CS771 - Introduction to Machine Learning

---

## 📞 Support

For issues or questions:
- Check the `docs/approach.md` for detailed explanations
- Review `notebooks/analysis.ipynb` for step-by-step walkthrough
- Ensure all dependencies are installed: `pip install -r requirements.txt`

