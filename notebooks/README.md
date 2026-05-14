# Notebooks

This directory contains Jupyter notebooks with detailed analysis and step-by-step walkthroughs.

## Files

### analysis.ipynb
Complete analysis notebook with:
- Data loading and exploration
- Feature preprocessing and scaling
- Kernel function implementation
- Gram matrix construction
- Hyperparameter tuning with K-Fold CV
- Polynomial degree evaluation
- Results visualization
- Performance metrics calculation

## How to Use

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook

# Open analysis.ipynb
```

## Contents

- **Section 1**: Data Loading & Exploration
  - Load public/secret data
  - Display shapes and statistics
  
- **Section 2**: Data Preprocessing
  - Standardization with StandardScaler
  - Visualization of scaled features
  
- **Section 3**: Kernel Implementation
  - Define polynomial kernel
  - Test with sample data
  - Verify symmetry and properties
  
- **Section 4**: Hyperparameter Tuning
  - Implement K-Fold CV loop
  - Grid search over parameter space
  - Find best hyperparameters
  
- **Section 5**: Polynomial Degree Evaluation
  - Evaluate degrees 1-6
  - Average over 5 trials
  - Visualize results
  
- **Section 6**: Final Model Training
  - Train on full training set
  - Evaluate on test set
  - Generate performance report

## Tips

- Run cells sequentially (don't skip)
- Data must be in data/ directory
- Installation: `pip install -r requirements.txt`
- Execution time: ~5-10 minutes for full run
