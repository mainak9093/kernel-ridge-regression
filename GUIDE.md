# Kernel Ridge Regression Project

## Quick Reference

### Installation
```bash
git clone https://github.com/mainak9093/kernel-ridge-regression.git
cd kernel-ridge-regression
pip install -r requirements.txt
```

### Running the Code
```bash
python solution.py
```

### Expected Output
```
✓ Data preprocessing complete
✓ Best Parameters: {'degree': 2, 'coef0': 1.0, 'alpha': 0.1, 'bias': 1.0}
  Best CV MSE: 0.000159

📊 Final Model Training Complete
  R² Score: 0.921075
  MSE: 0.000159
  MAE: 0.009300
```

## Project Structure
- `README.md` - Full project documentation
- `solution.py` - Main implementation
- `requirements.txt` - Dependencies
- `.gitignore` - Git configuration
- `data/` - Training and test data
- `notebooks/` - Jupyter analysis notebook
- `results/` - Output predictions and metrics
- `docs/` - Detailed documentation

## Key Results
- **Best Polynomial Degree**: 2
- **R² Score**: 0.921075
- **Kernel Configuration**: Custom composite polynomial kernel

## For More Information
See `docs/approach.md` for detailed methodology and analysis.
