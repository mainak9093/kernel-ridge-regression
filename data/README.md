# Data Directory

## Structure

```
data/
├── public/              # Public test data (provided)
│   ├── public_x_trn.txt
│   ├── public_x_tst.txt
│   ├── public_Z_trn.txt
│   ├── public_Z_tst.txt
│   ├── public_y_trn.txt
│   └── public_y_tst.txt
│
└── secret/              # Secret evaluation data
    ├── secret_x_trn.txt
    ├── secret_x_tst.txt
    ├── secret_Z_trn.txt
    ├── secret_Z_tst.txt
    ├── secret_y_trn.txt
    └── secret_y_tst.txt
```

## Data Format

### Input Features

- **x (video length)**: Scalar feature representing video duration
  - Shape: (n_samples,)
  - Type: float
  - Range: typically [0, large_value]
  
- **Z (content features)**: Two-dimensional feature vector
  - Shape: (n_samples, 2)
  - Features: [z₁, z₂]
  - Type: float

### Target

- **y (engagement score)**: Continuous target variable
  - Shape: (n_samples,)
  - Type: float
  - Represents: Video engagement/recommendation score

## Dataset Sizes

- **Training set**: 4000 samples
- **Test set**: 1000 samples
- **Total dimensions**: 3 (1 for x + 2 for Z)

## Loading Data

```python
import numpy as np

# Load training data
x_train = np.loadtxt("data/public/public_x_trn.txt")
Z_train = np.loadtxt("data/public/public_Z_trn.txt")
y_train = np.loadtxt("data/public/public_y_trn.txt")

# Load test data
x_test = np.loadtxt("data/public/public_x_tst.txt")
Z_test = np.loadtxt("data/public/public_Z_tst.txt")
y_test = np.loadtxt("data/public/public_y_tst.txt")
```

## Notes

- Secret data is used for final evaluation by the course
- Public data can be used for intermediate testing and development
- All features should be standardized before model training
