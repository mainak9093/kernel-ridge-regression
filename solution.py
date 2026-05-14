"""
CS771 Major Assignment 1: Kernel Ridge Regression for Video Recommendation

This module implements Kernel Ridge Regression (KRR) with a custom polynomial kernel
for predicting video engagement based on video length and content features.

Author: Mainak
Date: 2025-26
Course: CS771 - Introduction to Machine Learning, IIT Kanpur
"""

import numpy as np
import time
import itertools
from typing import Tuple, Dict, List
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import polynomial_kernel


class VideoRecommendationKRR:
    """
    Kernel Ridge Regression model for video recommendation prediction.

    Uses a composite kernel combining video length (x) and content features (z1, z2).
    """

    def __init__(self):
        """Initialize the model components."""
        self.X_train = None
        self.Z_train = None
        self.X_test = None
        self.Z_test = None
        self.y_train = None
        self.y_test = None

        self.scaler_x = None
        self.scaler_z = None

        self.best_params = None
        self.model = None
        self.y_pred = None

    def load_data(self, x_train_path: str, x_test_path: str,
                  z_train_path: str, z_test_path: str,
                  y_train_path: str, y_test_path: str) -> None:
        """
        Load raw data from text files.

        Args:
            x_train_path: Path to training x (video length) data
            x_test_path: Path to test x data
            z_train_path: Path to training z (features) data
            z_test_path: Path to test z data
            y_train_path: Path to training y (target) data
            y_test_path: Path to test y data
        """
        n_train = 4000
        n_test = 1000

        # Load raw data
        x_train_raw = np.loadtxt(x_train_path).reshape((n_train, 1)).ravel()
        x_test_raw = np.loadtxt(x_test_path).reshape((n_test, 1)).ravel()
        z_train_raw = np.loadtxt(z_train_path)  # shape (n_train, 2)
        z_test_raw = np.loadtxt(z_test_path)    # shape (n_test, 2)
        y_train_raw = np.loadtxt(y_train_path).reshape((n_train, 1)).ravel()
        y_test_raw = np.loadtxt(y_test_path).reshape((n_test, 1)).ravel()

        # Store for preprocessing
        self.x_train_raw = x_train_raw
        self.x_test_raw = x_test_raw
        self.z_train_raw = z_train_raw
        self.z_test_raw = z_test_raw
        self.y_train = y_train_raw
        self.y_test = y_test_raw

    def preprocess_data(self) -> None:
        """
        Standardize features using training statistics.

        Applies StandardScaler to:
        - X (video length): scaled independently
        - Z (content features): scaled as 2D features
        """
        # Scale Z features (z1, z2)
        self.scaler_z = StandardScaler().fit(self.z_train_raw)
        self.Z_train = self.scaler_z.transform(self.z_train_raw)
        self.Z_test = self.scaler_z.transform(self.z_test_raw)

        # Scale X feature (video length)
        self.scaler_x = StandardScaler().fit(self.x_train_raw.reshape(-1, 1))
        self.X_train = self.scaler_x.transform(self.x_train_raw.reshape(-1, 1)).ravel()
        self.X_test = self.scaler_x.transform(self.x_test_raw.reshape(-1, 1)).ravel()

        print("✓ Data preprocessing complete")
        print(f"  X_train shape: {self.X_train.shape}, Z_train shape: {self.Z_train.shape}")

    def kernel_function(self, X1: np.ndarray, Z1: np.ndarray,
                       X2: np.ndarray, Z2: np.ndarray,
                       degree: int = 3, coef0: float = 1.0,
                       bias: float = 1.0) -> np.ndarray:
        """
        Compute composite kernel: K̃((x1,z1),(x2,z2)) = x1*x2*Kz(z1,z2) + bias

        Args:
            X1, X2: Video lengths (1D arrays)
            Z1, Z2: Content features (2D arrays of shape (n, 2))
            degree: Polynomial degree for Kz
            coef0: Coefficient in polynomial kernel
            bias: Bias term for numerical stability

        Returns:
            Kernel matrix of shape (n1, n2)
        """
        X1 = np.asarray(X1).ravel()
        X2 = np.asarray(X2).ravel()

        # Polynomial kernel on z features
        Kz = polynomial_kernel(Z1, Z2, degree=degree, coef0=coef0)

        # Outer product of x values
        outer_x = np.outer(X1, X2)

        return outer_x * Kz + bias

    def build_gram_matrices(self, degree: int, coef0: float,
                           bias: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build Gram matrices for training and test sets.

        Args:
            degree, coef0, bias: Kernel parameters

        Returns:
            Tuple of (K_train_train, K_test_train) Gram matrices
        """
        K_train_train = self.kernel_function(self.X_train, self.Z_train,
                                            self.X_train, self.Z_train,
                                            degree=degree, coef0=coef0, bias=bias)

        K_test_train = self.kernel_function(self.X_test, self.Z_test,
                                           self.X_train, self.Z_train,
                                           degree=degree, coef0=coef0, bias=bias)

        return K_train_train, K_test_train

    def tune_hyperparameters(self, degrees: List[int] = None,
                            coef0s: List[float] = None,
                            alphas: List[float] = None,
                            bias_vals: List[float] = None,
                            n_splits: int = 4,
                            verbose: bool = True) -> Tuple[Dict, float, List]:
        """
        Grid search for optimal hyperparameters using K-Fold CV.

        Args:
            degrees: Polynomial degrees to test
            coef0s: Kernel coefficients to test
            alphas: Ridge regularization parameters to test
            bias_vals: Bias terms to test
            n_splits: Number of CV folds
            verbose: Print progress information

        Returns:
            Tuple of (best_params, best_score, all_results)
        """
        if degrees is None:
            degrees = [2, 3, 4]
        if coef0s is None:
            coef0s = [0.0, 1.0]
        if alphas is None:
            alphas = [1e-2, 1e-1, 1.0]
        if bias_vals is None:
            bias_vals = [1.0]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        param_grid = list(itertools.product(degrees, coef0s, alphas, bias_vals))
        all_results = []
        best_score = float('inf')
        best_params = None

        if verbose:
            print(f"\n🔍 Hyperparameter Tuning ({len(param_grid)} combinations):")
            print("=" * 70)

        for idx, (degree, coef0, alpha, bias) in enumerate(param_grid, 1):
            fold_scores = []

            for train_idx, val_idx in kf.split(self.X_train):
                X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
                Z_tr, Z_val = self.Z_train[train_idx], self.Z_train[val_idx]
                y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]

                # Build Gram matrices for this fold
                K_tr_tr = self.kernel_function(X_tr, Z_tr, X_tr, Z_tr,
                                              degree=degree, coef0=coef0, bias=bias)
                K_val_tr = self.kernel_function(X_val, Z_val, X_tr, Z_tr,
                                               degree=degree, coef0=coef0, bias=bias)

                # Train and evaluate
                kr = KernelRidge(kernel='precomputed', alpha=alpha)
                kr.fit(K_tr_tr, y_tr)
                y_val_pred = kr.predict(K_val_tr)
                fold_scores.append(mean_squared_error(y_val, y_val_pred))

            mean_val = float(np.mean(fold_scores))
            all_results.append(((degree, coef0, alpha, bias), mean_val))

            if verbose and idx % max(1, len(param_grid) // 10) == 0:
                print(f"  [{idx:3d}/{len(param_grid)}] degree={degree}, coef0={coef0}, "
                      f"alpha={alpha:.0e} → MSE={mean_val:.6f}")

            if mean_val < best_score:
                best_score = mean_val
                best_params = {'degree': degree, 'coef0': coef0,
                              'alpha': alpha, 'bias': bias}

        if verbose:
            print("=" * 70)
            print(f"✓ Best Parameters: {best_params}")
            print(f"  Best CV MSE: {best_score:.6f}\n")

        self.best_params = best_params
        return best_params, best_score, all_results

    def train_final_model(self) -> None:
        """
        Train the final model using best hyperparameters on full training set.
        """
        if self.best_params is None:
            raise ValueError("Must run tune_hyperparameters first")

        params = self.best_params

        # Build Gram matrices
        t0 = time.time()
        K_train_train, K_test_train = self.build_gram_matrices(
            degree=params['degree'],
            coef0=params['coef0'],
            bias=params['bias']
        )
        t_kernel = time.time() - t0

        # Train model
        t1 = time.time()
        self.model = KernelRidge(kernel='precomputed', alpha=params['alpha'])
        self.model.fit(K_train_train, self.y_train)
        t_train = time.time() - t1

        # Predict
        t2 = time.time()
        self.y_pred = self.model.predict(K_test_train)
        t_pred = time.time() - t2

        # Evaluate
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        print("\n📊 Final Model Training Complete")
        print("=" * 70)
        print(f"Hyperparameters: {params}")
        print(f"Timing:")
        print(f"  Kernel build: {t_kernel:.4f}s")
        print(f"  Model training: {t_train:.4f}s")
        print(f"  Prediction: {t_pred:.4f}s")
        print(f"\nPerformance Metrics:")
        print(f"  R² Score: {r2:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print("=" * 70)

    def evaluate_polynomial_degrees(self, degrees: List[int] = None,
                                   n_trials: int = 5) -> Tuple[List, List]:
        """
        Evaluate R² scores for different polynomial degrees (averaged over trials).

        Args:
            degrees: Polynomial degrees to test
            n_trials: Number of evaluation trials for averaging

        Returns:
            Tuple of (degree_values, r2_scores)
        """
        if degrees is None:
            degrees = [1, 2, 3, 4, 5, 6]

        if self.best_params is None:
            raise ValueError("Must run tune_hyperparameters first")

        # Use best hyperparameters from tuning
        coef0 = self.best_params['coef0']
        bias = self.best_params['bias']
        alpha = self.best_params['alpha']

        r2_scores = []
        degree_values = []

        print(f"\n📈 Evaluating Polynomial Degrees ({n_trials} trials each)")
        print("=" * 70)

        for degree in degrees:
            trial_scores = []

            for trial in range(n_trials):
                K_train_train, K_test_train = self.build_gram_matrices(
                    degree=degree, coef0=coef0, bias=bias
                )

                kr = KernelRidge(kernel='precomputed', alpha=alpha)
                kr.fit(K_train_train, self.y_train)
                r2 = kr.score(K_test_train, self.y_test)
                trial_scores.append(r2)

            avg_r2 = np.mean(trial_scores)
            r2_scores.append(avg_r2)
            degree_values.append(degree)

            print(f"  Degree {degree}: R² = {avg_r2:.6f}")

        best_degree_idx = np.argmax(r2_scores)
        print("=" * 70)
        print(f"✓ Best Degree: {degree_values[best_degree_idx]} "
              f"(R² = {r2_scores[best_degree_idx]:.6f})\n")

        return degree_values, r2_scores

    def save_results(self, output_path: str = "results/") -> None:
        """Save model predictions and metrics."""
        import os
        os.makedirs(output_path, exist_ok=True)

        if self.y_pred is None:
            raise ValueError("Must train model first")

        # Save predictions
        np.savetxt(f"{output_path}/predictions.txt", self.y_pred)

        # Save model parameters
        with open(f"{output_path}/model_config.txt", 'w') as f:
            f.write(f"Best Parameters: {self.best_params}\n")
            f.write(f"R² Score: {r2_score(self.y_test, self.y_pred):.6f}\n")
            f.write(f"MSE: {mean_squared_error(self.y_test, self.y_pred):.6f}\n")
            f.write(f"MAE: {mean_absolute_error(self.y_test, self.y_pred):.6f}\n")

        print(f"✓ Results saved to {output_path}")


def main():
    """
    Main execution function.

    Example workflow:
    1. Load data
    2. Preprocess
    3. Tune hyperparameters
    4. Train final model
    5. Evaluate polynomial degrees
    """

    # Initialize model
    model = VideoRecommendationKRR()

    # Load data (update paths as needed)
    print("📂 Loading data...")
    model.load_data(
        x_train_path="data/secret/secret_x_trn.txt",
        x_test_path="data/secret/secret_x_tst.txt",
        z_train_path="data/secret/secret_Z_trn.txt",
        z_test_path="data/secret/secret_Z_tst.txt",
        y_train_path="data/secret/secret_y_trn.txt",
        y_test_path="data/secret/secret_y_tst.txt"
    )

    # Preprocess
    model.preprocess_data()

    # Hyperparameter tuning
    model.tune_hyperparameters(
        degrees=[2, 3, 4],
        coef0s=[0.0, 1.0],
        alphas=[1e-2, 1e-1, 1.0],
        n_splits=4,
        verbose=True
    )

    # Train final model
    model.train_final_model()

    # Evaluate polynomial degrees
    degree_values, r2_scores = model.evaluate_polynomial_degrees(
        degrees=[1, 2, 3, 4, 5, 6],
        n_trials=5
    )

    # Save results
    model.save_results()

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
