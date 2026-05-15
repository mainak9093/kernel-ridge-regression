"""
CS771 Major Assignment 1: Kernel Ridge Regression for Video Recommendation

This module implements Kernel Ridge Regression (KRR) with a custom polynomial kernel
for predicting video engagement based on video length and content features.

Author: Mainak
Date: 2025-26
Course: CS771 - Introduction to Machine Learning, IIT Kanpur
"""

import argparse
import logging
import os
import time
import itertools
from typing import Tuple, Dict, List, Optional

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import polynomial_kernel

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VideoRecommendationKRR:
    """Kernel Ridge Regression model for video recommendation prediction."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

        self.x_train_raw: Optional[np.ndarray] = None
        self.x_test_raw: Optional[np.ndarray] = None
        self.z_train_raw: Optional[np.ndarray] = None
        self.z_test_raw: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.scaler_x: Optional[StandardScaler] = None
        self.scaler_z: Optional[StandardScaler] = None

        self.X_train: Optional[np.ndarray] = None
        self.Z_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.Z_test: Optional[np.ndarray] = None

        self.best_params: Optional[Dict[str, float]] = None
        self.model: Optional[KernelRidge] = None
        self.y_pred: Optional[np.ndarray] = None

    def load_data(self, x_train_path: str, x_test_path: str,
                  z_train_path: str, z_test_path: str,
                  y_train_path: str, y_test_path: str) -> None:
        n_train = 4000
        n_test = 1000

        self.x_train_raw = np.loadtxt(x_train_path).reshape((n_train, 1)).ravel()
        self.x_test_raw = np.loadtxt(x_test_path).reshape((n_test, 1)).ravel()
        self.z_train_raw = np.loadtxt(z_train_path)
        self.z_test_raw = np.loadtxt(z_test_path)
        self.y_train = np.loadtxt(y_train_path).reshape((n_train, 1)).ravel()
        self.y_test = np.loadtxt(y_test_path).reshape((n_test, 1)).ravel()

        logger.info("Loaded data successfully")
        logger.info(f"  x_train: {self.x_train_raw.shape}, z_train: {self.z_train_raw.shape}")
        logger.info(f"  x_test: {self.x_test_raw.shape}, z_test: {self.z_test_raw.shape}")

    def preprocess_data(self) -> None:
        if self.x_train_raw is None or self.z_train_raw is None:
            raise ValueError("Data must be loaded before preprocessing")

        self.scaler_z = StandardScaler().fit(self.z_train_raw)
        self.Z_train = self.scaler_z.transform(self.z_train_raw)
        self.Z_test = self.scaler_z.transform(self.z_test_raw)

        self.scaler_x = StandardScaler().fit(self.x_train_raw.reshape(-1, 1))
        self.X_train = self.scaler_x.transform(self.x_train_raw.reshape(-1, 1)).ravel()
        self.X_test = self.scaler_x.transform(self.x_test_raw.reshape(-1, 1)).ravel()

        logger.info("Data preprocessing complete")
        logger.info(f"  X_train shape: {self.X_train.shape}, Z_train shape: {self.Z_train.shape}")

    def kernel_function(self, X1: np.ndarray, Z1: np.ndarray,
                        X2: np.ndarray, Z2: np.ndarray,
                        degree: int = 3, coef0: float = 1.0,
                        bias: float = 1.0) -> np.ndarray:
        X1 = np.asarray(X1).ravel()
        X2 = np.asarray(X2).ravel()

        Kz = polynomial_kernel(Z1, Z2, degree=degree, coef0=coef0)
        outer_x = np.outer(X1, X2)
        return outer_x * Kz + bias

    def build_gram_matrices(self, degree: int, coef0: float,
                            bias: float) -> Tuple[np.ndarray, np.ndarray]:
        K_train_train = self.kernel_function(self.X_train, self.Z_train,
                                            self.X_train, self.Z_train,
                                            degree=degree, coef0=coef0, bias=bias)

        K_test_train = self.kernel_function(self.X_test, self.Z_test,
                                           self.X_train, self.Z_train,
                                           degree=degree, coef0=coef0, bias=bias)

        return K_train_train, K_test_train

    @staticmethod
    def concatenate_features(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return np.column_stack((X.reshape(-1, 1), Z))

    def tune_hyperparameters(self, degrees: Optional[List[int]] = None,
                              coef0s: Optional[List[float]] = None,
                              alphas: Optional[List[float]] = None,
                              bias_vals: Optional[List[float]] = None,
                              n_splits: int = 4,
                              verbose: bool = True) -> Tuple[Dict[str, float], float, List]:
        if self.x_train_raw is None or self.z_train_raw is None:
            raise ValueError("Train data must be loaded before tuning")

        if degrees is None:
            degrees = [1, 2, 3, 4]
        if coef0s is None:
            coef0s = [0.0, 1.0, 2.0]
        if alphas is None:
            alphas = list(np.logspace(-3, 1, 7))
        if bias_vals is None:
            bias_vals = [1.0]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        param_grid = list(itertools.product(degrees, coef0s, alphas, bias_vals))
        all_results = []
        best_score = float('inf')
        best_params: Dict[str, float] = {}

        if verbose:
            logger.info(f"Hyperparameter tuning: {len(param_grid)} combinations")

        for degree, coef0, alpha, bias in param_grid:
            fold_scores = []
            for train_idx, val_idx in kf.split(self.x_train_raw):
                X_tr_raw = self.x_train_raw[train_idx]
                X_val_raw = self.x_train_raw[val_idx]
                Z_tr_raw = self.z_train_raw[train_idx]
                Z_val_raw = self.z_train_raw[val_idx]
                y_tr = self.y_train[train_idx]
                y_val = self.y_train[val_idx]

                scaler_z = StandardScaler().fit(Z_tr_raw)
                Z_tr = scaler_z.transform(Z_tr_raw)
                Z_val = scaler_z.transform(Z_val_raw)

                scaler_x = StandardScaler().fit(X_tr_raw.reshape(-1, 1))
                X_tr = scaler_x.transform(X_tr_raw.reshape(-1, 1)).ravel()
                X_val = scaler_x.transform(X_val_raw.reshape(-1, 1)).ravel()

                K_tr_tr = self.kernel_function(X_tr, Z_tr, X_tr, Z_tr,
                                              degree=degree, coef0=coef0, bias=bias)
                K_val_tr = self.kernel_function(X_val, Z_val, X_tr, Z_tr,
                                               degree=degree, coef0=coef0, bias=bias)

                kr = KernelRidge(kernel='precomputed', alpha=alpha)
                kr.fit(K_tr_tr, y_tr)
                y_val_pred = kr.predict(K_val_tr)
                fold_scores.append(mean_squared_error(y_val, y_val_pred))

            mean_val = float(np.mean(fold_scores))
            all_results.append(((degree, coef0, alpha, bias), mean_val))

            if mean_val < best_score:
                best_score = mean_val
                best_params = {'degree': degree, 'coef0': coef0,
                               'alpha': alpha, 'bias': bias}

        self.best_params = best_params
        if verbose:
            logger.info(f"Best params: {best_params}")
            logger.info(f"Best CV MSE: {best_score:.6f}")

        return best_params, best_score, all_results

    def train_final_model(self) -> None:
        if self.best_params is None:
            raise ValueError("Must run tune_hyperparameters first")

        params = self.best_params
        t0 = time.time()
        K_train_train, K_test_train = self.build_gram_matrices(
            degree=params['degree'], coef0=params['coef0'], bias=params['bias'])
        t_kernel = time.time() - t0

        t1 = time.time()
        self.model = KernelRidge(kernel='precomputed', alpha=params['alpha'])
        self.model.fit(K_train_train, self.y_train)
        t_train = time.time() - t1

        t2 = time.time()
        self.y_pred = self.model.predict(K_test_train)
        t_pred = time.time() - t2

        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        logger.info("Final model training complete")
        logger.info(f"Kernel build time: {t_kernel:.4f}s")
        logger.info(f"Training time: {t_train:.4f}s")
        logger.info(f"Prediction time: {t_pred:.4f}s")
        logger.info(f"R² Score: {r2:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}")

    def evaluate_polynomial_degrees(self, degrees: Optional[List[int]] = None,
                                    n_trials: int = 5) -> Tuple[List[int], List[float]]:
        if degrees is None:
            degrees = [1, 2, 3, 4, 5, 6]
        if self.best_params is None:
            raise ValueError("Must run tune_hyperparameters first")

        coef0 = self.best_params['coef0']
        bias = self.best_params['bias']
        alpha = self.best_params['alpha']

        r2_scores = []
        logger.info("Evaluating polynomial degrees")

        for degree in degrees:
            trial_scores = []
            for _ in range(n_trials):
                K_train_train, K_test_train = self.build_gram_matrices(
                    degree=degree, coef0=coef0, bias=bias)
                kr = KernelRidge(kernel='precomputed', alpha=alpha)
                kr.fit(K_train_train, self.y_train)
                trial_scores.append(r2_score(self.y_test, kr.predict(K_test_train)))
            avg_r2 = float(np.mean(trial_scores))
            r2_scores.append(avg_r2)
            logger.info(f"  degree={degree}: R²={avg_r2:.6f}")

        return degrees, r2_scores

    def evaluate_baselines(self) -> Dict[str, Dict[str, float]]:
        if self.X_train is None or self.Z_train is None:
            raise ValueError("Data must be preprocessed before evaluating baselines")

        X_train_features = self.concatenate_features(self.X_train, self.Z_train)
        X_test_features = self.concatenate_features(self.X_test, self.Z_test)

        models = {
            'LinearRegression': LinearRegression(),
            'Ridge(alpha=0.1)': Ridge(alpha=0.1),
            'KRR_poly': KernelRidge(kernel='poly', degree=2, coef0=1.0, alpha=0.1),
            'KRR_rbf': KernelRidge(kernel='rbf', alpha=0.1)
        }

        baseline_results: Dict[str, Dict[str, float]] = {}
        logger.info("Evaluating baseline models")

        for name, baseline_model in models.items():
            baseline_model.fit(X_train_features, self.y_train)
            y_pred = baseline_model.predict(X_test_features)
            baseline_results[name] = {
                'r2': float(r2_score(self.y_test, y_pred)),
                'mse': float(mean_squared_error(self.y_test, y_pred)),
                'mae': float(mean_absolute_error(self.y_test, y_pred))
            }
            logger.info(f"  {name}: R²={baseline_results[name]['r2']:.6f}, "
                        f"MSE={baseline_results[name]['mse']:.6f}")

        return baseline_results

    def save_results(self, output_path: str = 'results', baseline_results: Optional[Dict] = None) -> None:
        os.makedirs(output_path, exist_ok=True)

        if self.y_pred is None:
            raise ValueError("No predictions available to save")

        np.savetxt(os.path.join(output_path, 'predictions.txt'), self.y_pred)

        with open(os.path.join(output_path, 'model_config.txt'), 'w') as f:
            f.write(f"Best Parameters: {self.best_params}\n")
            f.write(f"R² Score: {r2_score(self.y_test, self.y_pred):.6f}\n")
            f.write(f"MSE: {mean_squared_error(self.y_test, self.y_pred):.6f}\n")
            f.write(f"MAE: {mean_absolute_error(self.y_test, self.y_pred):.6f}\n")

        if baseline_results is not None:
            with open(os.path.join(output_path, 'baseline_results.txt'), 'w') as f:
                for name, metrics in baseline_results.items():
                    f.write(f"{name}: {metrics}\n")

        logger.info(f"Saved results to {output_path}")


def parse_comma_list(value: str, dtype):
    return [dtype(item) for item in value.split(',') if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Kernel Ridge Regression for video engagement prediction')

    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--dataset', choices=['public', 'secret'], default='public',
                        help='Choose public or secret data split')
    parser.add_argument('--degrees', type=lambda s: parse_comma_list(s, int),
                        default='2,3,4', help='Comma-separated degrees')
    parser.add_argument('--coef0s', type=lambda s: parse_comma_list(s, float),
                        default='0.0,1.0', help='Comma-separated coef0 values')
    parser.add_argument('--alphas', type=lambda s: parse_comma_list(s, float),
                        default='0.001,0.01,0.1,1.0', help='Comma-separated alpha values')
    parser.add_argument('--biases', type=lambda s: parse_comma_list(s, float),
                        default='1.0', help='Comma-separated bias values')
    parser.add_argument('--n-splits', type=int, default=4, help='CV fold count')
    parser.add_argument('--run-baselines', action='store_true', help='Evaluate baseline models')
    parser.add_argument('--evaluate-degrees', action='store_true',
                        help='Evaluate performance across polynomial degrees')
    parser.add_argument('--save-dir', default='results', help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def build_data_paths(data_dir: str, dataset: str) -> Dict[str, str]:
    base = os.path.join(data_dir, dataset)
    return {
        'x_train': os.path.join(base, f'{dataset}_x_trn.txt'),
        'x_test': os.path.join(base, f'{dataset}_x_tst.txt'),
        'z_train': os.path.join(base, f'{dataset}_Z_trn.txt'),
        'z_test': os.path.join(base, f'{dataset}_Z_tst.txt'),
        'y_train': os.path.join(base, f'{dataset}_y_trn.txt'),
        'y_test': os.path.join(base, f'{dataset}_y_tst.txt')
    }


def main() -> None:
    args = parse_args()
    logger.info('Starting Kernel Ridge Regression pipeline')
    logger.info(f"Using dataset: {args.dataset}")

    paths = build_data_paths(args.data_dir, args.dataset)
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

    model = VideoRecommendationKRR(random_state=args.seed)
    model.load_data(
        x_train_path=paths['x_train'],
        x_test_path=paths['x_test'],
        z_train_path=paths['z_train'],
        z_test_path=paths['z_test'],
        y_train_path=paths['y_train'],
        y_test_path=paths['y_test']
    )
    model.preprocess_data()

    model.tune_hyperparameters(
        degrees=args.degrees,
        coef0s=args.coef0s,
        alphas=args.alphas,
        bias_vals=args.biases,
        n_splits=args.n_splits,
        verbose=True
    )

    model.train_final_model()

    baseline_results = None
    if args.run_baselines:
        baseline_results = model.evaluate_baselines()

    if args.evaluate_degrees:
        degrees, r2_scores = model.evaluate_polynomial_degrees()
        logger.info('Degree evaluation complete')

    model.save_results(output_path=args.save_dir, baseline_results=baseline_results)
    logger.info('Pipeline complete')


if __name__ == '__main__':
    main()