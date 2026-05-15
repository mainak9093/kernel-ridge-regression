import numpy as np

from solution import VideoRecommendationKRR


def test_kernel_symmetry_and_psd():
    model = VideoRecommendationKRR()
    X = np.array([1.0, 2.0, 3.0])
    Z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    K = model.kernel_function(X, Z, X, Z, degree=2, coef0=1.0, bias=1.0)
    assert K.shape == (3, 3)
    assert np.allclose(K, K.T)

    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues >= -1e-8)


def test_feature_matrix_shape():
    model = VideoRecommendationKRR()
    X = np.array([1.0, 2.0])
    Z = np.array([[0.5, 1.0], [1.5, 2.0]])

    features = model.concatenate_features(X, Z)
    assert features.shape == (2, 3)
    assert np.allclose(features[:, 0], X)
    assert np.allclose(features[:, 1:], Z)
