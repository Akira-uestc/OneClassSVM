import numpy as np
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
import numba


class OneClassSVM:
    def __init__(self, nu=0.1, gamma=0.1):
        self.nu = nu
        self.gamma = gamma
        self.alpha = None
        self.support_vectors = None
        self.rho = None
        self.X_train = None
        self.scaler = None

    def _rbf_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        return self._static_rbf_kernel(X, Y, self.gamma)

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def _static_rbf_kernel(X, Y, gamma):
        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]
        n_features = X.shape[1]
        K = np.zeros((n_samples_X, n_samples_Y))
        for i in numba.prange(n_samples_X):
            x_i = X[i]
            for j in range(n_samples_Y):
                y_j = Y[j]
                sum_sq = 0.0
                for k in range(n_features):
                    diff = x_i[k] - y_j[k]
                    sum_sq += diff * diff
                K[i, j] = np.exp(-gamma * sum_sq)
        return K

    def fit(self, X):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.X_train = X_scaled
        n_samples = X_scaled.shape[0]

        K = self._rbf_kernel(X_scaled)
        P = matrix(K, tc="d")
        q = matrix(np.zeros(n_samples), tc="d")

        C = 1.0 / (self.nu * n_samples)
        G_std = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h_std = np.hstack((np.zeros(n_samples), C * np.ones(n_samples)))
        G = matrix(G_std, tc="d")
        h = matrix(h_std, tc="d")

        A = matrix(np.ones((1, n_samples)), tc="d")
        b = matrix(np.array([1.0]), tc="d")

        solvers.options["show_progress"] = False

        solution = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(solution["x"])

        sv_mask = self.alpha > 1e-5
        self.support_vectors = self.X_train[sv_mask]

        boundary_sv = (self.alpha > 1e-5) & (self.alpha < C - 1e-5)
        if np.any(boundary_sv):
            K_sv = self._rbf_kernel(self.X_train[boundary_sv], self.X_train)
            self.rho = np.mean(np.dot(K_sv, self.alpha))
        else:
            K_sv = self._rbf_kernel(self.support_vectors, self.X_train)
            self.rho = np.mean(np.dot(K_sv, self.alpha))

    def decision_function(self, X):
        X_scaled = self.scaler.transform(X)
        K_test = self._rbf_kernel(X_scaled, self.X_train)
        return np.dot(K_test, self.alpha) - self.rho

    def predict(self, X):
        decision = self.decision_function(X)
        return np.where(decision >= 0, 1, -1)
