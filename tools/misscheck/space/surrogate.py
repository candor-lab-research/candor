import numpy as np
from profiles import profiles


def design(df, dirs):
    P = profiles(df)
    X = np.stack([P[:, i:i + 3].mean(1) for i in range(0, 28, 3)] + [P[:, 28 + i:28 + i + 3].mean(1) for i in range(0, 28, 3)], 1)
    if dirs:
        dl = df.direction_layer.values.astype(float)
        X = np.concatenate([X, np.stack([np.exp(-0.5 * ((dl - k) / 1.0) ** 2) for k in range(11, 25)], 1)], 1)
    return X


class Ridge:
    def __init__(self, X, y, lam=3.0):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.concatenate([np.ones((len(X), 1)), (X - self.mu) / self.sd], 1)
        I = np.eye(Z.shape[1]); I[0, 0] = 0
        self.w = np.linalg.solve(Z.T @ Z + lam * I, Z.T @ y)

    def __call__(self, X):
        Z = np.concatenate([np.ones((len(X), 1)), (X - self.mu) / self.sd], 1)
        return Z @ self.w
