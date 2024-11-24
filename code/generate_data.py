import numpy as np

def generate_uniform(n_samples,seed = None):
    np.random.seed(seed)
    X_plus = np.random.uniform(0.5, 1.5, (n_samples, 2))  # Class +1
    X_minus = np.random.uniform(-1.5, -0.5, (n_samples, 2))  # Class -1
    X = np.vstack([X_plus, X_minus])
    y = np.hstack([np.ones(n_samples), -np.ones(n_samples)])
    return X,y
