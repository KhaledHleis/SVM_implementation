import numpy as np

def Train_SVM(X, y, eta=1.0, rho=1.0, max_iter=100, tol=1e-5):
    """
    Solve the soft margin SVM using the augmented Lagrangian method.

    Parameters:
    - X: ndarray, shape (n_samples, n_features), input data.
    - y: ndarray, shape (n_samples,), labels (+1 or -1).
    - eta: float, regularization parameter for the margin.
    - rho: float, penalty parameter for augmented Lagrangian.
    - max_iter: int, maximum number of iterations.
    - tol: float, tolerance for convergence.

    Returns:
    - w: ndarray, shape (n_features,), the optimal weight vector.
    - b: float, the optimal bias term.
    - lambda: the support vectors dual variable
    """
    n_samples, n_features = X.shape
    
    # Augment data matrix and initialize variables
    X_aug = np.hstack((X, np.ones((n_samples, 1)))).T  # Augmented X: [x; 1]
    w = np.zeros(n_features + 1)  # Augmented w: [w; b]
    zeta = np.ones(n_samples)# Slack variables
    lambda_dual = np.zeros(n_samples)# Lagrange multipliers
    Dy = np.diag(y)
    # Projection matrix for regularization
    P_w = np.diag([1] * n_features + [0])           # No regularization for the bias term

    for iteration in range(max_iter):
        # Save previous w and zeta for convergence check
        w_prev = w.copy()
        zeta_prev = zeta.copy()

        # Update w (minimize over w) forward steo
        w = np.linalg.inv(P_w + rho * X_aug @ X_aug.T) @ X_aug @ Dy @ (lambda_dual + rho * zeta_prev)

        # Update zeta (minimize over zeta) backward step
        margin = y * (X_aug.T @ w)
        zeta = np.maximum(margin - lambda_dual / rho, 1)

        # Update lambda (maximize over lambda) dual step
        lambda_dual += rho * (zeta - margin)

        # Check convergence
        if np.linalg.norm(w - w_prev) < tol and np.linalg.norm(zeta - zeta_prev) < tol:
            print(f"Converged in {iteration} iterations.")
            break

    # Extract w and b from the augmented vector
    w_opt = w[:-1]  # Weight vector
    b_opt = w[-1]   # Bias term

    return w_opt, b_opt,lambda_dual