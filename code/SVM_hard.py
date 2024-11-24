import numpy as np


# Dual objective and gradient calculations
def dual_objective(lambda_, y, XX):
    return np.sum(lambda_) - 0.5 * np.sum(
        (lambda_ * y)[:, None] * (lambda_ * y)[None, :] * XX
    )


def dual_gradient(lambda_, y, XX):
    return 1 - (XX @ (lambda_ * y)) * y


# Project onto constraints: non-negativity and lambda^T y = 0
def project_feasible(lambda_, y):
    lambda_ -= np.dot(lambda_, y) / np.dot(y, y) * y  # Equality constraint
    lambda_ = np.maximum(lambda_, 0)  # Non-negativity constraint
    return lambda_


# Solve the dual problem using gradient ascent
def solve_dual(X, y, max_iter=1000, tol=1e-1, step_size=1e-3):
    lambda_ = np.random.uniform(0, 0.5, (len(y)))
    costs = []  # To store cost at each iteration
    cost_changes = []  # To store changes in cost between iterations
    XX = X @ X.T  # Precompute X @ X.T once

    for i in range(max_iter):
        cost = dual_objective(lambda_, y, XX)
        grad = dual_gradient(lambda_, y, XX)
        lambda_ += step_size * grad
        lambda_ = project_feasible(lambda_, y)
        costs.append(np.abs(cost))

        if i > 0:  # Skip the first iteration as there is no previous cost to compare
            cost_change = np.abs(costs[-1] - costs[-2])
            cost_changes.append(cost_change)
            if cost_change < tol:  # Check if the change in cost is less than tolerance
                print(f"Converged in {i + 1} iterations.")
                break
        else:
            cost_changes.append(0)  # First iteration has no previous cost to compare

    else:
        print("Reached maximum iterations without convergence.")

    w_star = (lambda_ * y) @ X
    support_vectors = np.where(lambda_ > 1e-5)[0]
    b_star = np.mean(y[support_vectors] - X[support_vectors] @ w_star)

    return lambda_, costs, cost_changes, w_star, support_vectors, b_star
