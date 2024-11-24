import numpy as np


def SVM_Train(X, y, lambda_=1.0, step_size=1e-4, max_iter=1000, tol=1e-2):

    def prox_f1(v, lambda_m):
        prox_v = np.zeros_like(v)
        for i in range(len(v)):
            if v[i] > 1:
                prox_v[i] = v[i] + lambda_m
            elif v[i] <= 1:
                prox_v[i] = v[i]
        return prox_v

    # Objective function
    def objective(term, z):
        return (term @ z).T @ (term @ z)

    # Forward-backward algorithm
    def forward_backward(
        z_ini, Dy, X_tilde, Pw, lambda_, m, max_iter=max_iter, tol=tol, step_size=0.01
    ):
        """
        Minimize the objective function using the forward-backward algorithm.
        """
        z = z_ini
        costs = []  # To track the objective function values

        for iteration in range(max_iter):

            # Compute gradient of the smooth term
            term = Pw @ np.linalg.inv(X_tilde @ X_tilde.T) @ X_tilde @ Dy
            grad = term.T @ (term @ z)

            # Forward step
            z_forward = z - step_size * grad

            # Backward step (apply proximal operator)
            z = prox_f1(z_forward, lambda_ / m)

            # Compute objective function value
            cost = objective(term, z)
            costs.append(cost)

            # Check for convergence
            if  np.linalg.norm(grad) < tol:
                print(f"Converged in {iteration + 1} iterations.")
                break
        else:
            print("Reached maximum iterations without convergence.")
            #alternative aproach 
            if len(costs) > 1 and (np.abs(costs[-1]) > np.abs(costs[-2])):
                print(f"cost started to increase after {iteration + 1} iterations.")
                #break


        return z, costs

    # Extend variables for soft-threshold SVM problem
    X_tilde = np.hstack((X, -np.ones((X.shape[0], 1)))).T  # Augment with -1
    Dy = np.diag(y)
    n = X_tilde.shape[0]
    m = X_tilde.shape[1]
    Pw = np.block(
        [
            [np.eye(n - 1), np.zeros((n - 1, 1))],
            [np.zeros((1, n - 1)), 0],
        ]
    )  # Projection matrix

    z_ini = np.ones(m)

    # Solve using forward-backward algorithm
    z_star, costs = forward_backward(
        z_ini, Dy, X_tilde, Pw, lambda_, m, max_iter=1000, step_size=step_size
    )

    w_tild_star = np.linalg.inv(X_tilde @ X_tilde.T) @ X_tilde @ Dy @ z_star
    b_star = w_tild_star[-1]
    w_star = w_tild_star[0:-1]
    return w_star, b_star, costs
