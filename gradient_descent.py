import copy
import numpy as np
import math

def cost_j(X, y, w, b):
    """
    Computes the cost function for linear regression.

    Parameters:
    X : numpy array of shape (m, n)  -> Feature matrix
    y : numpy array of shape (m, 1)  -> True target values
    w : numpy array of shape (n, 1)  -> Weight vector
    b : scalar (bias term)

    Returns:
    J : float -> Computed cost
    """
    y_pred = X.dot(w) + b
    res = (y_pred - y)**2
    return res.sum() / (2*y.size)


def derivative(X, y, w, b):
    """
    Computes the derivative of the cost function for polynomial regression.

    Parameters:
    X : numpy array of shape (m, n)  -> Feature matrix
    y : numpy array of shape (m, 1)  -> True target values
    w : numpy array of shape (n, 1)  -> Weight vector
    b : scalar (bias term)

    Returns:
    Returns:
    dj/dw : numpy array of shape (n, 1)  -> gradient vector
    dj/db : float scalar  -> derivative of cost function by b
    """
    m = X.shape[0]
    y_pred = X.dot(w) + b
    dj_dw = X.T.dot(y_pred - y)
    dj_db = (y_pred - y).sum()
    return dj_dw / m, dj_db / m


def gradient_descent(X, y, w, b, alpha, num_iters):
    """
    Computes the cost function for linear regression.

    Parameters:
    X : numpy array of shape (m, n)  -> Feature matrix
    y : numpy array of shape (m, 1)  -> True target values
    w : numpy array of shape (n, 1)  -> Weight vector
    b : scalar (bias term)
    alpha : float -> learning rate
    num_iters : int -> number of iterations

    Returns:
    w_out : numpy array of shape (n, 1) -> Updated vector of weights after running gradient descent
    b_out : scalar float value -> Updated value of parameter b after running gradient descent
    J_history : List of floats >  History of cost values
    p_history : List of [w,b] -> History of parameters [w,b]

    """
    w_out = copy.deepcopy(w)
    b_out = copy.deepcopy(b)
    J_history = []
    p_history = []


    for iteration in range(num_iters):
        dj_dw, dj_db = derivative(X, y, w_out, b_out)


        b_out -= alpha * dj_db
        w_out -= alpha * dj_dw

        if iteration <  num_iters:
            if iteration % 1000 == 0:
                J_history.append(cost_j(X, y, w_out, b_out))
                p_history.append([w_out, b_out])

        # if iteration%math.ceil(num_iters/10) == 0:
        #     print(f"Iteration {iteration:4}: Cost {J_history[-1]:0.2e} ",
        #           f"dj_dw: {dj_dw}, dj_db: {dj_db: 0.3e}  ",
        #           f"w: {w}, b:{b: 0.5e}")



    return w_out, b_out, J_history, p_history






# import numpy as np
# import matplotlib.pyplot as plt
#
# # Generate synthetic data for y = 2x^2 + 3x + 5
# np.random.seed(42)
# X_raw = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 values from -5 to 5
# y_true = 2 * X_raw**2 + 3 * X_raw + 5  # True polynomial function
# y_noise = y_true + np.random.randn(*y_true.shape) * 3  # Adding noise
#
# # Feature Engineering: Create polynomial features (x, x^2)
# X_poly = np.hstack([X_raw, X_raw**2])  # X = [x, x^2]
#
# print(X_poly)
# # Initialize parameters
# w_init = np.zeros((2, 1))  # Two weights (one for x, one for x^2)
# b_init = 0
# alpha = 0.01  # Learning rate
# num_iters = 10000  # Number of iterations
#
# # Shuffle dataset (simulate real-world unordered data)
# shuffle_indices = np.random.permutation(len(X_raw))
# X_shuffled = X_raw[shuffle_indices]
# y_shuffled = y_noise[shuffle_indices]
#
# # Create polynomial features (x, x^2)
# X_poly_shuffled = np.hstack([X_shuffled, X_shuffled**2])
#
# # Train model
# w_opt, b_opt, J_history, p_history = gradient_descent(X_poly_shuffled, y_shuffled, w_init, b_init, alpha, num_iters)
#
# # Compute predictions
# y_pred = X_poly_shuffled.dot(w_opt) + b_opt
#
# # Sort for plotting (ONLY for visualization)
# sort_indices = np.argsort(X_shuffled.flatten())  # Sort by x values
# X_sorted = X_shuffled[sort_indices]
# y_pred_sorted = y_pred[sort_indices]
#
# # Plot sorted predictions
# plt.figure(figsize=(10, 6))
# plt.scatter(X_shuffled, y_shuffled, label="Noisy Data (Unsorted)", color="blue", alpha=0.5)
# plt.plot(X_sorted, y_pred_sorted, label="Model Fit (Sorted for Visualization)", color="red", linewidth=2)
# plt.xlabel("X")
# plt.ylabel("y")
# plt.title("Polynomial Regression Fit with Unsorted Data")
# plt.legend()
# plt.grid(True)
# plt.show()


# # Run gradient descent
# w_opt, b_opt, J_history, p_history = gradient_descent(X_poly, y_noise, w_init, b_init, alpha, num_iters)
#
# # Compute predictions
# y_pred = X_poly.dot(w_opt) + b_opt
#
# # Plot Results
# plt.figure(figsize=(10, 6))
# plt.scatter(X_raw, y_noise, label="Noisy Data", color="blue", alpha=0.5)
# plt.plot(X_raw, y_pred, label="Model Fit", color="red", linewidth=2)
# plt.xlabel("X")
# plt.ylabel("y")
# plt.title("Polynomial Regression Fit using Gradient Descent")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Print optimized parameters
# print(f"Optimized Weights:\n {w_opt}")
# print(f"Optimized Bias: {b_opt}")
