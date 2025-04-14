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

