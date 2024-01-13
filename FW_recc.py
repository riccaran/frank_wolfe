import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

def f_loss(R, P):
    '''
    The "f_loss" function computes the Root Mean Squared Error (RMSE) between the matrices R and P.

    Parameters:
    - R: numpy array
        The first matrix (e.g., the actual ratings).
        
    - P: numpy array
        The second matrix (e.g., the predicted ratings).

    Returns:
    - float
        The Root Mean Squared Error (RMSE) value between the two matrices.
    '''
    R_csr = csr_matrix(R)
    P_csr = csr_matrix(P)
    diff = R_csr - P_csr
    squared_error = np.square(diff.data)
    sum_squared_error = np.sum(squared_error)
    N = R_csr.nnz
    loss = np.sqrt(sum_squared_error / N)

    return loss

def gradient(R, P):
    '''
    The "gradient" computes the gradient of the Root Mean Squared Error (RMSE) 
    with respect to the matrix P for given matrices R and P.

    Parameters:
    - R: numpy array
        The original matrix (e.g., actual ratings or values).
    - P: numpy array
        The predicted matrix.

    Returns:
    - csr_matrix
        The gradient of the RMSE with respect to P in CSR format.
    '''
    R_csr = csr_matrix(R)
    P_csr = csr_matrix(P)
    N = R_csr.nnz
    grad = (P_csr - R_csr) * 2 / N

    return grad

def gamma_line_search(R, P, D, f_loss, gamma_max):
    '''
    The "gamma_line_search" function performs a line search to find the best step size gamma 
    that minimizes the loss function over a given direction D.

    Parameters:
    - R, P: numpy arrays
        The original and predicted matrices, respectively.
    - D: csr_matrix
        The direction in which to search for the optimal step size.
    - f_loss: function
        The loss function to minimize.
    - gamma_max: float
        The maximum permissible value of gamma.

    Returns:
    - float
        The optimal gamma value that minimizes the loss over direction D.
    '''
    losses_t = list()
    gammas = np.linspace(0, gamma_max, 100)
    best_loss = np.inf
    for gamma in gammas:
        gamma_loss = f_loss(R, P + gamma * D)
        losses_t.append(gamma_loss)
        if gamma_loss < best_loss:
            best_loss = gamma_loss
            best_gamma = gamma
    return best_gamma

# Frank-Wolfe implementation with nuclear norm-ball constraint
def Frank_Wolfe(R, epochs, ball_radius, f_loss, gradient, stop = 1e-3, verbose = False):
    '''
    The "Frank_Wolfe" function implements the Frank-Wolfe algorithm with a nuclear norm-ball constraint 
    to minimize the given loss function.

    Parameters:
    - R: numpy array
        The original matrix.
    - epochs: int
        The number of iterations to run the algorithm.
    - ball_radius: float
        The radius of the nuclear norm-ball.
    - f_loss: function
        The loss function to minimize.
    - gradient: function
        The function to compute the gradient of the loss.
    - stop: float, optional (default is 1e-3)
        The threshold for the early stopping criterion based on the Frank-Wolfe gap.
    - verbose: bool, optional (default is False)
        If True, prints the progress of the algorithm.

    Returns:
    - P: numpy array
        The matrix that minimizes the loss function.
    - losses: list
        A list of loss values at each iteration.
    - gaps: list
        A list of Frank-Wolfe gap values at each iteration.
    '''
    # P initialization on the nuclear norm-ball
    P_random = np.random.random(R.shape)
    U, Sigma, Vt = svd(P_random, full_matrices = False)
    Sigma_normalized = ball_radius * Sigma / np.sum(Sigma)
    P = U @ np.diag(Sigma_normalized) @ Vt

    losses = list()
    gaps = list()

    for epoch in range(epochs):
        # Loss calculation
        loss = f_loss(R, P)
        losses.append(loss)

        # Gradient computation
        grad = gradient(R, P)

        # Truncated SVD calculation and top singular vectors extraction
        u, _, vt = svds(-grad, k = 1)  # k = 1 to only compute the top singular vectors and singular value
        S = ball_radius * np.outer(u, vt) # Outer product of the top singular vectors to S

        D = csr_matrix(P - S)
        g = grad.multiply(D).sum()
        gaps.append(g)
        if g < stop:
            print('\nearly_stop')
            return P, losses, gaps

        # P Update step
        gamma = 2 / (epoch + 2)
        P = (1 - gamma) * P + gamma * S

        if verbose:
            print("\rEpoch {}, Loss: {}".format(epoch + 1, loss), end = "")
    
    return P, losses, gaps

# Away-Steps Frank-Wolfe implementation with nuclear norm-ball constraint
def away_steps_FW(R, epochs, ball_radius, f_loss, gradient, stop = 1e-3 ,verbose = False):
    '''
    Implements the Away-Steps Frank-Wolfe algorithm variant with a nuclear norm-ball constraint 
    to minimize the given loss function.

    Parameters:
    - R: numpy array
        The original matrix.
    - epochs: int
        The number of iterations to run the algorithm.
    - ball_radius: float
        The radius of the nuclear norm-ball.
    - f_loss: function
        The loss function to minimize.
    - gradient: function
        The function to compute the gradient of the loss.
    - stop: float, optional (default is 1e-3)
        The threshold for the early stopping criterion based on the Frank-Wolfe gap.
    - verbose: bool, optional (default is False)
        If True, prints the progress of the algorithm.

    Returns:
    - P: numpy array
        The matrix that minimizes the loss function.
    - losses: list
        A list of loss values at each iteration.
    - gaps: list
        A list of Frank-Wolfe gap values at each iteration.
    '''
    # P initialization on the nuclear norm-ball
    P_random = np.random.random(R.shape)
    U, Sigma, Vt = svd(P_random, full_matrices = False)
    Sigma_normalized = ball_radius * Sigma / np.sum(Sigma)
    P = U @ np.diag(Sigma_normalized) @ Vt

    losses = list()
    gaps = list()
    active_set = list()
    alpha = 1 + 1e-10 # alpha initialization

    for epoch in range(epochs):
        # Loss calculation
        loss = f_loss(R, P)
        losses.append(loss)
    
        # Gradient calculation
        grad = gradient(R, P)

        # Frank-Wolfe direction and decrement calculation
        u, _, vt = svds(-grad, k=1)
        S = ball_radius * np.outer(u, vt)
        D_FW = csr_matrix(S - P)
        decrement_FW = (-grad).multiply(D_FW).sum()

        active_set.append(S)

        # Away step direction and decrement calculation
        max_d = -np.inf
        best_v = None
        for v in active_set:
            prod = grad.multiply(csr_matrix(v)).sum()
            if prod > max_d:
                max_d = prod
                best_v = v

        D_A = csr_matrix(P - best_v)
        decrement_A = (-grad).multiply(D_A).sum()

        if decrement_FW >= decrement_A:
            D = D_FW
            gamma_max = 1
            update = "FW"
        else:
            D = D_A
            gamma_max = alpha / (1 - alpha)
            update = 'A'

        gamma = gamma_line_search(R, P, D, f_loss, gamma_max)

        P = P + gamma * D

        g_FW = grad.multiply(-D_FW).sum()
        gaps.append(g_FW)
        if g_FW < stop:
            print('\nearly_stop')
            return P, losses, gaps

        # Active set update step
        if gamma != gamma_max:
            alpha = (1 - gamma) * alpha + gamma
            is_duplicate = any(np.allclose(S, v, atol=1e-8) for v in active_set) 
            if not is_duplicate:
                active_set.append(S)
        else:
            alpha = (1 + gamma) * alpha - gamma
            for idx, v in enumerate(active_set):
                if np.array_equal(v, best_v):
                    del active_set[idx]
                    break

        if verbose:
            print("\rEpoch {}, Loss: {}".format(epoch + 1, loss), end="")
    return P, losses, gaps
