import math
import numpy as np
from numpy.linalg import eigh
from scipy.linalg import cholesky, qr, solve_triangular
from scipy.signal import savgol_filter

def matmul_diag(A: np.ndarray, d: np.ndarray, inplace: int = 0):
    if d.ndim == 2:
        d = np.sum(d, axis=0)
    
    if not inplace:
        return np.stack([a * d for a in A])

    for x in range(A.shape[0]):
        A[x,:] *= d

def zerodiag_perturbation(A: np.ndarray, perturbation: float = pow(10, -99)):
    for x in range(A.shape[0]):
        if A[x,x] == 0:
            A[x,x] = perturbation

def modcholesky(A: np.ndarray, lower: int = False, perturbation: float = pow(10, -99)):
    """ parameters:
            lower: int [False]
            perturbation: float
                the minimum eigenvalue of the modified matrix A'.
                where perturbation = 0, A' is semidefinite.
    """
    # closest hermitian matrix transformation
    A = 0.5 * (A + np.transpose(A))
    # eigendecomposition of A = UΛU*
    eigenvalues, U = eigh(A)
    # non-negative coercion of eigenvalues such that A' = UΛ'U*
    eigenvalues[eigenvalues <= 0] = perturbation
    # QR decomposition of QR = (U√Λ')*
    Q, R = qr(matmul_diag(U, np.sqrt(eigenvalues)).transpose())
    # cholesky of A' = R*R
    return R.transpose() if lower else R

def mvgaussian_logprobdensity(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray,
    perturbation: float = pow(10, -99)):
    """ N(x, μ, Σ) = √((2π)^p detΣ)^-1 e^[-0.5 (x - μ)*Σ^-1(x - μ)]
        LogN(x, μ, Σ) = -0.5p Log(2π) -0.5 Log(detΣ) -0.5[(x - μ)Σ^-1(x - μ)*]
                      sub Σ = LL*, and M = L^-1
                      = -0.5p Log(2π) -0.5 Log(det(LL*)) -0.5[(x - μ)(LL*)^-1(x - μ)*]
                      = -0.5p Log(2π) -0.5 Log(det(LL*)) -0.5[(x-μ)M* M(x-μ)*]
                      sub det(LL*) = det(L)det(L*) = ∏ diag(L)^2
                      = -0.5p Log(2π) -0.5 Log(∏ diag(L)^2) -0.5[(x-μ)M* M(x-μ)*]
                      = -0.5p Log(2π) -Σ[Log(diag(L))] -0.5[(x-μ)M* M(x-μ)*]
                      sub Ly = (x-μ)* then y = M(x-μ)*
                      = -0.5p Log(2π) -Σ[Log(diag(L))] -0.5(y* y)

        Let Σ^ = kΣ such that min(kΣ) is num.significant, then kΣ = LL* and sub (Σ)^-1 = k(kΣ)^-1
        LogN(x, μ, Σ) = -0.5p Log(2π) -0.5 Log(detΣ) -0.5k[(x - μ)(kΣ)^-1(x - μ)*]
                      sub Log(det(Σ)) = Log(det(kΣ)) - Log(k)
                      = -0.5p Log(2π) -0.5(Log(det(kΣ)) - Log(k)) -0.5k[(x - μ)(kΣ)^-1(x - μ)*]
                      = -0.5p Log(2π) -Σ[Log(diag(L))] + 0.5 Log(k) -0.5k[(x-μ)M* M(x-μ)*]
                      = -0.5p Log(2π) -Σ[Log(diag(L))] + 0.5 Log(k) -0.5k(y* y)
    """
    covariance = np.copy(covariance)
    zerodiag_perturbation(covariance, perturbation)

    min_variance = min(np.diag(covariance))
    reg_exponent = max(min(-round(math.log10(min_variance), 0), 99.0), -99.0)
    reg_constant = pow(10, reg_exponent - reg_exponent % 2)

    covariance *= reg_constant # Σ^ = kΣ 

    try:
        L = cholesky(covariance, lower=True)
    except: # non-positive definite matrix
        L = modcholesky(covariance, True, perturbation)
        zerodiag_perturbation(L, perturbation)
    
    p = x.shape[1]
    y = solve_triangular(L, np.transpose(x - mean), lower=True)
    logdet = np.sum(np.log(np.abs(np.diag(L))))

    return (-0.5 * p * math.log(2 * math.pi) -logdet + 0.5 * math.log(reg_constant)
            -0.5 * reg_constant * np.sum(y.transpose() ** 2, axis=1))

def savgolsmooth_derivatives(x: np.ndarray, orders: int, window_length: int, polyorder: int,
    recursive_smoothing: int = 0):
    """
    """
    if not recursive_smoothing:
        smooth_x = savgol_filter(x, window_length, polyorder, axis=0)
        
        x_derivatives = np.stack([
            np.diff(smooth_x, order, axis=0)[orders - order:]
            for order in range(orders + 1)
        ])

        x_derivatives[0,:] = x[orders:]
        return x_derivatives

    x_derivatives = [x]

    for _ in range(orders):
        x_derivatives.append(np.diff(savgol_filter(x_derivatives[-1], window_length,
                polyorder, axis=0), 1, axis=0))

    return np.stack([
        x_derivative[orders - order:] for order, x_derivative in enumerate(x_derivatives)
    ])
