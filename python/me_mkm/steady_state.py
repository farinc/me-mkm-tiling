"""
Solve the master equation for its stationary distribution.

Given a generator W (from me_mkm.generator), find the steady-state distribution
Theta_ss and its derivatives with respect to a control parameter.

This is a shortcut helper: You can do this using just what is avaiable in generator.py
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def solve_steady_state(Wbar: sp.csc_array):
    """
    Solve Wbar @ Theta_ss = e_n (e_n = [0,...,0,1]) for the steady-state
    distribution using serial SuperLU, returning (Theta_ss, lu)
    so the same LU factorization can be reused by steady_state_derivative
    for any number of derivatives without re-factorizing.
    """
    n = Wbar.shape[0]
    lu = splu(Wbar.tocsc())
    e_n = np.zeros(n)
    e_n[-1] = 1.0
    Theta_ss = lu.solve(e_n)
    return Theta_ss, lu


def steady_state_derivative(lu, dWbar_dx: sp.csc_array, Theta_ss) -> np.ndarray:
    """
    dTheta_ss/dx from the already-factorized Wbar (via solve_steady_state),
    by reusing the same factorization on the right-hand side
    -(dWbar/dx) @ Theta_ss. Derived by differentiating Wbar @ Theta_ss = e_n
    (a constant RHS): Wbar @ (dTheta_ss/dx) = -(dWbar/dx) @ Theta_ss.
    """
    return lu.solve(-(dWbar_dx @ Theta_ss))
