"""
Quasi-stationary distribution (QSD) of a metastable ME-MKM generator.

For a generator W with an absorbing (or near-absorbing) sink, the ordinary
stationary distribution collapses onto the sink. The QSD instead answers:
conditioned on not yet having been absorbed, what does the system's
distribution over the reactive states look like at long times. This is
the Yaglom limit / Perron root of the sub-generator restricted to the interior.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs


def quasi_stationary_distribution(W, sink):
    """Quasi-stationary distribution and escape rate for a generator with a
    sink.

    W    : dynamical generator, column convention (W[to, from], dTheta/dt =
           W Theta); this codebase's build_W(builder, steady_state=False).
    sink : length-n boolean mask marking the absorbing/sink state(s), e.g. a
           basin mask from me_mkm.microstates.microstate_mask.

    Restricting W to the interior (non-sink) states gives the sub-generator
    W_II. Since sink states don't feed back, interior probability obeys
    dTheta_I/dt = W_II Theta_I exactly -- leakage to the sink is already
    baked into W_II's diagonal (each diagonal entry is -(total outgoing rate),
    sink-bound or not). At long times Theta_I decays as exp(lambda_1 t) times
    its dominant eigenvector, where lambda_1 is the eigenvalue of W_II with
    the largest real part (W_II's spectrum has non-positive real parts, so
    this is the slowest-decaying mode). That eigenvector, normalized to sum
    to 1, is the QSD.

    Returns (nu, lam):
    - nu  : length-n array, zero on sink states, the QSD (sums to 1) on the
      interior states.
    - lam : the escape rate, -Re(lambda_1) >= 0 -- the quasi-stationary decay
      rate of interior mass into the sink.
    """
    sink = np.asarray(sink, dtype=bool)
    interior = ~sink

    W_II = sp.csc_array(W)[interior][:, interior]
    n = W_II.shape[0]
    # ARPACK (eigs) needs k < n-1, which a tiny interior can't satisfy -- dense
    # np.linalg.eig is the only option there, and cheap at that size anyway.
    if n <= 2:
        vals, vecs = np.linalg.eig(W_II.toarray())
    else:
        vals, vecs = eigs(W_II, k=1, which="LR")
    i = int(np.argmax(vals.real))
    lam = -vals[i].real

    v = vecs[:, i].real
    v = v * np.sign(v.sum())
    v = v / v.sum()

    nu = np.zeros(W.shape[0])
    nu[interior] = v
    return nu, lam
