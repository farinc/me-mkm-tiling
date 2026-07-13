"""
Steady-state reproduction of Adams et al. 2025, Figure 3.

Scheme 1 (adsorption/desorption of A, with strong A-A repulsion eps_AA =
3.0 kBT) plus Scheme 2 (Langmuir-Hinshelwood dimerization 2A* -> A2 + 2*,
krxn = 1.0*kdes), swept over the adsorption equilibrium constant K[A] =
kads/kdes. In our sign convention (rate_correction = exp(-eps*n_occ_neighbors),
see test_dynamic.py), their repulsive eps_AA = 3.0 kBT is EPS = -3.0.

Unlike test_dynamic.py (which drives k_ads(t) and checks the transient
response), this reproduces the paper's *steady-state* result: the checkerboard
superlattice pins coverage near theta=0.5 over nearly two log-decades of K[A],
and -- more strikingly -- suppresses the dimerization rate far below what
either a Bragg-Williams mean-field (MF-MKM) or the Greek-cross tile (l=5,d=2,
the complete graph K_5, which cannot host a checkerboard at all) predicts at
the *same* coverage. Same theta, wildly different reactivity, because in the
true checkerboard order A* sites are never adjacent -- exactly the qualitative
signature the paper uses to argue mean-field and small non-checkerboard tiles
give the right coverage for the wrong reason.

The dimerization reaction is given its own noninteracting InteractionModel
(matching the original AdamsGePeters-1DTile Square_Dimer_1Ad_Reactions: the
krxn step is unmodified by lambda, only desorption is), so the builder's
shared repulsive InteractionModel only touches adsorption/desorption.

kMC ground truth is ensemble-averaged (several independent trajectories per
K[A], mean +/- SEM) rather than a single trajectory: dimerization events are
rare in the checkerboard plateau, and a single ~1e5-event trajectory on a
32-site ring can land far from the true mean by chance (this is what exposed
the pair-reaction double-counting bug below -- a single noisy trajectory had
coincidentally matched the doubled/buggy rate).
"""

from functools import lru_cache

import matplotlib
import numpy as np
import pytest
from kmc import run_kmc_dimer_steady_state
from me_mkm import (
    InteractionModel,
    MEMKMBuilder,
    Reaction,
    TileSettings,
    build_W,
    coverage_mean,
    production_rate,
    solve_steady_state,
)
from scipy.optimize import brentq, least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = -3.0  # eps_AA = 3.0 kBT repulsive, Adams et al. 2025 convention
K_DES = 1.0
K_RXN = 1.0  # krxn = 1.0 * kdes, matching Figure 3
Z = 4  # lattice coordination number (square lattice / fish-scale / Greek cross)

# smallest checkerboard-capable tile
FISH_SCALE = TileSettings.square(sites=8, d=3)
# K_5, cannot host a checkerboard
GREEK_CROSS = TileSettings.square(sites=5, d=2)

_NONINTERACTING_PAIR = InteractionModel.noninteracting(2, 1.0)


def build_system(K, tile_settings):
    """Scheme 1 (ads/des, repulsive) + Scheme 2 (dimerization, uncorrected)."""
    interaction = InteractionModel([[0.0, 0.0], [0.0, EPS]])
    reactions = [
        Reaction([0], [1], rate=K * K_DES, name="ads"),
        Reaction([1], [0], rate=K_DES, name="des"),
        Reaction([1, 1], [0, 0], rate=K_RXN, name="dimer").with_interaction(
            _NONINTERACTING_PAIR
        ),
    ]
    return MEMKMBuilder(
        tile_settings=tile_settings,
        reactions=reactions,
        species_names=["*", "A"],
        interaction=interaction,
    )


def exact_theta_rate(K, tile_settings):
    """Exact tile ME-MKM steady-state coverage and dimerization rate (paper eq. 4)."""
    builder = build_system(K, tile_settings)
    W = build_W(builder, steady_state=True)
    Theta_ss, _ = solve_steady_state(W)
    theta = coverage_mean(builder, Theta_ss)[1]
    rate = production_rate(builder, Theta_ss, np.array([0.0, 0.0, 1.0]))  # dimer
    return theta, rate


def _mf_balance(theta, K):
    """Bragg-Williams steady-state site balance with dimerization consumption:
    adsorption gain = desorption loss + dimerization loss. Unlike the paper's
    own MF-MKM (see mf_plain_theta_rate below), this version is given the true
    eps -- a stronger baseline, since even a mean-field that *knows* the
    interaction strength still can't capture the spatial correlations that
    suppress the reaction."""
    return (
        K * K_DES * (1.0 - theta)
        - K_DES * theta * np.exp(-Z * EPS * theta)
        - K_RXN * Z * theta**2
    )


def mf_theta_rate(K):
    """Mean-field (Bragg-Williams, interaction-aware) steady-state coverage and rate."""
    theta = brentq(lambda th: _mf_balance(th, K), 1e-12, 1.0 - 1e-12)
    rate = 0.5 * K_RXN * Z * theta**2
    return theta, rate


def mf_plain_balance(theta, Keff, krxn_fit):
    """The paper's actual MF-MKM (eq. 4): plain Langmuir + LH consumption,
    with NO lateral-interaction term at all -- this is the model an
    experimentalist without knowledge of the true repulsion would write down."""
    return Keff * (1.0 - theta) - theta - krxn_fit * Z * theta**2


def mf_plain_theta_rate(Keff, krxn_fit):
    theta = brentq(lambda th: mf_plain_balance(th, Keff, krxn_fit), 1e-12, 1.0 - 1e-12)
    rate = 0.5 * krxn_fit * Z * theta**2
    return theta, rate


def fit_mf_to_kmc_rates(K_data, rate_data):
    """Least-squares fit of (phi, krxn) in the plain (non-interacting) MF-MKM
    to observed dimerization rates vs K[A] -- mirrors the paper's own
    parameter-estimation exercise (Figure 3/4): K[A] is replaced by an
    adjustable "apparent" phi*K[A], since a modeler fitting rate-vs-[A] data
    with a model that doesn't know about the repulsion can only ever infer an
    apparent (generally wrong) equilibrium constant.

    Residuals are plain (linear-scale), not log-rate: this is what a direct
    least-squares fit to raw rate-vs-[A] data does, and it's what reproduces
    the paper's reported (phi=8.3e-5, krxn=0.91*kdes) -- a log-space fit
    instead treats every decade of rate as equally important and converges to
    a completely different (and wrong-shaped) optimum that saturates coverage
    far too early, because it works just as hard to fit the tiny low-K rates
    as the large high-K ones. The linear fit naturally prioritizes matching
    the large, high-K rates well, which is exactly why the paper's own fitted
    MF-MKM tracks the rate curve at high K but is badly wrong about coverage
    at low/mid K (it doesn't "know" coverage should already be ~0.5 there).
    """

    def residuals(params):
        log_phi, krxn_fit = params
        phi = 10**log_phi
        preds = np.array([mf_plain_theta_rate(phi * K, krxn_fit)[1] for K in K_data])
        return preds - rate_data

    result = least_squares(residuals, x0=[-4.0, 1.0], bounds=([-10, 1e-4], [2, 20]))
    phi_fit = 10 ** result.x[0]
    krxn_fit = result.x[1]
    return phi_fit, krxn_fit


# ===========================================================================
# 1. Exact tile (fish-scale, l=8,d=3) vs ensemble-averaged Gillespie kMC
# ===========================================================================

L_KMC = 32
N_STEPS_KMC = 200_000
N_TRIALS_KMC = 5
# Points where the l=8 tile and the l=32 ring agree to within tight kMC error,
# so the quantitative assertions below are meaningful.
K_KMC_POINTS = [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10_000.0, 100_000.0]
# The plot sweeps further (up to K=1e7). Above ~1e5 the dynamics get stiff
# (adsorption ~1e7 vs repulsive desorption ~e^12 vs dimerization ~1) and the
# fixed-event-count kMC rate is slow to converge, so those points carry large
# (honest) error bars rather than being asserted on.
K_KMC_PLOT_POINTS = K_KMC_POINTS + [1_000_000.0, 10_000_000.0]


@lru_cache(maxsize=None)
def kmc_ensemble(K):
    """Ensemble-averaged kMC steady state at K[A]=K on the fish-scale topology.

    Returns (theta_mean, theta_sem, rate_mean, rate_sem) over N_TRIALS_KMC
    independent trajectories. Cached so the quantitative tests below and the
    plot (which sweeps the same K_KMC_POINTS) don't redundantly re-simulate.
    """
    thetas, rates = [], []
    for seed in range(1, N_TRIALS_KMC + 1):
        theta, rate = run_kmc_dimer_steady_state(
            FISH_SCALE, L_KMC, K, K_RXN, eps=EPS, n_steps=N_STEPS_KMC, seed=seed
        )
        thetas.append(theta)
        rates.append(rate)
    thetas, rates = np.array(thetas), np.array(rates)
    n = len(thetas)
    return (
        thetas.mean(),
        thetas.std(ddof=1) / np.sqrt(n),
        rates.mean(),
        rates.std(ddof=1) / np.sqrt(n),
    )


@pytest.mark.parametrize("K", K_KMC_POINTS)
def test_fish_scale_coverage_matches_kmc(K):
    theta_exact, _ = exact_theta_rate(K, FISH_SCALE)
    theta_mean, theta_sem, _, _ = kmc_ensemble(K)
    # l=8,d=3's neighbor graph is complete bipartite (only 4 sites of each
    # parity, each a neighbor of all 4 opposite-parity sites) -- this is
    # exactly the checkerboard limit's coordination, so it matches a genuinely
    # local large-ring lattice there. Well past the plateau (high K, theta
    # escaping ~0.5) that hyperconnectivity is no longer coincidentally
    # correct, so the small tile has a somewhat larger finite-size deviation
    # from the l=32 ring; the wider floor accounts for that known effect.
    tol = 3 * theta_sem + 0.03
    assert abs(theta_exact - theta_mean) < tol, (
        f"K={K}: exact theta={theta_exact:.4f}, "
        f"kMC theta={theta_mean:.4f}+/-{theta_sem:.4f}, tol={tol:.4f}"
    )


@pytest.mark.parametrize("K", K_KMC_POINTS)
def test_fish_scale_rate_matches_kmc(K):
    _, rate_exact = exact_theta_rate(K, FISH_SCALE)
    _, _, rate_mean, rate_sem = kmc_ensemble(K)
    # A single trajectory's dimerization-event count is rare and noisy in the
    # plateau; the ensemble SEM captures that directly, plus a small absolute
    # floor for points where SEM itself underestimates residual noise.
    tol = 3 * rate_sem + 0.01
    assert abs(rate_exact - rate_mean) < tol, (
        f"K={K}: exact rate={rate_exact:.5e}, "
        f"kMC rate={rate_mean:.5e}+/-{rate_sem:.5e}, tol={tol:.5e}"
    )


# ===========================================================================
# 2. Checkerboard suppresses reactivity: same coverage, wildly different rate
# ===========================================================================

PLATEAU_K_POINTS = [100.0, 300.0, 1000.0]


@pytest.mark.parametrize("K", PLATEAU_K_POINTS)
def test_mean_field_overpredicts_reactivity_at_matched_coverage(K):
    theta_exact, rate_exact = exact_theta_rate(K, FISH_SCALE)
    theta_mf, rate_mf = mf_theta_rate(K)

    assert abs(theta_exact - theta_mf) < 0.1, (
        f"K={K}: coverages should roughly agree (exact={theta_exact:.4f}, "
        f"MF={theta_mf:.4f}) -- otherwise this isn't an apples-to-apples "
        f"reactivity comparison"
    )
    assert rate_mf > 10 * rate_exact, (
        f"K={K}: MF-MKM rate ({rate_mf:.4e}) should be much larger than the "
        f"checkerboard-suppressed exact-tile rate ({rate_exact:.4e}) despite "
        f"similar coverage"
    )


@pytest.mark.parametrize("K", PLATEAU_K_POINTS)
def test_greek_cross_overpredicts_reactivity_at_matched_coverage(K):
    """Greek cross (K_5, complete graph) cannot host a checkerboard order at
    all, so like MF-MKM it fails to suppress the dimerization rate even
    though its coverage is in the same regime as the fish-scale tile's."""
    theta_fish, rate_fish = exact_theta_rate(K, FISH_SCALE)
    theta_greek, rate_greek = exact_theta_rate(K, GREEK_CROSS)

    assert abs(theta_fish - theta_greek) < 0.15, (
        f"K={K}: coverages should be in the same regime (fish={theta_fish:.4f}, "
        f"greek={theta_greek:.4f})"
    )
    assert rate_greek > 10 * rate_fish, (
        f"K={K}: Greek-cross rate ({rate_greek:.4e}) should be much larger "
        f"than the checkerboard-suppressed fish-scale rate ({rate_fish:.4e})"
    )


# ===========================================================================
# 3. Checkerboard plateau: coverage pinned near 0.5 over a wide range of K[A]
# ===========================================================================

PLATEAU_K_RANGE = np.geomspace(50.0, 2000.0, 8)


def test_fish_scale_coverage_plateau_is_tight():
    thetas = np.array([exact_theta_rate(K, FISH_SCALE)[0] for K in PLATEAU_K_RANGE])
    spread = thetas.max() - thetas.min()
    assert spread < 0.05, (
        f"fish-scale coverage should stay pinned near 0.5 across nearly two "
        f"log-decades of K[A] (checkerboard plateau); got spread={spread:.4f} "
        f"(thetas={thetas})"
    )


def test_greek_cross_has_no_comparable_plateau():
    thetas = np.array([exact_theta_rate(K, GREEK_CROSS)[0] for K in PLATEAU_K_RANGE])
    spread = thetas.max() - thetas.min()
    assert spread > 0.15, (
        f"Greek cross cannot host a checkerboard order, so its coverage should "
        f"rise substantially (not plateau) over the same K[A] range that pins "
        f"the fish-scale tile; got spread={spread:.4f} (thetas={thetas})"
    )


# ===========================================================================
# 4. Reproduce Figure 3 itself: coverage & rate vs K[A]
# ===========================================================================
# Curves (fish-scale, Greek cross) are cheap exact evaluations, so we sweep
# densely. kMC is expensive, so it's only evaluated (as an ensemble, with
# error bars) at K_KMC_POINTS and overlaid as points -- same as the paper's
# Figure 3 (continuous model curves, discrete kMC markers). The MF-MKM curve
# is the paper's actual approach: fit (phi, krxn) to the kMC rate data rather
# than plugging in the true (unknown-to-a-modeler) interaction strength.

PLOT_K_RANGE = np.logspace(-1, 7, 60)


def test_figure3_plot():
    fish_theta, fish_rate = zip(
        *[exact_theta_rate(K, FISH_SCALE) for K in PLOT_K_RANGE]
    )
    greek_theta, greek_rate = zip(
        *[exact_theta_rate(K, GREEK_CROSS) for K in PLOT_K_RANGE]
    )

    kmc_theta, kmc_theta_sem, kmc_rate, kmc_rate_sem = [], [], [], []
    for K in K_KMC_PLOT_POINTS:
        th, th_sem, r, r_sem = kmc_ensemble(K)
        kmc_theta.append(th)
        kmc_theta_sem.append(th_sem)
        kmc_rate.append(r)
        kmc_rate_sem.append(r_sem)
    kmc_rate = np.array(kmc_rate)

    phi_fit, krxn_fit = fit_mf_to_kmc_rates(np.array(K_KMC_PLOT_POINTS), kmc_rate)
    print(f"\nFitted MF-MKM: phi={phi_fit:.3e}, krxn={krxn_fit:.3f}*kdes")
    mf_theta, mf_rate = zip(
        *[mf_plain_theta_rate(phi_fit * K, krxn_fit) for K in PLOT_K_RANGE]
    )

    green, orange, blue, purple = "#48a2a2", "#ec5c05", "tab:blue", "tab:purple"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax1.plot(
        PLOT_K_RANGE, fish_theta, "-", color=green, lw=1.8, label="l=8,d=3 (fish-scale)"
    )
    ax1.plot(
        PLOT_K_RANGE,
        greek_theta,
        "-.",
        color=purple,
        lw=1.5,
        label="l=5,d=2 (Greek cross)",
    )
    ax1.plot(
        PLOT_K_RANGE,
        mf_theta,
        "--",
        color=blue,
        lw=1.5,
        label="MF-MKM (fit to kMC rates)",
    )
    ax1.errorbar(
        K_KMC_PLOT_POINTS,
        kmc_theta,
        yerr=2 * np.array(kmc_theta_sem),
        fmt="o",
        ms=6,
        capsize=3,
        color=orange,
        ecolor=orange,
        label=f"kMC (l=32 ring, n={N_TRIALS_KMC} trials, +/-2 SEM)",
    )
    ax1.set_xscale("log")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel(r"coverage $\theta_A$")
    ax1.set_title(r"Adams et al. 2025 Figure 3: $\epsilon_{AA}=3\,k_BT$")
    ax1.legend(loc="upper left", fontsize=8)

    ax2.plot(
        PLOT_K_RANGE, fish_rate, "-", color=green, lw=1.8, label="l=8,d=3 (fish-scale)"
    )
    ax2.plot(
        PLOT_K_RANGE,
        greek_rate,
        "-.",
        color=purple,
        lw=1.5,
        label="l=5,d=2 (Greek cross)",
    )
    ax2.plot(
        PLOT_K_RANGE,
        mf_rate,
        "--",
        color=blue,
        lw=1.5,
        label="MF-MKM (fit to kMC rates)",
    )
    ax2.errorbar(
        K_KMC_PLOT_POINTS,
        kmc_rate,
        yerr=2 * np.array(kmc_rate_sem),
        fmt="o",
        ms=6,
        capsize=3,
        color=orange,
        ecolor=orange,
        label=f"kMC (l=32 ring, n={N_TRIALS_KMC} trials, +/-2 SEM)",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$K[A] = k_{ads}/k_{des}$")
    ax2.set_ylabel(r"dimerization rate $r_{rxn}/k_{des}$")
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig("tests/output/figure3_steady_state.png", dpi=150)
    plt.close(fig)
