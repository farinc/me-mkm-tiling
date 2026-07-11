"""
Dynamic ME-MKM (W = W(t)) validated against Gillespie kMC.

Two regimes are tested:

1. test_dynamic_w_vs_kmc -- mild attractive interaction, fast sinusoidal
   k_ads(t). General agreement check between the exact tile ME-MKM and a
   Bragg-Williams mean-field control against kMC "experimental" data.

2. test_repulsive_checkerboard_vs_kmc -- strong repulsive interaction
   (eps = -3.0, matching eps_AA = 3.0 kBT "repulsive" in Adams et al. 2025),
   slow log-sinusoidal sweep of k_ads(t) through the checkerboard
   order-disorder transition. Adams et al. showed that this repulsion drives
   adsorbates into a checkerboard superlattice on the (l=8, d=3) brickwork
   tile -- the smallest tile that can represent it -- and that a mean-field
   model cannot capture the resulting coverage plateau at all. This is a
   qualitative, not just quantitative, mean-field failure: the exact tile
   (and kMC on a larger independent lattice) stay pinned near theta=0.5
   while mean-field tracks the drive over a much wider range.
"""

import matplotlib
import numpy as np
from kmc import run_kmc_dynamic_ensemble
from me_mkm import (
    InteractionModel,
    MEMKMBuilder,
    Reaction,
    TileSettings,
    build_graph,
    build_W_components,
    coverage_ic,
    coverages,
    save_html,
)
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 8  # ME-MKM tile: 4-regular, 2**8=256 states, small enough to solve exactly
TOPO = TileSettings.square(sites=L, d=3)  # tile settings (has .deltas, .sites)
Z = 2 * len(TOPO.deltas)  # lattice coordination number (4 for fish-scale)
K_DES = 1.0
rs = 42  # random state
np.random.seed(rs)

# The tile's neighbor structure (TOPO's deltas) describes a physically infinite
# periodic lattice; the small l=8 tile is just the periodic unit ME-MKM solves
# exactly. To get a fair kMC "ground truth", simulate the *same* neighbor
# generators on a much larger periodic ring rather than on the literal l=8
# tile -- comparing against kMC on the tiny tile would only show that the two
# methods agree on one finite system by construction, not that the tile
# result approximates the large/infinite-lattice limit.


def build_system(title, l, eps, k_des=K_DES):
    interaction = InteractionModel([[0.0, 0.0], [0.0, eps]])
    reactions = [
        Reaction(
            [0],
            [1],
            rate=1.0,
            rate_symbol="k_ads",
            rate_symbol_latex="k_{\\text{ads}}",
            name="ads",
        ),
        Reaction(
            [1],
            [0],
            rate=k_des,
            rate_symbol="k_des",
            rate_symbol_latex="k_{\\text{des}}",
            name="des",
        ),
    ]
    builder = MEMKMBuilder(
        tile_settings=TileSettings.square(sites=l, d=3),
        reactions=reactions,
        species_names=["*", "A"],
        interaction=interaction,
    )
    graph_data = build_graph(builder)
    save_html(graph_data, f"tests/output/{title}.html")

    return builder


def solve_memkm_dynamic(
    builder, k_ads_func, k_des, t_eval, theta0_cov=None, species_names=None
):
    """dTheta/dt = W(t) @ Theta, W(t) = k_ads(t)*W_ads + k_des*W_des.

    theta0_cov : dict of species_name -> coverage for the initial condition,
        e.g. {"A": 0.3}. Sites are assumed independent at t=0 (max-entropy
        distribution for the given coverage). Defaults to all-empty surface.
    species_names : passed through to coverage_ic, e.g. ["empty", "A"].
    """
    W_ads, W_des = build_W_components(builder)

    def rhs(t, theta):
        return (k_ads_func(t) * W_ads + k_des * W_des) @ theta

    if theta0_cov is not None:
        theta0 = coverage_ic(builder, theta0_cov, species_names)
    else:
        # all-empty surface: all probability in microstate 0 (every site = 0)
        theta0 = np.zeros(builder.n_states)
        theta0[0] = 1.0

    sol = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        theta0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-9,
        atol=1e-12,
    )
    assert sol.success, sol.message
    return coverages(builder, sol.y)["A"]


def solve_mean_field(eps, k_ads_func, k_des, t_eval, z=Z, theta0=0.0):
    """
    Bragg-Williams (random-mixing) approximation. Replaces the instantaneous
    neighbor occupancy in the desorption correction with the average coverage
    theta, discarding the short-range spatial correlations (clustering or,
    for repulsive eps, checkerboard ordering) that real lateral interactions
    actually build up:

        dtheta/dt = k_ads(t) (1 - theta) - k_des * theta * exp(-z * eps * theta)

    z is the lattice coordination number (degree), here 2*len(deltas) since
    each delta in Topology contributes a forward/backward neighbor pair.
    theta0 : initial coverage (default 0 = clean surface).
    """

    def rhs(t, theta):
        k_a = k_ads_func(t)
        th = theta[0]
        return [k_a * (1.0 - th) - k_des * th * np.exp(-z * eps * th)]

    sol = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        [theta0],
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-9,
        atol=1e-12,
    )
    assert sol.success, sol.message
    return sol.y[0]


def compare_to_kmc(theta_model, kmc_mean, kmc_sem, t_eval, thin_time, start=1):
    """
    Drop points before `start` (e.g. t=0, where every kMC trial deterministically
    starts at theta=0 with SEM=0, or an initial transient period before the
    driven system settles into its periodic cycle), plus any point where
    kmc_sem==0 by chance (possible with a small kMC lattice, since coverage
    is then quantized to multiples of 1/l and a handful of trials can land on
    the same value). Return (chi2_reduced, frac_within_2sigma, nrmse,
    n_thinned). chi2_reduced uses only every `thin`-th point, spaced by
    ~thin_time, so that autocorrelated nearby residuals within a kMC
    trajectory don't understate the effective degrees of freedom.
    """
    nonzero = kmc_sem[start:] > 0
    model_stat = theta_model[start:][nonzero]
    mean_stat = kmc_mean[start:][nonzero]
    sem_stat = kmc_sem[start:][nonzero]
    dt = t_eval[1] - t_eval[0]
    thin = max(1, int(round(thin_time / dt)))

    z_thinned = (model_stat[::thin] - mean_stat[::thin]) / sem_stat[::thin]
    chi2_reduced = np.mean(z_thinned**2)

    z_all = (model_stat - mean_stat) / sem_stat
    frac_within_2sigma = np.mean(np.abs(z_all) <= 2.0)

    nrmse = np.sqrt(np.mean((model_stat - mean_stat) ** 2)) / (
        mean_stat.max() - mean_stat.min()
    )
    return chi2_reduced, frac_within_2sigma, nrmse, len(z_thinned)


def plot_comparison(
    path, title, t_eval, kmc_mean, kmc_sem, n_trials, theta_exact, theta_mf, stats_text
):
    green = "#48a2a2"
    orange = "#ec5c05"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        t_eval,
        kmc_mean,
        yerr=2 * kmc_sem,
        fmt="o",
        ms=3,
        capsize=2,
        color=orange,
        ecolor=orange,
        alpha=0.7,
        label=f"kMC ensemble mean +/- 2 SEM (n={n_trials})",
    )
    ax.plot(
        t_eval,
        theta_exact,
        "-",
        color=green,
        lw=1.8,
        label="$l=8,d=3$",
    )
    ax.plot(t_eval, theta_mf, "--", color="tab:blue", lw=1.5, label="MF-MKM")
    ax.set_xlabel("t")
    ax.set_ylabel(r"coverage $\theta_A$")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    # ax.text(
    #     0.02,
    #     0.03,
    #     stats_text,
    #     transform=ax.transAxes,
    #     fontsize=8,
    #     va="bottom",
    #     ha="left",
    #     bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
    # )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===========================================================================
# 1. Mild attractive interaction: general dynamic agreement check
# ===========================================================================

L_KMC = 256
EPS = 1.0  # mild attractive A-A interaction, in kBT units

AMP = 0.6
PERIOD = 10.0
N_TRIALS = 60
N_STEPS_PER_PERIOD = 20
N_PERIODS = 5
T_EVAL = np.linspace(0.0, N_PERIODS * PERIOD, N_PERIODS * N_STEPS_PER_PERIOD + 1)


def k_ads_func(t):
    return 1.0 * (1.0 + AMP * np.sin(2 * np.pi * t / PERIOD))


def test_dynamic_w_vs_kmc():
    builder = build_system("dynamic_w_vs_kmc", L, EPS)

    # Both the exact solve and the KMC ensemble start from a clean surface.
    theta_exact = solve_memkm_dynamic(builder, k_ads_func, K_DES, T_EVAL)
    theta_mf = solve_mean_field(EPS, k_ads_func, K_DES, T_EVAL)

    trials = run_kmc_dynamic_ensemble(
        TOPO, L_KMC, k_ads_func, K_DES, T_EVAL, N_TRIALS, eps=EPS, seed=0
    )
    kmc_mean = trials.mean(axis=0)
    kmc_sem = trials.std(axis=0, ddof=1) / np.sqrt(N_TRIALS)

    chi2_exact, frac_exact, nrmse_exact, n_thin = compare_to_kmc(
        theta_exact, kmc_mean, kmc_sem, T_EVAL, thin_time=2.0 / K_DES
    )
    chi2_mf, frac_mf, nrmse_mf, _ = compare_to_kmc(
        theta_mf, kmc_mean, kmc_sem, T_EVAL, thin_time=2.0 / K_DES
    )

    print(
        f"\nexact tile ME-MKM:  chi^2_reduced (n={n_thin})={chi2_exact:.3f}  "
        f"frac within 2 SEM={frac_exact:.3f}  NRMSE={nrmse_exact:.4f}"
    )
    print(
        f"mean-field control: chi^2_reduced (n={n_thin})={chi2_mf:.3f}  "
        f"frac within 2 SEM={frac_mf:.3f}  NRMSE={nrmse_mf:.4f}"
    )

    stats_text = (
        rf"exact: $\chi^2_\nu$={chi2_exact:.2f}, frac$_{{2\sigma}}$={frac_exact:.2f}, NRMSE={nrmse_exact:.3f}"
        "\n"
        rf"mean-field: $\chi^2_\nu$={chi2_mf:.2f}, frac$_{{2\sigma}}$={frac_mf:.2f}, NRMSE={nrmse_mf:.3f}"
    )
    plot_comparison(
        "tests/output/dynamic_w_vs_kmc.png",
        "Forced oscillation: ME-MKM vs Gillespie kMC",
        T_EVAL,
        kmc_mean,
        kmc_sem,
        N_TRIALS,
        theta_exact,
        theta_mf,
        stats_text,
    )

    assert chi2_exact < 5.0, f"chi^2_reduced={chi2_exact:.2f} too large"
    assert frac_exact > 0.6, f"only {frac_exact:.2f} of points within 2 SEM"

    # Negative control: a model that throws away spatial correlations should
    # score measurably worse against the same kMC data, proving the metric
    # actually discriminates rather than rubber-stamping anything.
    assert chi2_mf > chi2_exact, (
        f"mean-field control (chi^2={chi2_mf:.2f}) should fit worse than the "
        f"exact tile (chi^2={chi2_exact:.2f}) -- metric isn't discriminating"
    )


# ===========================================================================
# 2. Strong repulsion: checkerboard order defeats mean-field
# ===========================================================================
#
# Adams et al. 2025  found that eps_AA = 3.0 kBT *repulsive* drives kMC into a checkerboard
# superlattice, that the (l=8, d=3) tile is the smallest brickwork tile that
# can represent that pattern, and that a mean-field model cannot capture it.
# In our sign convention (rate_correction = exp(-eps*n_occ_neighbors)),
# their repulsive eps_AA = 3.0 kBT is EPS_REP = -3.0.

L_KMC_REP = 32
EPS_REP = -3.0  # eps_AA = 3.0 kBT repulsive, Adams et al. 2025 convention

K_CENTER = 100.0
LOG_AMP = 1.2  # k_ads(t)
PERIOD_REP = 20.0
N_TRIALS_REP = 30
N_STEPS_PER_PERIOD_REP = 20
N_PERIODS_REP = 3
T_EVAL_REP = np.linspace(
    0.0, N_PERIODS_REP * PERIOD_REP, N_PERIODS_REP * N_STEPS_PER_PERIOD_REP + 1
)
# Skip the first period as a transient: even starting at THETA0_REP=0.5,
# spatial correlations (checkerboard ordering) need time to build up.
STAT_START_REP = N_STEPS_PER_PERIOD_REP


def k_ads_func_rep(t):
    return K_CENTER * np.exp(LOG_AMP * np.sin(2 * np.pi * t / PERIOD_REP))


THETA0_REP = 0.5  # start all models at the checkerboard plateau coverage


def test_repulsive_checkerboard_vs_kmc():
    builder = build_system("checkerboard", L, EPS_REP)

    theta_exact = solve_memkm_dynamic(
        builder,
        k_ads_func_rep,
        K_DES,
        T_EVAL_REP,
        theta0_cov={"A": THETA0_REP},
        species_names=["empty", "A"],
    )
    theta_mf = solve_mean_field(
        EPS_REP, k_ads_func_rep, K_DES, T_EVAL_REP, theta0=THETA0_REP
    )

    trials = run_kmc_dynamic_ensemble(
        TOPO,
        L_KMC_REP,
        k_ads_func_rep,
        K_DES,
        T_EVAL_REP,
        N_TRIALS_REP,
        eps=EPS_REP,
        seed=0,
        theta0=THETA0_REP,
    )
    kmc_mean = trials.mean(axis=0)
    kmc_sem = trials.std(axis=0, ddof=1) / np.sqrt(N_TRIALS_REP)

    chi2_exact, frac_exact, nrmse_exact, n_thin = compare_to_kmc(
        theta_exact,
        kmc_mean,
        kmc_sem,
        T_EVAL_REP,
        thin_time=2.0 / K_DES,
        start=STAT_START_REP,
    )
    chi2_mf, frac_mf, nrmse_mf, _ = compare_to_kmc(
        theta_mf,
        kmc_mean,
        kmc_sem,
        T_EVAL_REP,
        thin_time=2.0 / K_DES,
        start=STAT_START_REP,
    )

    print(
        f"\n[checkerboard] exact tile ME-MKM:  chi^2_reduced (n={n_thin})={chi2_exact:.3f}  "
        f"frac within 2 SEM={frac_exact:.3f}  NRMSE={nrmse_exact:.4f}"
    )
    print(
        f"[checkerboard] mean-field control: chi^2_reduced (n={n_thin})={chi2_mf:.3f}  "
        f"frac within 2 SEM={frac_mf:.3f}  NRMSE={nrmse_mf:.4f}"
    )
    print(
        f"[checkerboard] theta_exact range (post-transient)="
        f"[{theta_exact[STAT_START_REP:].min():.3f}, {theta_exact[STAT_START_REP:].max():.3f}]  "
        f"theta_mf range=[{theta_mf[STAT_START_REP:].min():.3f}, {theta_mf[STAT_START_REP:].max():.3f}]"
    )

    stats_text = (
        rf"exact: $\chi^2_\nu$={chi2_exact:.2f}, frac$_{{2\sigma}}$={frac_exact:.2f}, NRMSE={nrmse_exact:.3f}"
        "\n"
        rf"mean-field: $\chi^2_\nu$={chi2_mf:.2f}, frac$_{{2\sigma}}$={frac_mf:.2f}, NRMSE={nrmse_mf:.3f}"
    )

    print(stats_text)

    plot_comparison(
        "tests/output/dynamic_repulsive_checkerboard.png",
        r"Checkerboard plateau ($\epsilon_{AA}=3\,k_BT$ repulsive): ME-MKM vs kMC",
        T_EVAL_REP,
        kmc_mean,
        kmc_sem,
        N_TRIALS_REP,
        theta_exact,
        theta_mf,
        stats_text,
    )

    # The exact tile should track kMC well despite the strong correlations.
    assert chi2_exact < 5.0, f"chi^2_reduced={chi2_exact:.2f} too large"
    assert frac_exact > 0.6, f"only {frac_exact:.2f} of points within 2 SEM"

    assert chi2_mf > chi2_exact, (
        f"mean-field control (chi^2={chi2_mf:.2f}) should fit worse than the "
        f"exact tile (chi^2={chi2_exact:.2f}) -- metric isn't discriminating"
    )
    mf_amplitude = theta_mf[STAT_START_REP:].max() - theta_mf[STAT_START_REP:].min()
    exact_amplitude = (
        theta_exact[STAT_START_REP:].max() - theta_exact[STAT_START_REP:].min()
    )
    assert mf_amplitude > 3 * exact_amplitude, (
        f"mean-field oscillation amplitude ({mf_amplitude:.3f}) should be much larger "
        f"than the checkerboard-pinned exact tile's ({exact_amplitude:.3f})"
    )
