"""
Validation of analytic steady-state derivatives (Adams & Peters 2026,
J. Phys. Chem. C 2026, 130, 3267-3276) against central finite differences.

beta = 1/kbt is the inverse-temperature variable; lnC is log-concentration,
modeled here via the "ads" reaction's bare rate constant (mass action:
rate = k0 * C => d(rate)/d(lnC) = rate).

All tests use the greek_cross tile (l=5, Topology.square(d=2), 32 states) --
small enough for tight FD tolerances, matching test_diagnostic.py's style.
"""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from me_mkm import (
    MEMKMBuilder,
    TileSettings,
    Reaction,
    BepInteraction,
    InitialStateInteraction,
    coverage_mean,
)
from me_mkm.sparse import (
    build_W,
    build_W_components,
    assemble_W,
    assemble_dW_dbeta,
    assemble_dW_dlnC,
    to_steady_state_form,
    to_steady_state_derivative_form,
    solve_steady_state,
    steady_state_derivative,
    production_rate_dbeta_vector,
    production_rate,
    production_rate_derivative,
)

TILE, L = TileSettings.square(sites=5, d=2), 5


def steady_state(W):
    """Solve W @ Theta = 0, sum(Theta) = 1 by replacing the last row."""
    n = W.shape[0]
    Wb = W.tolil()
    Wb[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    return spsolve(Wb.tocsc(), rhs)


def make_builder(k_ads=1.0, k_des=1.0, eps=0.0, kbt=1.0):
    """Single-adsorbate Langmuir builder, optionally with A-A interaction."""
    interaction = InitialStateInteraction([[0.0, 0.0], [0.0, eps]], kbt=kbt)
    reactions = [
        Reaction([0], [1], rate=k_ads, name="ads"),
        Reaction([1], [0], rate=k_des, name="des"),
    ]
    return MEMKMBuilder(
        tile_settings=TILE,
        reactions=reactions,
        species_names=["*", "A"],
        interaction=interaction,
    )


# ===========================================================================
# 1. assemble_W / to_steady_state_form regression vs. the direct Rust path
# ===========================================================================


class TestAssembleRegression:
    @pytest.mark.parametrize("eps", [0.0, 0.5, -0.7])
    def test_assemble_W_matches_build_W(self, eps):
        builder = make_builder(k_ads=1.3, k_des=0.7, eps=eps)
        W_direct = build_W(builder, steady_state=False)
        W_assembled = assemble_W(builder)
        diff = np.abs((W_direct - W_assembled).toarray())
        assert diff.max() < 1e-12

    @pytest.mark.parametrize("eps", [0.0, 0.5, -0.7])
    def test_steady_state_form_matches_build_W(self, eps):
        builder = make_builder(k_ads=1.3, k_des=0.7, eps=eps)
        Wbar_direct = build_W(builder, steady_state=True)
        Wbar_assembled = to_steady_state_form(assemble_W(builder))
        diff = np.abs((Wbar_direct - Wbar_assembled).toarray())
        assert diff.max() < 1e-12

    def test_rate_override(self):
        builder = make_builder(k_ads=1.0, k_des=1.0)
        W_default = assemble_W(builder)
        W_override = assemble_W(builder, rates=np.array([2.0, 1.0]))  # [ads, des]
        # only the "ads" component should change, scaled by (2.0 - 1.0)
        components = {
            r.name: c
            for r, c in zip(builder.get_reactions(), build_W_components(builder))
        }
        expected = W_default + 1.0 * components["ads"]
        diff = np.abs((W_override - expected).toarray())
        assert diff.max() < 1e-12


# ===========================================================================
# 2. dTheta_ss/dbeta vs. central finite differences
# ===========================================================================


def theta_ss_at(k_ads, k_des, eps, kbt):
    builder = make_builder(k_ads=k_ads, k_des=k_des, eps=eps, kbt=kbt)
    return steady_state(build_W(builder, steady_state=False))


class TestBetaDerivative:
    @pytest.mark.parametrize("eps", [0.8, -0.6])
    def test_convergence(self, eps):
        """
        FD error should shrink ~quadratically as h shrinks, until float noise.
        eps=0.0 is excluded here since dTheta/dbeta is then exactly zero
        (no beta-dependence at all -- see test_noninteracting_baseline), which
        makes a relative convergence-rate check ill-defined.
        """
        k_ads, k_des = 1.2, 0.8
        beta0 = 1.0

        builder = make_builder(k_ads=k_ads, k_des=k_des, eps=eps, kbt=1.0 / beta0)
        Wbar = to_steady_state_form(assemble_W(builder))
        Theta_ss, lu = solve_steady_state(Wbar)

        dWbar_dbeta = to_steady_state_derivative_form(
            assemble_dW_dbeta(builder, dk_dbeta=np.zeros(builder.n_rxns))
        )
        dTheta_analytic = steady_state_derivative(lu, dWbar_dbeta, Theta_ss)

        errors = []
        hs = [1e-3, 1e-4, 1e-5]
        for h in hs:
            theta_p = theta_ss_at(k_ads, k_des, eps, 1.0 / (beta0 + h))
            theta_m = theta_ss_at(k_ads, k_des, eps, 1.0 / (beta0 - h))
            dTheta_fd = (theta_p - theta_m) / (2 * h)
            errors.append(np.linalg.norm(dTheta_fd - dTheta_analytic))

        # error should shrink by ~100x when h shrinks by 10x (quadratic FD error)
        for e_prev, e_next in zip(errors, errors[1:]):
            assert e_next < e_prev / 20, (
                f"errors did not converge quadratically: {errors}"
            )
        assert errors[-1] < 1e-6

    def test_noninteracting_baseline(self):
        """eps=0 case: dTheta/dbeta should be exactly zero (bare rates don't depend on beta)."""
        builder = make_builder(k_ads=1.0, k_des=1.0, eps=0.0)
        Wbar = to_steady_state_form(assemble_W(builder))
        Theta_ss, lu = solve_steady_state(Wbar)
        dWbar_dbeta = to_steady_state_derivative_form(
            assemble_dW_dbeta(builder, dk_dbeta=np.zeros(builder.n_rxns))
        )
        dTheta = steady_state_derivative(lu, dWbar_dbeta, Theta_ss)
        assert np.allclose(dTheta, 0.0, atol=1e-12)


# ===========================================================================
# 3. dTheta_ss/d(lnC) vs. central finite differences
# ===========================================================================


class TestLnCDerivative:
    @pytest.mark.parametrize("eps", [0.0, 0.5])
    def test_convergence(self, eps):
        k_ads0, k_des = 1.1, 0.9

        builder = make_builder(k_ads=k_ads0, k_des=k_des, eps=eps)
        Wbar = to_steady_state_form(assemble_W(builder))
        Theta_ss, lu = solve_steady_state(Wbar)

        dWbar_dlnC = to_steady_state_derivative_form(
            assemble_dW_dlnC(builder, conc_mask=np.array([True, False]))  # [ads, des]
        )
        dTheta_analytic = steady_state_derivative(lu, dWbar_dlnC, Theta_ss)

        errors = []
        hs = [1e-3, 1e-4, 1e-5]
        for h in hs:
            theta_p = theta_ss_at(k_ads0 * np.exp(h), k_des, eps, 1.0)
            theta_m = theta_ss_at(k_ads0 * np.exp(-h), k_des, eps, 1.0)
            dTheta_fd = (theta_p - theta_m) / (2 * h)
            errors.append(np.linalg.norm(dTheta_fd - dTheta_analytic))

        for e_prev, e_next in zip(errors, errors[1:]):
            assert e_next < e_prev / 20, (
                f"errors did not converge quadratically: {errors}"
            )
        assert errors[-1] < 1e-6


# ===========================================================================
# 4. Coverage derivative cross-check (eq. 5) -- coverage_mean() needs no new code
# ===========================================================================


class TestCoverageDerivative:
    def test_coverage_dbeta_matches_fd(self):
        k_ads, k_des, eps, beta0 = 1.2, 0.8, 0.7, 1.0
        builder = make_builder(k_ads=k_ads, k_des=k_des, eps=eps, kbt=1.0 / beta0)
        Wbar = to_steady_state_form(assemble_W(builder))
        Theta_ss, lu = solve_steady_state(Wbar)
        dWbar_dbeta = to_steady_state_derivative_form(
            assemble_dW_dbeta(builder, dk_dbeta=np.zeros(builder.n_rxns))
        )
        dTheta = steady_state_derivative(lu, dWbar_dbeta, Theta_ss)
        dtheta_analytic = coverage_mean(builder, dTheta)[1]

        h = 1e-5
        theta_p = coverage_mean(
            make_builder(k_ads, k_des, eps, kbt=1.0 / (beta0 + h)),
            theta_ss_at(k_ads, k_des, eps, 1.0 / (beta0 + h)),
        )[1]
        theta_m = coverage_mean(
            make_builder(k_ads, k_des, eps, kbt=1.0 / (beta0 - h)),
            theta_ss_at(k_ads, k_des, eps, 1.0 / (beta0 - h)),
        )[1]
        dtheta_fd = (theta_p - theta_m) / (2 * h)

        assert abs(dtheta_analytic - dtheta_fd) < 1e-6


# ===========================================================================
# 5. Production rate (eq. 4/6)
# ===========================================================================


class TestProductionRate:
    def test_langmuir_closed_form(self):
        """Noninteracting Langmuir: r_des = k_des * theta_A (intensive, per site)."""
        k_ads, k_des = 1.3, 0.6
        builder = make_builder(k_ads=k_ads, k_des=k_des, eps=0.0)
        Theta_ss = steady_state(build_W(builder, steady_state=False))
        theta_A = coverage_mean(builder, Theta_ss)[1]

        r = production_rate(builder, Theta_ss, stoich=np.array([0.0, 1.0]))  # des
        expected = k_des * theta_A
        assert abs(r - expected) < 1e-10

    def test_production_rate_dbeta_vs_fd(self):
        """des rate is Arrhenius in beta: k_des(beta) = k_des0 * exp(-beta * Ea)."""
        k_ads, k_des0, Ea, eps, beta0 = 1.0, 1.0, 0.6, 0.5, 1.0

        def k_des_at(beta):
            return k_des0 * np.exp(-beta * Ea)

        builder = make_builder(
            k_ads=k_ads, k_des=k_des_at(beta0), eps=eps, kbt=1.0 / beta0
        )
        Wbar = to_steady_state_form(assemble_W(builder))
        Theta_ss, lu = solve_steady_state(Wbar)

        dk_dbeta = np.array([0.0, -Ea * k_des_at(beta0)])  # [ads, des]
        dWbar_dbeta = to_steady_state_derivative_form(
            assemble_dW_dbeta(builder, dk_dbeta=dk_dbeta)
        )
        dTheta = steady_state_derivative(lu, dWbar_dbeta, Theta_ss)

        stoich = np.array([0.0, 1.0])  # des
        dr_P_dbeta_vec = production_rate_dbeta_vector(builder, stoich, dk_dbeta)
        dr_analytic = production_rate_derivative(
            builder, Theta_ss, dTheta, stoich, dr_P_dbeta_vec
        )

        def r_at(beta):
            b = make_builder(k_ads=k_ads, k_des=k_des_at(beta), eps=eps, kbt=1.0 / beta)
            ts = steady_state(build_W(b, steady_state=False))
            return production_rate(b, ts, stoich)

        h = 1e-5
        dr_fd = (r_at(beta0 + h) - r_at(beta0 - h)) / (2 * h)

        assert abs(dr_analytic - dr_fd) < 1e-6

    def test_ea_eff_saturated_coverage_limit(self):
        """
        Ea_eff = -(dr_P/dbeta)/r_P should approach the bare desorption Ea in
        the desorption-limited / saturated-coverage limit (k_ads >> k_des =>
        theta_A -> 1, so r_P -> k_des(beta) and Ea_eff -> Ea exactly).
        (In the opposite, adsorption-limited limit k_ads << k_des, theta_A is
        small and r_P -> k_ads instead, which is beta-independent here, so
        Ea_eff -> 0 -- not the bare desorption barrier.)
        """
        k_ads, k_des0, Ea, beta0 = 1000.0, 0.1, 0.8, 1.0

        def k_des_at(beta):
            return k_des0 * np.exp(-beta * Ea)

        builder = make_builder(
            k_ads=k_ads, k_des=k_des_at(beta0), eps=0.0, kbt=1.0 / beta0
        )
        Wbar = to_steady_state_form(assemble_W(builder))
        Theta_ss, lu = solve_steady_state(Wbar)

        dk_dbeta = np.array([0.0, -Ea * k_des_at(beta0)])  # [ads, des]
        dWbar_dbeta = to_steady_state_derivative_form(
            assemble_dW_dbeta(builder, dk_dbeta=dk_dbeta)
        )
        dTheta = steady_state_derivative(lu, dWbar_dbeta, Theta_ss)

        stoich = np.array([0.0, 1.0])  # des
        r_P = production_rate(builder, Theta_ss, stoich)
        dr_P_dbeta_vec = production_rate_dbeta_vector(builder, stoich, dk_dbeta)
        dr_P = production_rate_derivative(
            builder, Theta_ss, dTheta, stoich, dr_P_dbeta_vec
        )

        Ea_eff = -dr_P / r_P
        assert abs(Ea_eff - Ea) < 1e-3, f"Ea_eff={Ea_eff}, expected ~{Ea}"


class TestDWdbetaOmegaBEP:
    def test_dW_dbeta_with_omega_matches_fd(self):
        """BEP mode (omega set): delta_e is w*(S_in - S_out), so dW/dbeta must
        track it. With omega on the global model, ads gets a nonzero S_out
        correction too, exercising the final-state branch of the derivative."""
        eps, beta0, h = 0.4, 1.3, 1e-6

        def builder_at(beta):
            im = BepInteraction([[0.0, 0.0], [0.0, eps]], 0.35, kbt=1.0 / beta)
            reactions = [
                Reaction([0], [1], rate=1.2, name="ads"),
                Reaction([1], [0], rate=0.7, name="des"),
            ]
            return MEMKMBuilder(
                tile_settings=TILE,
                reactions=reactions,
                species_names=["*", "A"],
                interaction=im,
            )

        dW = assemble_dW_dbeta(builder_at(beta0), dk_dbeta=np.zeros(2)).toarray()
        # steady_state=False: the FD must see the raw dynamical W, not the
        # version whose last row is the (beta-independent) normalisation row.
        fd = (
            build_W(builder_at(beta0 + h), steady_state=False).toarray()
            - build_W(builder_at(beta0 - h), steady_state=False).toarray()
        ) / (2 * h)
        assert np.allclose(dW, fd, rtol=1e-5, atol=1e-8)
