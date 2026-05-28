from datetime import datetime

import numpy as np
import pytest

import astropy.units as u

from stixpy.calibration.energy import (
    ELUTCorrectedCounts,
    _apply_elut_correction,
    _corr_boundary,
    _corr_single_at_peak,
    _corr_single_off_peak,
    apply_elut_correction,
    correct_counts,
    estimate_spectral_index,
    get_elut,
)
from stixpy.product import Product


def test_get_elut():
    with pytest.raises(ValueError, match=r"No ELUT for for date.*"):
        get_elut(datetime(2019, 1, 1))

    elut = get_elut(datetime(2020, 6, 7))
    assert elut.file == "elut_table_20200519.csv"

    elut = get_elut(datetime(2024, 4, 9))
    assert elut.file == "elut_table_20240205.csv"


@pytest.mark.skip(reason="Better test data")
def test_correct_counts():
    uncorrected_prod = Product(
        "/Users/shane/Downloads/solo_L1_stix-sci-xray-cpd-2109270021_20210927T085625_20210927T092350_V01_63392.fits"
    )
    corrected_prod = correct_counts(uncorrected_prod)
    assert corrected_prod.e_cal == "rebin"


# ---------------------------------------------------------------------------
# ELUT correction (stx_elut_correction port)
# ---------------------------------------------------------------------------


def test_estimate_spectral_index_returns_shape_and_peak():
    """Basic shape + peak-location contract."""
    e_lo = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    e_hi = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    # Peak in the middle bin.
    spectrum = np.array([1.0, 5.0, 100.0, 5.0, 1.0])

    sp_index, idx_peak = estimate_spectral_index(e_lo, e_hi, spectrum)

    assert sp_index.shape == (5,)
    assert idx_peak == 2
    # The peak bin's own index is left at 0 by definition.
    assert sp_index[2] == 0.0
    # Below peak the spectrum is rising → negative γ (in E^-γ convention).
    assert sp_index[0] < 0.0
    # Above peak the spectrum is falling → positive γ.
    assert sp_index[3] > 0.0


def test_estimate_spectral_index_pure_power_law():
    """A noise-free E^-γ spectrum recovers γ to numerical precision."""
    e_lo = np.array([10.0, 12.0, 14.0, 16.0])
    e_hi = np.array([12.0, 14.0, 16.0, 18.0])
    e_mean = (e_lo + e_hi) / 2.0
    gamma = 2.5
    spectrum = e_mean ** (-gamma)

    sp_index, idx_peak = estimate_spectral_index(e_lo, e_hi, spectrum)

    # Falling spectrum → peak at the first bin; only the >idx_peak slopes
    # are filled, others stay 0.
    assert idx_peak == 0
    # Indices 1..3 should be very close to γ = 2.5.
    assert np.allclose(sp_index[1:], gamma, atol=0.1)


def test_estimate_spectral_index_clips_extremes():
    """Spectra with huge slopes are clipped to ±8."""
    e_lo = np.array([4.0, 5.0, 6.0])
    e_hi = np.array([5.0, 6.0, 7.0])
    # Spectrum that explodes — slope > 8 in absolute value.
    spectrum = np.array([1.0, 1.0, 1e30])

    sp_index, _ = estimate_spectral_index(e_lo, e_hi, spectrum)
    assert np.all(np.abs(sp_index) <= 8.0)


def test_estimate_spectral_index_handles_non_finite():
    """log(0) etc. shouldn't leak NaN/Inf into the output."""
    e_lo = np.array([4.0, 5.0, 6.0])
    e_hi = np.array([5.0, 6.0, 7.0])
    # A zero in the spectrum gives log(0) = -inf in a slope calculation.
    spectrum = np.array([0.0, 1.0, 0.5])
    sp_index, _ = estimate_spectral_index(e_lo, e_hi, spectrum)
    assert np.all(np.isfinite(sp_index))


def _make_uniform_elut(n_E, n_pix, n_det, e_lo, e_hi, shift=0.0):
    """
    Build a synthetic ELUT where every pixel/detector sees the same edges,
    optionally offset by `shift` keV. Returns (E_low, E_high) with shape
    [n_E, n_pix, n_det] so the case-helper math runs on familiar numbers.
    """
    E_low = np.broadcast_to((e_lo + shift)[:, None, None], (n_E, n_pix, n_det)).copy()
    E_high = np.broadcast_to((e_hi + shift)[:, None, None], (n_E, n_pix, n_det)).copy()
    return E_low, E_high


def test_corr_single_off_peak_matches_closed_form():
    """corr = ratio of power-law integrals over nominal vs ELUT edges."""
    n_E, n_pix, n_det = 5, 12, 32
    e_lo = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    e_hi = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    # ELUT edges shifted down by 0.02 keV (each pixel sees identical edges)
    E_low, E_high = _make_uniform_elut(n_E, n_pix, n_det, e_lo, e_hi, shift=-0.02)

    sp_index = np.full(n_E, 2.0)  # γ = 2 → exponent 1-γ = -1
    k = 2

    corr = _corr_single_off_peak(k, sp_index, e_lo, e_hi, E_low, E_high)

    # Closed form with γ=2: ∫ E^-2 dE = -1/E. Ratio is finite and >1 when ELUT
    # interval is wider than the nominal one (shifted down → wider on the low side).
    expected_num = e_hi[k] ** -1.0 - e_lo[k] ** -1.0
    expected_den = E_high[k, 0, 0] ** -1.0 - E_low[k, 0, 0] ** -1.0
    expected = expected_num / expected_den

    assert corr.shape == (n_pix, n_det)
    assert np.allclose(corr, expected)


def test_corr_single_off_peak_unity_when_edges_match():
    """If ELUT edges == nominal edges, corr = 1 exactly."""
    n_E = 3
    e_lo = np.array([5.0, 10.0, 15.0])
    e_hi = np.array([10.0, 15.0, 20.0])
    E_low, E_high = _make_uniform_elut(n_E, 12, 32, e_lo, e_hi, shift=0.0)
    sp_index = np.full(n_E, 2.0)

    corr = _corr_single_off_peak(1, sp_index, e_lo, e_hi, E_low, E_high)
    assert np.allclose(corr, 1.0)


def test_corr_single_at_peak_matches_closed_form():
    """Split-power-law correction at the spectral peak."""
    n_E = 5
    e_lo = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    e_hi = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    E_low, E_high = _make_uniform_elut(n_E, 12, 32, e_lo, e_hi, shift=-0.03)

    # Bin index 2 is the peak; spectral index below (α) and above (β).
    # γ = 1 is the documented singular case (1−γ = 0 → ÷0), so we avoid 1.
    sp_index = np.array([-1.5, -0.5, 0.0, 2.0, 3.0])
    idx_peak = 2

    corr = _corr_single_at_peak(idx_peak, sp_index, idx_peak, n_E, e_lo, e_hi, E_low, E_high)

    # Hand-compute against the same formulas in _corr_single_at_peak — the
    # test catches accidental sign / index errors.
    alpha = sp_index[idx_peak - 1]  # -0.5
    beta = sp_index[idx_peak + 1]  # 2.0
    E_m = (e_hi[idx_peak] + e_lo[idx_peak]) / 2.0
    ea, eb = 1.0 - alpha, 1.0 - beta
    I_lo_ELUT = (E_m**ea - E_low[idx_peak, 0, 0] ** ea) / ea
    I_hi_ELUT = (E_high[idx_peak, 0, 0] ** eb - E_m**eb) / eb
    I_lo_SCI = (E_m**ea - e_lo[idx_peak] ** ea) / ea
    I_hi_SCI = (e_hi[idx_peak] ** eb - E_m**eb) / eb
    norm = E_m ** (alpha - beta)
    expected = (norm * I_lo_SCI + I_hi_SCI) / (norm * I_lo_ELUT + I_hi_ELUT)

    assert corr.shape == (12, 32)
    assert np.allclose(corr, expected)


def test_corr_single_at_peak_handles_edge_bins():
    """If peak == first / last bin in the file, both α and β reuse the
    only available neighbour."""
    n_E = 3
    e_lo = np.array([5.0, 6.0, 7.0])
    e_hi = np.array([6.0, 7.0, 8.0])
    E_low, E_high = _make_uniform_elut(n_E, 12, 32, e_lo, e_hi, shift=-0.05)
    sp_index = np.array([0.5, 1.5, 2.5])

    # k = 0 (lowest bin) — must use sp_index[1] for both α and β.
    corr0 = _corr_single_at_peak(0, sp_index, 0, n_E, e_lo, e_hi, E_low, E_high)
    # k = n-1 (highest) — must use sp_index[n-2] for both.
    corr_last = _corr_single_at_peak(n_E - 1, sp_index, n_E - 1, n_E, e_lo, e_hi, E_low, E_high)

    assert np.all(np.isfinite(corr0))
    assert np.all(np.isfinite(corr_last))
    assert corr0.shape == (12, 32)


def test_corr_boundary_low_and_high_match_closed_form():
    """Boundary corrections for first / last bin of a multi-bin range."""
    n_E = 4
    e_lo = np.array([5.0, 10.0, 15.0, 20.0])
    e_hi = np.array([10.0, 15.0, 20.0, 25.0])
    E_low, E_high = _make_uniform_elut(n_E, 12, 32, e_lo, e_hi, shift=-0.04)
    sp_index = np.full(n_E, 2.0)

    corr_low = _corr_boundary(0, sp_index[0], "low", e_lo, e_hi, E_low, E_high)
    corr_high = _corr_boundary(-1, sp_index[-1], "high", e_lo, e_hi, E_low, E_high)

    # γ = 2 → exponent = -1.
    exp_low = (E_high[0, 0, 0] ** -1 - e_lo[0] ** -1) / (E_high[0, 0, 0] ** -1 - E_low[0, 0, 0] ** -1)
    exp_high = (e_hi[-1] ** -1 - E_low[-1, 0, 0] ** -1) / (E_high[-1, 0, 0] ** -1 - E_low[-1, 0, 0] ** -1)
    assert np.allclose(corr_low, exp_low)
    assert np.allclose(corr_high, exp_high)


def test_corr_boundary_rejects_bad_edge():
    n_E = 2
    e_lo, e_hi = np.array([5.0, 10.0]), np.array([10.0, 15.0])
    E_low, E_high = _make_uniform_elut(n_E, 12, 32, e_lo, e_hi)
    with pytest.raises(ValueError, match="edge must be"):
        _corr_boundary(0, 2.0, "middle", e_lo, e_hi, E_low, E_high)


# ---------------------------------------------------------------------------
# Raw-array core: full _apply_elut_correction paths
# ---------------------------------------------------------------------------


def _synthetic_counts(n_t=2, n_det=32, n_pix=12, n_E=5):
    """Counts with a recognisable pattern so reshapes can be verified."""
    arr = np.arange(n_t * n_det * n_pix * n_E, dtype=float).reshape(n_t, n_det, n_pix, n_E)
    return arr * u.ct


def test_apply_elut_correction_single_bin_off_peak_shapes():
    """Full raw-array path through Case 1 — verify return shapes + sp_index."""
    n_E, n_pix, n_det, n_t = 5, 12, 32, 3
    e_lo = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    e_hi = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    E_low, E_high = _make_uniform_elut(n_E, n_pix, n_det, e_lo, e_hi, shift=-0.02)
    sp_index = np.full(n_E, 2.0)
    counts = _synthetic_counts(n_t, n_det, n_pix, n_E)
    counts_err = np.sqrt(counts.value) * u.ct

    result = _apply_elut_correction(
        counts=counts,
        counts_error=counts_err,
        energy_bin_idx=np.arange(n_E),
        energy_bin_low=E_low,
        energy_bin_high=E_high,
        energy_low=e_lo,
        energy_high=e_hi,
        energy_ind=2,  # single bin, not the peak (idx_peak = 0 here)
        sp_index=sp_index,
        idx_peak=0,
        pixel_ind=np.arange(3),
        subc_index=np.arange(n_det),
    )

    assert isinstance(result, ELUTCorrectedCounts)
    assert result.counts.shape == (n_pix, n_det, n_t)
    assert result.counts_error.shape == (n_pix, n_det, n_t)
    # Diagnostic shape: [4 quadrants, n_rows, n_subc, n_t]
    assert result.counts_no_elut.shape == (4, 3, n_det, n_t)
    assert result.counts_elut.shape == (4, 3, n_det, n_t)
    assert result.sp_index.shape == (n_E,)


def test_apply_elut_correction_single_bin_unity_when_edges_match():
    """If ELUT == nominal edges, counts pass through unchanged for Case 1."""
    n_E, n_pix, n_det, n_t = 5, 12, 32, 1
    e_lo = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    e_hi = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    E_low, E_high = _make_uniform_elut(n_E, n_pix, n_det, e_lo, e_hi, shift=0.0)
    sp_index = np.full(n_E, 2.0)
    counts = _synthetic_counts(n_t, n_det, n_pix, n_E)
    counts_err = counts.copy()

    result = _apply_elut_correction(
        counts=counts,
        counts_error=counts_err,
        energy_bin_idx=np.arange(n_E),
        energy_bin_low=E_low,
        energy_bin_high=E_high,
        energy_low=e_lo,
        energy_high=e_hi,
        energy_ind=2,
        sp_index=sp_index,
        idx_peak=0,
        pixel_ind=np.arange(3),
        subc_index=np.arange(n_det),
    )

    # No-op correction: the selected bin should equal the original counts.
    original = np.moveaxis(counts.value, [0, 1, 2, 3], [3, 2, 1, 0])[2]
    assert np.allclose(result.counts.value, original)


def test_apply_elut_correction_multi_bin_uncertainty_quadrature():
    """For a 2-bin range with unity correction, σ² = σ_k0² + σ_k1²."""
    n_E, n_pix, n_det, n_t = 5, 12, 32, 1
    e_lo = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    e_hi = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    E_low, E_high = _make_uniform_elut(n_E, n_pix, n_det, e_lo, e_hi, shift=0.0)
    sp_index = np.full(n_E, 2.0)

    # Unit counts so the test isolates the propagation behaviour.
    counts = np.ones((n_t, n_det, n_pix, n_E)) * u.ct
    counts_err = np.ones((n_t, n_det, n_pix, n_E)) * u.ct

    result = _apply_elut_correction(
        counts=counts,
        counts_error=counts_err,
        energy_bin_idx=np.arange(n_E),
        energy_bin_low=E_low,
        energy_bin_high=E_high,
        energy_low=e_lo,
        energy_high=e_hi,
        energy_ind=np.array([1, 2]),
        sp_index=sp_index,
        idx_peak=0,
        pixel_ind=np.arange(3),
        subc_index=np.arange(n_det),
    )

    # With corr = 1 everywhere: counts → 2 (sum of two unity bins),
    # σ_total = √(1² + 1²) = √2 per (pixel, det, time).
    assert np.allclose(result.counts.value, 2.0)
    assert np.allclose(result.counts_error.value, np.sqrt(2.0))


def test_apply_elut_correction_no_time_axis():
    """3-D input (no time axis) is treated as n_t = 1."""
    n_E, n_pix, n_det = 5, 12, 32
    e_lo = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    e_hi = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    E_low, E_high = _make_uniform_elut(n_E, n_pix, n_det, e_lo, e_hi)
    sp_index = np.full(n_E, 2.0)

    counts = np.ones((n_det, n_pix, n_E)) * u.ct
    counts_err = counts.copy()

    result = _apply_elut_correction(
        counts=counts,
        counts_error=counts_err,
        energy_bin_idx=np.arange(n_E),
        energy_bin_low=E_low,
        energy_bin_high=E_high,
        energy_low=e_lo,
        energy_high=e_hi,
        energy_ind=2,
        sp_index=sp_index,
        idx_peak=0,
        pixel_ind=np.arange(3),
        subc_index=np.arange(n_det),
    )
    assert result.counts.shape == (n_pix, n_det, 1)


# ---------------------------------------------------------------------------
# Product wrapper requires sp_index / idx_peak (estimator stubbed)
# ---------------------------------------------------------------------------


def test_apply_elut_correction_smoke_against_real_cpd():
    """End-to-end smoke test on a real `CompressedPixelData` from the test data tarball.

    The physics isn't validated here (we fabricate `sp_index` / `idx_peak`
    since the estimator is stubbed); the test catches column-name /
    shape / ELUT-edge-extraction mismatches against an actual L1 product.
    """
    from stixpy.data.test import STIX_SCI_XRAY_CPD
    from stixpy.product import Product

    p = Product(STIX_SCI_XRAY_CPD)
    # CompressedPixelData layout: counts shape (n_t, n_det, n_pix, n_E) = (5, 32, 12, 32).
    n_t, n_det, n_pix, n_E = p.data["counts"].shape

    result = apply_elut_correction(
        p,
        [4, 16] * u.keV,  # multi-bin range → Case 3
        sp_index=np.full(n_E, 2.0),  # fabricated, uniform γ = 2
        idx_peak=3,  # arbitrary (not used in Case 3 inner bins)
    )

    assert isinstance(result, ELUTCorrectedCounts)
    assert result.counts.shape == (n_pix, n_det, n_t)
    assert result.counts_error.shape == (n_pix, n_det, n_t)
    # Diagnostics: (4 quadrants, n_rows = 1 since default pixel_indices=(0,), n_subc, n_t)
    assert result.counts_no_elut.shape == (4, 1, n_det, n_t)
    assert result.counts_elut.shape == (4, 1, n_det, n_t)
    assert result.sp_index.shape == (n_E,)
    # Result must carry the same unit family as the input counts.
    assert result.counts.unit == p.data["counts"].unit


def test_apply_elut_correction_smoke_against_real_cpd_single_bin():
    """Same as above but Case 1 path (single bin, off the peak)."""
    from stixpy.data.test import STIX_SCI_XRAY_CPD
    from stixpy.product import Product

    p = Product(STIX_SCI_XRAY_CPD)
    n_t, n_det, n_pix, n_E = p.data["counts"].shape

    result = apply_elut_correction(
        p,
        [10, 11] * u.keV,  # exactly one nominal bin → Case 1
        sp_index=np.full(n_E, 2.0),
        idx_peak=3,  # bin 7 (10–11 keV) ≠ idx_peak → off-peak path
    )

    assert isinstance(result, ELUTCorrectedCounts)
    assert result.counts.shape == (n_pix, n_det, n_t)
    assert result.counts_no_elut.shape == (4, 1, n_det, n_t)
