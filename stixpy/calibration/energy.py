from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

import astropy.units as u
from astropy.units import Quantity

from stixpy.calibration.detector import get_sci_channels
from stixpy.io.readers import read_elut, read_elut_index
from stixpy.product import Product
from stixpy.utils.rebining import rebin_proportional

__all__ = [
    "get_elut",
    "correct_counts",
    "estimate_spectral_index",
    "apply_elut_correction",
    "ELUTCorrectedCounts",
]


@dataclass
class ELUTCorrectedCounts:
    """
    Result of :func:`apply_elut_correction`.

    Note: this is currently the only ``@dataclass`` in stixpy. The other
    calibration entry points return plain dicts (e.g.
    :func:`stixpy.calibration.visibility.create_meta_pixels`), tuples
    (:func:`stixpy.calibration.visibility.get_elut_correction`), or mutate
    the product in place (:func:`correct_counts`). Introduced here for
    typed attribute access on the five-field result; if the codebase later
    standardises on a different convention the knowledge is in this one
    class.

    Attributes
    ----------
    counts : `~astropy.units.Quantity`
        ELUT-corrected counts integrated over the selected energy range,
        shape ``(n_pix, n_det, n_t)``.
    counts_error : `~astropy.units.Quantity`
        1σ uncertainty on ``counts``, same shape.
    counts_no_elut : `~astropy.units.Quantity`
        Counts integrated *before* ELUT correction, reshaped to
        ``(4 quadrants, n_rows, n_subc, n_t)`` for diagnostics.
    counts_elut : `~astropy.units.Quantity`
        Counts integrated *after* ELUT correction, same shape as
        ``counts_no_elut``.
    sp_index : `~numpy.ndarray`
        Spectral index used per energy bin, shape ``(n_E,)``.
    """

    counts: Quantity
    counts_error: Quantity
    counts_no_elut: Quantity
    counts_elut: Quantity
    sp_index: np.ndarray


_SPECTRAL_INDEX_CLIP = 8.0


def estimate_spectral_index(energy_low, energy_high, spectrum):
    """
    Estimate a per-energy-bin spectral index for ELUT correction.

    Port of the IDL ``stx_estimate_spectral_index`` routine (S. Krucker /
    P. Massa, April 2025). Assumes a power-law distribution ``E^{-γ}``
    locally in each energy bin and estimates ``γ`` from log–log slopes of
    neighbouring bins, refined by power-law-weighted bin centres.

    Algorithm
    ---------
    1. Compute bin centres ``e_mean = (e_low + e_high) / 2``.
    2. Locate the peak: ``idx_peak = argmax(spectrum)``. The spectral
       index at the peak itself is left at 0 (the routine cannot fit a
       single power law across the peak; see :func:`apply_elut_correction`
       case 2 for how this is handled downstream).
    3. **Below the peak** (forward slopes): for each ``i`` in
       ``[0, idx_peak)``, set
       ``γ_tmp = (log spectrum[i+1] - log spectrum[i])
               / (log e_mean[i+1] - log e_mean[i])``,
       then compute a power-law-weighted bin centre
       ``e_weighted[i] = h0/h1`` with
       ``h0 = (e_high^(γ_tmp+2) - e_low^(γ_tmp+2)) / (γ_tmp + 2)`` and
       ``h1 = (e_high^(γ_tmp+1) - e_low^(γ_tmp+1)) / (γ_tmp + 1)``.
    4. **Above the peak** (backward slopes): same idea using
       ``log spectrum[i] - log spectrum[i-1]`` etc. for ``i`` in
       ``(idx_peak, n_E)``.
    5. ``e_weighted[idx_peak]`` is set to the plain centre.
    6. **Refine**: recompute the slopes using ``e_weighted`` instead of
       ``e_mean``. These refined slopes become ``index_final``.
    7. Convert the sign to the ``E^{-γ}`` convention (negate).
    8. Replace any non-finite entry with 0 and clip to ``±8``.

    Parameters
    ----------
    energy_low, energy_high : `~numpy.ndarray`
        Nominal low / high edges of the science energy bins, shape ``(n_E,)``.
    spectrum : `~numpy.ndarray`
        Background-subtracted spectrum (counts / s / keV), shape ``(n_E,)``.

    Returns
    -------
    sp_index : `~numpy.ndarray`
        Per-bin spectral index, shape ``(n_E,)``.
    idx_peak : int
        Index of the bin containing the spectral peak.
    """
    e_low = np.asarray(energy_low, dtype=float)
    e_high = np.asarray(energy_high, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    n_E = spectrum.size
    e_mean = (e_low + e_high) / 2.0

    idx_peak = int(np.argmax(spectrum))

    index_tmp = np.zeros(n_E)
    e_weighted = np.zeros(n_E)

    def _weighted_centre(i, gamma):
        """Power-law moment ratio h0/h1 for bin i."""
        ga2 = gamma + 2.0
        ga1 = gamma + 1.0
        h0 = (e_high[i] ** ga2 - e_low[i] ** ga2) / ga2
        h1 = (e_high[i] ** ga1 - e_low[i] ** ga1) / ga1
        return h0 / h1

    # log(0), 0/0, and E**γ overflow can all arise on real or pathological
    # spectra; the routine relies on these turning into NaN/inf and being
    # cleaned up at the end (NaN/inf → 0, |γ| ≥ 8 → clip). Suppress the
    # matching RuntimeWarnings so `filterwarnings = error` doesn't promote
    # them to failures.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Below the peak: forward log-log slopes (i, i+1).
        if idx_peak > 0:
            for i in range(idx_peak):
                this_x = np.log(e_mean[i + 1]) - np.log(e_mean[i])
                this_y = np.log(spectrum[i + 1]) - np.log(spectrum[i])
                index_tmp[i] = this_y / this_x
                e_weighted[i] = _weighted_centre(i, index_tmp[i])

        # Above the peak: backward log-log slopes (i-1, i).
        for i in range(idx_peak + 1, n_E):
            this_x = np.log(e_mean[i]) - np.log(e_mean[i - 1])
            this_y = np.log(spectrum[i]) - np.log(spectrum[i - 1])
            index_tmp[i] = this_y / this_x
            e_weighted[i] = _weighted_centre(i, index_tmp[i])

        # No slope at the peak; keep the geometric centre.
        e_weighted[idx_peak] = e_mean[idx_peak]

        index_final = np.zeros(n_E)

        # Refine slopes using weighted centres. Above peak first.
        for i in range(idx_peak + 1, n_E):
            this_x = np.log(e_weighted[i]) - np.log(e_weighted[i - 1])
            this_y = np.log(spectrum[i]) - np.log(spectrum[i - 1])
            index_final[i] = this_y / this_x

        if idx_peak > 0:
            for i in range(idx_peak):
                this_x = np.log(e_weighted[i + 1]) - np.log(e_weighted[i])
                this_y = np.log(spectrum[i + 1]) - np.log(spectrum[i])
                index_final[i] = this_y / this_x

    # Convert to E^-γ convention.
    index_final = -index_final

    # Clean up: NaN / ±inf → 0; clip ±8.
    index_final[~np.isfinite(index_final)] = 0.0
    large = np.abs(index_final) >= _SPECTRAL_INDEX_CLIP
    index_final[large] = np.sign(index_final[large]) * _SPECTRAL_INDEX_CLIP

    return index_final, idx_peak


def get_elut(date):
    r"""
    Get the energy lookup table (ELUT) for the given date

    Combines the ELUT with the science energy channels for the same date.

    Parameters
    ----------
    date `astropy.time.Time`
        Date to look up the ELUT.

    Returns
    -------

    """
    root = Path(__file__).parent.parent
    elut_index_file = Path(root, *["config", "data", "elut", "elut_index.csv"])

    elut_index = read_elut_index(elut_index_file)
    elut_info = elut_index.at(date)
    if len(elut_info) == 0:
        raise ValueError(f"No ELUT for for date {date}")
    elif len(elut_info) > 1:
        raise ValueError(f"Multiple ELUTs for for date {date}")
    start_date, end_date, elut_file = list(elut_info)[0]
    sci_channels = get_sci_channels(date)
    elut_table = read_elut(elut_file, sci_channels)

    return elut_table


def correct_counts(product, method="rebin"):
    """
    Correct count for individual pixel energy calibration


    Parameters
    ----------
    product :
        A stix product
    method : str optional
        The correction method default is to rebin onto science energy channels

    Returns
    -------
    The product with the correct counts
    """
    if method not in ["rebin", "width"]:
        raise ValueError("method not on of the supported methods 'rebin' or 'width'")

    if not isinstance(product, Product) and product.level == "L1":
        raise ValueError("Only supports L1 science products")

    elut = get_elut(product.utc_timerange.center.datetime)

    # rebin back onto science channels
    counts = product.data["counts"][...]
    for did, pid in zip(elut.detector.flat, elut.pixel.flat):
        sum_orig_counts = counts[:, did, pid, 1:-1].sum(axis=-1)
        calibration_edges = elut.e_actual[did, pid, :]

        if method == "rebin":
            # need bin below 4 and above 150 to conserve total counts so [0, 4, ... 150, 160]
            sci_edges = np.hstack([elut.e, 160])
            rebinned = np.apply_along_axis(
                rebin_proportional, -1, counts[:, did, pid, 1:-1], calibration_edges, sci_edges
            )

            sum_rebined_counts = rebinned.sum(axis=-1)
            if not np.allclose(sum_orig_counts.value, sum_rebined_counts):
                raise ValueError("We are losing the precious counts")

            # zero the counts the were rebinned then add all 32 ch
            counts[:, did, pid, 1:-1] = 0
            counts[:, did, pid, :] = counts[:, did, pid, :] + rebinned * counts.unit
        elif method == "width":
            real_width = np.diff(calibration_edges)
            counts[:, pid, did, 1:-1] = counts[:, pid, did, 1:-1] / real_width

    product.data["counts"] = counts
    product.e_cal = method

    return product


# ---------------------------------------------------------------------------
# ELUT correction (spectral-index-weighted) — port of stx_elut_correction.pro
# ---------------------------------------------------------------------------


def _corr_single_off_peak(
    k: int,
    sp_index: np.ndarray,
    e_low: np.ndarray,
    e_high: np.ndarray,
    E_low: np.ndarray,
    E_high: np.ndarray,
) -> np.ndarray:
    """
    Case 1: single bin, not at the spectral peak.

    Ratio of nominal-edges power-law integral to ELUT-edges integral.
    Both numerator and denominator use the same spectral index γ_k, so
    the normalisation constant K of the power law cancels.

    Parameters
    ----------
    k : int
        Index of the energy bin in the science-bin grid.
    sp_index : (n_E,) array
    e_low, e_high : (n_E,) arrays — nominal science edges.
    E_low, E_high : (n_E, n_pix, n_det) arrays — actual ELUT edges.

    Returns
    -------
    corr : (n_pix, n_det) array
    """
    gamma = sp_index[k]
    exponent = 1.0 - gamma
    num = e_high[k] ** exponent - e_low[k] ** exponent
    den = E_high[k] ** exponent - E_low[k] ** exponent
    return num / den


def _corr_single_at_peak(
    k: int,
    sp_index: np.ndarray,
    idx_peak: int,
    n_bins_in_file: int,
    e_low: np.ndarray,
    e_high: np.ndarray,
    E_low: np.ndarray,
    E_high: np.ndarray,
) -> np.ndarray:
    """
    Case 2: single bin AT the spectral peak.

    The bin is split at its mid-point ``E_m = (e_low[k] + e_high[k]) / 2``
    and a different power-law index is used below (``α``) and above (``β``)
    the peak. Continuity at E_m is enforced via the normalisation factor
    ``E_m ** (α − β)`` so the constant cancels.

    Edge convention:
      * peak is the lowest bin in the file → α = β = index of bin above
      * peak is the highest bin in the file → α = β = index of bin below
      * otherwise → α = below-peak bin's index, β = above-peak bin's index
    """
    if k == 0:
        alpha = sp_index[k + 1]
        beta = sp_index[k + 1]
    elif k == n_bins_in_file - 1:
        alpha = sp_index[k - 1]
        beta = sp_index[k - 1]
    else:
        alpha = sp_index[k - 1]
        beta = sp_index[k + 1]

    E_m = (e_high[k] + e_low[k]) / 2.0
    ea = 1.0 - alpha
    eb = 1.0 - beta

    I_lo_ELUT = (E_m**ea - E_low[k] ** ea) / ea
    I_hi_ELUT = (E_high[k] ** eb - E_m**eb) / eb
    I_lo_SCI = (E_m**ea - e_low[k] ** ea) / ea
    I_hi_SCI = (e_high[k] ** eb - E_m**eb) / eb
    norm = E_m ** (alpha - beta)

    return (norm * I_lo_SCI + I_hi_SCI) / (norm * I_lo_ELUT + I_hi_ELUT)


def _corr_boundary(
    k: int,
    gamma: float,
    edge: str,
    e_low: np.ndarray,
    e_high: np.ndarray,
    E_low: np.ndarray,
    E_high: np.ndarray,
) -> np.ndarray:
    """
    Case 3 helper: boundary correction for the first (``edge="low"``) or
    last (``edge="high"``) bin of a multi-bin range.

    For the low edge, the nominal interval is ``[e_low[k], E_high[k]]``
    (the inner edge of the bin is shared with the next, ELUT-defined bin).
    For the high edge, the nominal interval is ``[E_low[k], e_high[k]]``.
    The denominator is always the full ELUT bin width.
    """
    exponent = 1.0 - gamma
    if edge == "low":
        num = E_high[k] ** exponent - e_low[k] ** exponent
    elif edge == "high":
        num = e_high[k] ** exponent - E_low[k] ** exponent
    else:
        raise ValueError(f"edge must be 'low' or 'high', got {edge!r}")
    den = E_high[k] ** exponent - E_low[k] ** exponent
    return num / den


def _apply_elut_correction(
    counts: Quantity,
    counts_error: Quantity,
    energy_bin_idx: Sequence[int],
    energy_bin_low: np.ndarray,
    energy_bin_high: np.ndarray,
    energy_low: np.ndarray,
    energy_high: np.ndarray,
    energy_ind: int | Sequence[int],
    sp_index: np.ndarray,
    idx_peak: int,
    pixel_ind: Sequence[int],
    subc_index: Sequence[int],
) -> ELUTCorrectedCounts:
    """
    Raw-array core for :func:`apply_elut_correction`.

    See the module docstring and ``docs/discussions/`` for the algorithm.
    Internal shapes use stixpy's ``[time, det, pixel, energy]`` convention
    for counts, and ``[n_E, n_pix, n_det]`` for the ELUT edges.

    Parameters
    ----------
    counts, counts_error : Quantity
        Background-subtracted counts and 1σ uncertainty, shape
        ``[n_t, n_det, n_pix, n_E]``.
    energy_bin_idx : array of int
        Indices (into the full science-channel grid) of the energy bins
        present in ``counts``.
    energy_bin_low, energy_bin_high : ndarray
        ELUT edges, shape ``[n_E, n_pix, n_det]``.
    energy_low, energy_high : ndarray
        Nominal science-bin edges, shape ``[n_E,]``.
    energy_ind : int or sequence of int
        Indices (into ``energy_bin_idx``-relative grid) of bins to integrate.
    sp_index : ndarray
        Per-bin spectral index, shape ``[n_E,]``.
    idx_peak : int
        Index of the spectral-peak bin (in the same grid as ``sp_index``).
    pixel_ind, subc_index : array of int
        Selectors for the diagnostic ``counts_no_elut`` / ``counts_elut``
        reshape (row dimension and sub-collimator dimension).

    Returns
    -------
    `ELUTCorrectedCounts`
    """
    energy_ind = np.atleast_1d(energy_ind)
    n_t = counts.shape[0] if counts.ndim == 4 else 1
    n_bins_in_file = len(energy_bin_idx)

    # Reshape counts to [n_E, 4 quadrants, 3 rows, n_subc, n_t] for the
    # diagnostic outputs. stixpy stores counts as [n_t, n_det, n_pix, n_E],
    # IDL stored as [n_E, n_pix, n_det, n_t]; we move axes and reshape.
    # n_pix = 12 → (4 quadrants, 3 rows). n_det = 32 sub-collimators.
    counts_arr = counts if counts.ndim == 4 else counts[np.newaxis]
    counts_err_arr = counts_error if counts_error.ndim == 4 else counts_error[np.newaxis]

    # Move axes: [n_t, n_det, n_pix, n_E] → [n_E, n_pix, n_det, n_t]
    counts_E_first = np.moveaxis(counts_arr, [0, 1, 2, 3], [3, 2, 1, 0])
    counts_err_E_first = np.moveaxis(counts_err_arr, [0, 1, 2, 3], [3, 2, 1, 0])

    def _reshape_diag(arr_2d_or_3d):
        """[n_pix=12, n_det, n_t] → [4, 3, n_det, n_t], then slice."""
        n_pix, n_det, n_t_ = arr_2d_or_3d.shape
        reshaped = arr_2d_or_3d.reshape(4, 3, n_det, n_t_)
        return reshaped[:, pixel_ind][:, :, subc_index]

    if energy_ind.size == 1:
        k = int(energy_ind[0])

        if k == idx_peak:
            corr = _corr_single_at_peak(
                k,
                sp_index,
                idx_peak,
                n_bins_in_file,
                energy_low,
                energy_high,
                energy_bin_low,
                energy_bin_high,
            )
        else:
            corr = _corr_single_off_peak(
                k,
                sp_index,
                energy_low,
                energy_high,
                energy_bin_low,
                energy_bin_high,
            )

        # corr shape: [n_pix, n_det]. Broadcast over time with a new axis.
        corr_t = np.broadcast_to(corr[..., np.newaxis], (*corr.shape, n_t))

        # Counts BEFORE correction (sum over energy_ind = single bin):
        # counts_E_first[k, :, :, :] shape [n_pix, n_det, n_t].
        pre = counts_E_first[k]
        counts_no_elut = _reshape_diag(pre)

        # Apply correction.
        post = pre * corr_t
        post_err = counts_err_E_first[k] * corr_t

        counts_elut = _reshape_diag(post)

        out_counts = post
        out_counts_error = post_err

    else:
        k0 = int(energy_ind[0])
        k1 = int(energy_ind[-1])

        corr_low = _corr_boundary(
            k0,
            sp_index[k0],
            "low",
            energy_low,
            energy_high,
            energy_bin_low,
            energy_bin_high,
        )
        corr_high = _corr_boundary(
            k1,
            sp_index[k1],
            "high",
            energy_low,
            energy_high,
            energy_bin_low,
            energy_bin_high,
        )
        corr_low_t = np.broadcast_to(corr_low[..., np.newaxis], (*corr_low.shape, n_t))
        corr_high_t = np.broadcast_to(corr_high[..., np.newaxis], (*corr_high.shape, n_t))

        # Counts BEFORE correction: sum over energy_ind.
        # counts_E_first[energy_ind] shape [n_sel, n_pix, n_det, n_t].
        pre_stack = counts_E_first[energy_ind]
        pre_sum = pre_stack.sum(axis=0)  # [n_pix, n_det, n_t]
        counts_no_elut = _reshape_diag(pre_sum)

        # Apply boundary corrections to first / last bins only, then sum.
        counts_corr_stack = counts_E_first[energy_ind].copy()
        counts_err_corr_stack = counts_err_E_first[energy_ind].copy()
        counts_corr_stack[0] = counts_corr_stack[0] * corr_low_t
        counts_corr_stack[-1] = counts_corr_stack[-1] * corr_high_t
        counts_err_corr_stack[0] = counts_err_corr_stack[0] * corr_low_t
        counts_err_corr_stack[-1] = counts_err_corr_stack[-1] * corr_high_t

        out_counts = counts_corr_stack.sum(axis=0)
        # Add errors in quadrature (correction factors are deterministic).
        out_counts_error = np.sqrt((counts_err_corr_stack**2).sum(axis=0))

        counts_elut = _reshape_diag(out_counts)

    return ELUTCorrectedCounts(
        counts=out_counts,
        counts_error=out_counts_error,
        counts_no_elut=counts_no_elut,
        counts_elut=counts_elut,
        sp_index=sp_index,
    )


def apply_elut_correction(
    product,
    energy_range,
    pixel_indices: Sequence[int] = (0,),
    subc_indices: Sequence[int] | None = None,
    *,
    spectrum: np.ndarray | None = None,
    sp_index: np.ndarray | None = None,
    idx_peak: int | None = None,
    spectrum_with_bkg: np.ndarray | None = None,
    spectrum_bkg: np.ndarray | None = None,
) -> ELUTCorrectedCounts:
    """
    Apply spectral-index-weighted ELUT correction to background-subtracted counts.

    Thin wrapper that pulls counts, uncertainty, and energy edges out of
    ``product``, fetches the ELUT via :func:`get_elut`, derives
    ``energy_ind`` from ``energy_range``, then delegates to the raw-array
    core (:func:`_apply_elut_correction`).

    Ported from ``stx_elut_correction.pro`` (Paolo Massa, March 2026). See
    ``docs/discussions/`` and the function's source for the algorithm.

    Parameters
    ----------
    product : `~stixpy.product.sources.science.ScienceData`
        STIX science product with ``data["counts"]`` of shape
        ``[n_t, n_det, n_pix, n_E]`` and a matching uncertainty column
        (``counts_comp_err`` or ``counts_comp_comp_err``). Counts are
        assumed background-subtracted.
    energy_range : 2-element sequence
        Low / high edge of the energy range to integrate, in keV (raw
        floats or `~astropy.units.Quantity`).
    pixel_indices : sequence of int
        Row indices (0=TOP, 1=BOT, 2=SMALL) to keep in the diagnostic
        outputs. Default: top row only.
    subc_indices : sequence of int, optional
        Sub-collimator indices to keep in the diagnostic outputs. Default:
        all 32.
    spectrum : ndarray, optional
        Background-subtracted spectrum (n_E,). Currently unused (the
        spectral-index estimator is stubbed); kept for API parity.
    sp_index, idx_peak
        Per-bin spectral index and peak-bin index. **Required for now**,
        since :func:`estimate_spectral_index` is not yet ported.
    spectrum_with_bkg, spectrum_bkg : ndarray, optional
        Kept for API parity with the IDL routine; not used here.

    Returns
    -------
    `ELUTCorrectedCounts`
    """
    # Extract counts + uncertainty. The column name for the uncertainty
    # varies across science products — try the most common variants.
    counts = product.data["counts"]
    counts_error = None
    for name in ("counts_err", "counts_comp_err", "counts_comp_comp_err"):
        if name in product.data.colnames:
            counts_error = product.data[name]
            break
    if counts_error is None:
        raise ValueError(
            "Could not find a counts-error column on the product (looked for "
            "counts_err / counts_comp_err / counts_comp_comp_err)."
        )

    # Energy bins: nominal edges from the product's `energies` table, ELUT
    # actual edges from the daily ELUT. Some products do not carry an
    # `energy_bin_edge_mask` in their control table (and therefore no
    # `energy_masks` attribute); in that case every energy bin in `energies`
    # is treated as present in the file.
    energies = product.energies
    energy_low_full = np.asarray(energies["e_low"].to_value(u.keV))
    energy_high_full = np.asarray(energies["e_high"].to_value(u.keV))

    if hasattr(product, "energy_masks"):
        energy_mask = product.energy_masks.energy_mask.astype(bool)
    else:
        energy_mask = np.ones(energy_low_full.shape, dtype=bool)
    energy_bin_idx = np.flatnonzero(energy_mask)
    n_E_file = energy_bin_idx.size

    # Nominal edges restricted to the bins present in the file.
    energy_low = energy_low_full[energy_mask]
    energy_high = energy_high_full[energy_mask]

    # ELUT edges in shape [n_E_file, n_pix, n_det]. e_actual is the inner
    # 31 channel edges; we pad to 32 channels then mask. The lowest channel
    # has an implicit lower edge at 0 and the highest has an implicit upper
    # edge at +∞ (NaN here, masked out of any used range in practice).
    elut = get_elut(product.time_range.center.datetime)
    # e_actual: [n_det, n_pix, 31] → transpose to [31, n_pix, n_det]
    e_actual_T = np.moveaxis(elut.e_actual, [0, 1, 2], [2, 1, 0])
    ebin_low_full = np.zeros((32, e_actual_T.shape[1], e_actual_T.shape[2]))
    ebin_low_full[1:] = e_actual_T
    ebin_high_full = np.zeros((32, e_actual_T.shape[1], e_actual_T.shape[2]))
    ebin_high_full[:-1] = e_actual_T
    ebin_high_full[-1] = np.nan
    energy_bin_low = ebin_low_full[energy_mask]
    energy_bin_high = ebin_high_full[energy_mask]

    # Determine which file-relative bins fall in `energy_range`.
    e_lo, e_hi = energy_range
    if hasattr(e_lo, "to_value"):
        e_lo = e_lo.to_value(u.keV)
    if hasattr(e_hi, "to_value"):
        e_hi = e_hi.to_value(u.keV)
    energy_ind = np.flatnonzero((energy_low >= e_lo) & (energy_high <= e_hi))
    if energy_ind.size == 0:
        raise ValueError(
            f"No energy bins in the product fall fully inside {energy_range}; "
            f"available file edges: {list(zip(energy_low, energy_high))}"
        )

    # Resolve sub-collimator default lazily.
    if subc_indices is None:
        subc_indices = np.arange(32)

    # Estimate sp_index / idx_peak from a spectrum if the caller didn't
    # supply them. Default spectrum: counts summed over time, detector,
    # pixel — units cancel in the log-log slopes, so this is enough.
    if sp_index is None or idx_peak is None:
        if spectrum is None:
            counts_arr = counts.value if hasattr(counts, "value") else np.asarray(counts)
            spectrum = counts_arr.sum(axis=(0, 1, 2))  # → (n_E_file,)
        sp_index_est, idx_peak_est = estimate_spectral_index(energy_low, energy_high, spectrum)
        if sp_index is None:
            sp_index = sp_index_est
        if idx_peak is None:
            idx_peak = idx_peak_est

    sp_index_arr = np.asarray(sp_index)
    if sp_index_arr.shape != (n_E_file,):
        raise ValueError(
            f"sp_index must have shape ({n_E_file},) matching the file's energy bins; got {sp_index_arr.shape}"
        )

    return _apply_elut_correction(
        counts=counts,
        counts_error=counts_error,
        energy_bin_idx=energy_bin_idx,
        energy_bin_low=energy_bin_low,
        energy_bin_high=energy_bin_high,
        energy_low=energy_low,
        energy_high=energy_high,
        energy_ind=energy_ind,
        sp_index=sp_index_arr,
        idx_peak=int(idx_peak),
        pixel_ind=np.atleast_1d(pixel_indices),
        subc_index=np.atleast_1d(subc_indices),
    )
