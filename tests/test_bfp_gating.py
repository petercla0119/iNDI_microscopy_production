"""Tests for scripts/5_bfp_gating.py.

Run:
    python -m pytest tests/test_bfp_gating.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load the script module (leading digit prevents plain import)
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("bfp_gating", _SCRIPTS / "5_bfp_gating.py")
_bfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bfp)

warp_bfp = _bfp.warp_bfp
_valid_bg = _bfp._valid_bg
measure_bfp_cell = _bfp.measure_bfp_cell
gate_bfp = _bfp.gate_bfp
_um_to_px = _bfp._um_to_px

_R_NUC = _um_to_px(_bfp.R_NUCLEUS_UM)
_R_ADJ = _um_to_px(_bfp.R_ADJ_UM)
_R_CYTO = _um_to_px(_bfp.R_CYTO_UM)


def _soma_field(cy, cx, val, bg=100.0, r=_R_CYTO, H=400, W=400):
    """Synthetic BFP plane: background + a soma-filling disk (BFP fills the soma)."""
    img = np.full((H, W), bg, np.float32)
    yy, xx = np.ogrid[:H, :W]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = val
    return img


# ---------------------------------------------------------------------------
# warp_bfp + _valid_bg
# ---------------------------------------------------------------------------


def test_warp_restores_shifted_soma():
    """A soma offset by (dy,dx) in R1 lands back on its R2 centroid after warp."""
    cy, cx, dy, dx = 200, 200, 6, -4
    r1 = _soma_field(cy + dy, cx + dx, val=4000.0)
    warped = warp_bfp(r1, -dy, -dx)
    bg = _valid_bg(warped, -dy, -dx)
    on_centroid = measure_bfp_cell(warped, cy, cx, np.empty((0, 2)),
                                   _R_NUC, _R_ADJ, _R_CYTO, bg)
    assert on_centroid["bfp_cell_mean"] > 3000  # soma sits on the centroid


def test_valid_bg_ignores_zero_padded_border():
    """Background is read from the interior, not the shift's cval=0 border."""
    img = np.full((100, 100), 500.0, np.float32)
    warped = warp_bfp(img, 20, 20)          # a 20px zero border appears
    assert _valid_bg(warped, 20, 20) == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# measure_bfp_cell — whole-cell union
# ---------------------------------------------------------------------------


def test_whole_cell_area_matches_disk_union():
    """No neighbours: whole-cell mask ≈ the r_cyto disk (nucleus ∪ adjacent ∪ cyto)."""
    img = _soma_field(200, 200, val=1000.0)
    m = measure_bfp_cell(img, 200, 200, np.empty((0, 2)), _R_NUC, _R_ADJ, _R_CYTO, bg=100.0)
    expected = np.pi * _R_CYTO ** 2
    assert abs(m["bfp_cell_n"] - expected) / expected < 0.05


def test_neighbour_exclusion_shrinks_whole_cell():
    """A neighbour nucleus inside the annulus reduces the whole-cell pixel count."""
    img = _soma_field(200, 200, val=1000.0)
    no_nb = measure_bfp_cell(img, 200, 200, np.empty((0, 2)), _R_NUC, _R_ADJ, _R_CYTO, 100.0)
    nb = np.array([[200, 200 + int(_R_ADJ + 6)]], float)
    with_nb = measure_bfp_cell(img, 200, 200, nb, _R_NUC, _R_ADJ, _R_CYTO, 100.0)
    assert with_nb["bfp_cell_n"] < no_nb["bfp_cell_n"]


def test_integrated_is_bg_subtracted_sum():
    """bfp_integrated = background-subtracted whole-cell sum (= mean_bg * n)."""
    img = _soma_field(200, 200, val=1000.0)
    m = measure_bfp_cell(img, 200, 200, np.empty((0, 2)), _R_NUC, _R_ADJ, _R_CYTO, bg=100.0)
    assert m["bfp_integrated"] == pytest.approx(m["bfp_cell_mean_bg"] * m["bfp_cell_n"], rel=1e-6)


# ---------------------------------------------------------------------------
# gate_bfp — per-plate Otsu on log10(whole-cell mean)
# ---------------------------------------------------------------------------


def test_gate_splits_bimodal_population():
    """A bimodal set gates the bright cells BFP+ and the dim cells BFP-."""
    means = [110, 120, 130, 140, 150, 3000, 3200, 3400, 3600, 3800]
    df = pd.DataFrame({"bfp_cell_mean": np.array(means, float)})
    gated, thr = gate_bfp(df)
    assert (~gated["bfp_pos"].iloc[:5]).all(), "dim cells should gate BFP-"
    assert gated["bfp_pos"].iloc[5:].all(), "bright cells should gate BFP+"
    assert 150 < thr < 3000, f"threshold should fall in the antimode, got {thr}"


def test_gate_leaves_gmm_prob_nan_on_otsu_path():
    df = pd.DataFrame({"bfp_cell_mean": np.array([100.0, 200.0, 3000.0, 4000.0])})
    gated, _ = gate_bfp(df)
    assert gated["bfp_gmm_prob"].isna().all()


def test_gate_handles_degenerate_input():
    """Too few positive values → all BFP- rather than crashing."""
    df = pd.DataFrame({"bfp_cell_mean": np.array([np.nan, 0.0, -5.0])})
    gated, thr = gate_bfp(df)
    assert (~gated["bfp_pos"]).all()
    assert np.isnan(thr)
