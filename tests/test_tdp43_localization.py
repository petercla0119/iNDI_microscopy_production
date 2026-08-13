"""Tests for scripts/3_tdp43_localization.py.

Run:
    python -m pytest tests/test_tdp43_localization.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Load the script module (leading digit prevents plain import)
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location(
    "tdp43_localization", _SCRIPTS / "3_tdp43_localization.py"
)
_tdp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tdp)

_um_to_px = _tdp._um_to_px
pick_best_focus_plane = _tdp.pick_best_focus_plane
build_compartment_masks = _tdp.build_compartment_masks
_stats = _tdp._stats
measure_cell = _tdp.measure_cell

# Canonical radii in pixels (matching the script's live config)
_R_NUC = _um_to_px(_tdp.R_NUCLEUS_UM)
_R_ADJ = _um_to_px(_tdp.R_ADJ_UM)
_R_CYTO = _um_to_px(_tdp.R_CYTO_UM)


# ---------------------------------------------------------------------------
# _um_to_px
# ---------------------------------------------------------------------------


def test_um_to_px_matches_pixel_size():
    """Pixel size constant must be 0.2967 µm/px (from index.xml)."""
    assert _tdp.PIXEL_SIZE_UM == pytest.approx(0.2967)
    assert _um_to_px(4.5) == pytest.approx(4.5 / 0.2967)


# ---------------------------------------------------------------------------
# pick_best_focus_plane
# ---------------------------------------------------------------------------


def test_pick_best_focus_single_plane():
    assert pick_best_focus_plane([np.zeros((10, 10), np.uint16)]) == 0


def test_pick_best_focus_selects_sharpest():
    """The sharpest plane (highest Laplacian variance) must win."""
    rng = np.random.default_rng(42)
    # Checkerboard = high-frequency content → high Laplacian variance
    sharp = np.zeros((80, 80), np.uint16)
    sharp[::2, ::2] = 1000
    blurry = gaussian_filter(sharp.astype(float), sigma=6).astype(np.uint16)
    idx = pick_best_focus_plane([blurry, blurry, sharp, blurry])
    assert idx == 2


# ---------------------------------------------------------------------------
# build_compartment_masks
# ---------------------------------------------------------------------------


class TestBuildCompartmentMasks:
    H, W = 200, 200
    cy, cx = 100.0, 100.0
    r_nuc, r_adj, r_cyto = 15, 30, 60

    def _masks(self, others=None):
        yx = np.empty((0, 2)) if others is None else np.array(others, float)
        return build_compartment_masks(
            self.cy, self.cx, yx,
            self.r_nuc, self.r_adj, self.r_cyto,
            0, self.H, 0, self.W,
        )

    def test_masks_are_mutually_exclusive(self):
        nuc_m, adj_m, cyto_m = self._masks()
        assert not np.any(nuc_m & adj_m), "nucleus / adjacent overlap"
        assert not np.any(adj_m & cyto_m), "adjacent / cytoplasm overlap"
        assert not np.any(nuc_m & cyto_m), "nucleus / cytoplasm overlap"

    def test_nucleus_mask_area_matches_circle(self):
        nuc_m, _, _ = self._masks()
        expected = np.pi * self.r_nuc ** 2
        assert abs(nuc_m.sum() - expected) / expected < 0.05  # within 5%

    def test_neighbour_exclusion_shrinks_cytoplasm(self):
        _, _, cyto_no_nb = self._masks()
        # Neighbour nucleus centred inside the cytoplasm annulus
        nb = [[100, 100 + int(self.r_adj + 8)]]
        _, _, cyto_with_nb = self._masks(others=nb)
        assert cyto_with_nb.sum() < cyto_no_nb.sum()

    def test_neighbour_at_nucleus_position_does_not_affect_nucleus_mask(self):
        """Neighbour exclusion only punches out ring+annulus, not the nucleus mask."""
        nuc_no_nb, _, _ = self._masks()
        nb = [[100, 100 + int(self.r_adj + 8)]]
        nuc_with_nb, _, _ = self._masks(others=nb)
        assert nuc_with_nb.sum() == nuc_no_nb.sum()


# ---------------------------------------------------------------------------
# _stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_empty_array_returns_nan_and_zero_sum(self):
        s = _stats(np.array([]), bg=100.0)
        assert s["n"] == 0
        assert np.isnan(s["mean"])
        assert np.isnan(s["median"])
        assert s["sum"] == 0.0
        assert np.isnan(s["mean_bg"])

    def test_known_values(self):
        vals = np.array([100.0, 200.0, 300.0])
        s = _stats(vals, bg=50.0)
        assert s["mean"] == pytest.approx(200.0)
        assert s["median"] == pytest.approx(200.0)
        assert s["n"] == 3
        assert s["mean_bg"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# measure_cell — N:C ratio and three-zone logic
# ---------------------------------------------------------------------------


def _field(nuc_val, cyto_val, bg_val=100.0, H=200, W=200):
    """Synthetic TDP-43 plane: background + nuclear disk + cytoplasm annulus."""
    img = np.full((H, W), bg_val, np.float32)
    yy, xx = np.ogrid[:H, :W]
    d2 = (yy - 100) ** 2 + (xx - 100) ** 2
    img[d2 <= _R_NUC ** 2] = nuc_val
    img[(d2 > _R_ADJ ** 2) & (d2 <= _R_CYTO ** 2)] = cyto_val
    return img


class TestMeasureCell:
    """Centroid at (100, 100) in a 200×200 field, no neighbours by default."""
    _empty_nb = np.empty((0, 2))

    def _measure(self, img, cy=100.0, cx=100.0, others=None, bg=100.0):
        nb = self._empty_nb if others is None else np.array(others, float)
        return measure_cell(img, cy, cx, nb, _R_NUC, _R_ADJ, _R_CYTO, bg)

    def test_healthy_nc_above_threshold(self):
        """High nuclear / low cytoplasmic TDP-43 → N:C > 1.5 (healthy)."""
        m = self._measure(_field(nuc_val=1000, cyto_val=200))
        assert m["NC_ratio"] > 1.5, f"healthy N:C should be > 1.5, got {m['NC_ratio']:.3f}"

    def test_mislocalized_nc_below_threshold(self):
        """Low nuclear / high cytoplasmic TDP-43 → N:C < 0.7 (mislocalized)."""
        m = self._measure(_field(nuc_val=200, cyto_val=1000))
        assert m["NC_ratio"] < 0.7, f"mislocalized N:C should be < 0.7, got {m['NC_ratio']:.3f}"

    def test_nc_is_nan_when_cyto_at_background(self):
        """Cytoplasm mean ≈ background → denominator near-zero → N:C = nan."""
        img = np.full((200, 200), 100.0, np.float32)
        m = self._measure(img, bg=100.0)
        assert np.isnan(m["NC_ratio"])

    def test_all_compartments_have_pixels(self):
        """Nucleus, adjacent, and cytoplasm zones must all contain > 0 pixels."""
        m = self._measure(_field(nuc_val=1000, cyto_val=200))
        assert m["tdp_nuc_n"] > 0, "nucleus compartment is empty"
        assert m["tdp_adj_n"] > 0, "adjacent compartment is empty"
        assert m["tdp_cyto_n"] > 0, "cytoplasm compartment is empty"

    def test_neighbour_exclusion_reduces_cyto_pixel_count(self):
        """A neighbouring nucleus inside the annulus must reduce cyto pixel count."""
        img = np.full((200, 200), 500.0, np.float32)
        m_no_nb = self._measure(img, bg=100.0)
        nb = [[100, 100 + int(_R_ADJ + 6)]]
        m_with_nb = self._measure(img, others=nb, bg=100.0)
        assert m_with_nb["tdp_cyto_n"] < m_no_nb["tdp_cyto_n"], (
            "Neighbour nucleus exclusion did not shrink the cytoplasm mask"
        )

    def test_nc_ratio_magnitude_matches_intensity_ratio(self):
        """N:C ratio must approximate (nuc_intensity - bg) / (cyto_intensity - bg)."""
        nuc_val, cyto_val, bg = 800.0, 300.0, 100.0
        m = self._measure(_field(nuc_val=nuc_val, cyto_val=cyto_val), bg=bg)
        expected = (nuc_val - bg) / (cyto_val - bg)  # = 700/200 = 3.5
        assert m["NC_ratio"] == pytest.approx(expected, rel=0.05)
