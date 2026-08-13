"""Tests for scripts/1_nucleus_segmentation.py.

Run unit tests (fast, no real data):
    python -m pytest tests/test_nucleus_segmentation.py -v -m "not slow"
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

# ---------------------------------------------------------------------------
# Load the script module (leading digit prevents plain import)
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location(
    "nucleus_segmentation", _SCRIPTS / "1_nucleus_segmentation.py"
)
_seg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_seg)

experiment_name_from_parquet = _seg.experiment_name_from_parquet
discover_parquets = _seg.discover_parquets
process_nucleus_site = _seg.process_nucleus_site


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiff(path: Path, img: np.ndarray) -> Path:
    tifffile.imwrite(str(path), img)
    return path


def _circle_blob(shape, cy, cx, r, bg=100, fg=3000, dtype=np.uint16):
    """Uniform background + one bright circular blob."""
    img = np.full(shape, bg, dtype=dtype)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = fg
    return img


def _run_site(plane_paths, row="A", col="1", frame="1"):
    """Compute a delayed segmentation task synchronously."""
    return process_nucleus_site(plane_paths, row, col, frame).compute(
        scheduler="synchronous"
    )


# ---------------------------------------------------------------------------
# experiment_name_from_parquet
# ---------------------------------------------------------------------------


class TestExperimentNameFromParquet:
    def test_column_takes_precedence(self):
        df = pd.DataFrame({"Experiment_name": ["myexp"]})
        assert experiment_name_from_parquet(Path("x_metadata.parquet"), df) == "myexp"

    def test_falls_back_to_stem_before_metadata_keyword(self):
        df = pd.DataFrame({"other_col": [1]})
        assert (
            experiment_name_from_parquet(Path("uuid123_metadata_20260806.parquet"), df)
            == "uuid123"
        )

    def test_falls_back_to_full_stem_when_no_metadata_keyword(self):
        assert (
            experiment_name_from_parquet(Path("plain_name.parquet"), pd.DataFrame())
            == "plain_name"
        )


# ---------------------------------------------------------------------------
# discover_parquets
# ---------------------------------------------------------------------------


class TestDiscoverParquets:
    def test_returns_all_without_filter(self, tmp_path):
        for name in ("exp_a.parquet", "exp_b.parquet"):
            (tmp_path / name).touch()
        assert len(discover_parquets(tmp_path)) == 2

    def test_filters_by_prefix(self, tmp_path):
        for name in ("exp_a.parquet", "exp_b.parquet", "other.parquet"):
            (tmp_path / name).touch()
        result = discover_parquets(tmp_path, selected=["exp_a"])
        assert len(result) == 1 and result[0].name == "exp_a.parquet"

    def test_warns_for_missing_name(self, tmp_path, capsys):
        result = discover_parquets(tmp_path, selected=["nonexistent"])
        assert result == []
        assert "no parquet found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Segmentation — happy path
# ---------------------------------------------------------------------------


def test_segment_detects_three_blobs(tmp_path):
    """Three well-separated bright blobs → ≥2 nuclei, correct output columns."""
    shape = (300, 300)
    img = np.full(shape, 100, dtype=np.uint16)
    for cy, cx in [(60, 60), (150, 150), (240, 240)]:
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 25 ** 2] = 4000

    tif = _make_tiff(tmp_path / "dapi.tif", img)
    df, status = _run_site([tif])

    assert len(df) >= 2, f"expected ≥2 nuclei, got {len(df)}"
    assert status["filter_status"] == "pass"
    assert {"area", "centroid-0", "centroid-1", "Row", "Column", "Frame"} <= set(df.columns)


def test_mip_over_multiple_planes(tmp_path):
    """Nucleus visible in only one z-plane must be detected via MIP."""
    shape = (200, 200)
    dark = np.full(shape, 100, dtype=np.uint16)
    bright = _circle_blob(shape, cy=100, cx=100, r=22)
    paths = []
    for i, img in enumerate([dark, bright, dark]):
        p = _make_tiff(tmp_path / f"z{i}.tif", img)
        paths.append(p)

    df, status = _run_site(paths)
    assert len(df) >= 1, "nucleus present in one z-plane must survive MIP"
    assert status["n_planes"] == 3


# ---------------------------------------------------------------------------
# MIN_AREA regression — the documented 3500→400 fix
# ---------------------------------------------------------------------------


def test_min_area_constant_is_400():
    """Guard: MIN_AREA must not silently revert to the miscalibrated 3500 value.

    Old value (3500) was calibrated for Panel_1 at 0.094 µm/px; hs-array runs
    at 0.2967 µm/px → 3500 px² ≈ 308 µm² floor, silently dropping ~87% of
    iNeuron nuclei. Correct floor for this dataset is 400 ≈ 35 µm².
    """
    assert _seg.MIN_AREA == 400, (
        f"MIN_AREA is {_seg.MIN_AREA}; expected 400 for hs-array at 0.2967 µm/px."
    )


def test_small_nucleus_survives_min_area_400(tmp_path):
    """A ~800 px² nucleus survives MIN_AREA=400 (>400) and is reported."""
    # radius=16 → area ≈ π*16² ≈ 804 px²  (valid iNeuron nucleus at 0.2967 µm/px)
    img = _circle_blob((250, 250), cy=125, cx=125, r=16)
    tif = _make_tiff(tmp_path / "small_nuc.tif", img)
    df, status = _run_site([tif])
    assert len(df) >= 1, (
        "~800 px² nucleus not detected with MIN_AREA=400. "
        "Likely cause: MIN_AREA reverted to 3500."
    )


def test_min_area_regression_3500_drops_same_nucleus(tmp_path, monkeypatch):
    """Confirm old MIN_AREA=3500 drops the same ~800 px² blob (regression proof)."""
    img = _circle_blob((250, 250), cy=125, cx=125, r=16)
    tif = _make_tiff(tmp_path / "reg.tif", img)
    monkeypatch.setattr(_seg, "MIN_AREA", 3500)
    df, status = _run_site([tif])
    assert len(df) == 0, (
        "Expected 0 nuclei with MIN_AREA=3500 (old miscalibrated value). "
        "Blob may be larger than 3500 px² — check test image size."
    )


# ---------------------------------------------------------------------------
# Edge / failure cases
# ---------------------------------------------------------------------------


def test_sub_min_area_blob_filtered_out(tmp_path):
    """A blob smaller than MIN_AREA (radius=5, area≈78 px²) yields 0 nuclei."""
    img = _circle_blob((200, 200), cy=100, cx=100, r=5)
    tif = _make_tiff(tmp_path / "tiny.tif", img)
    df, status = _run_site([tif])
    assert len(df) == 0
    assert status["filter_status"] == "no_nuclei"


def test_status_dict_contains_expected_keys(tmp_path):
    """Status dict must carry Row/Column/Frame/n_planes/n_nuclei keys."""
    img = _circle_blob((200, 200), cy=100, cx=100, r=22)
    tif = _make_tiff(tmp_path / "dapi.tif", img)
    _, status = _run_site([tif], row="B", col="3", frame="2")
    for key in ("Row", "Column", "Frame", "n_planes", "n_nuclei", "filter_status"):
        assert key in status, f"missing key: {key}"
    assert status["Row"] == "B"
    assert status["n_planes"] == 1
