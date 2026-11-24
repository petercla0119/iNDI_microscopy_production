"""Per-experiment pixel size and flatfield key come from Step 0 metadata (40x defects D1-D3).

Steps 1/3/5 used to hardcode D28 Rep1 values (0.2967 µm/px; NPZ keys ch1/ch3/ch1).
On the 40x WARD00048 export the pixel size is 0.1483 µm/px and DAPI is ChannelID 4,
CF568 ChannelID 2, so the hardcoded values silently gave the wrong floor, radii
and flatfield. These tests pin the metadata-driven resolution.

    python -m pytest tests/test_imaging_params.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from dask import delayed

_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import run_utils as ru  # noqa: E402


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_seg = _load("seg_params", "1_nucleus_segmentation.py")
_s3 = _load("s3_params", "3_tdp43_localization.py")
_s5 = _load("s5_params", "5_bfp_gating.py")

PX_20X, PX_40X = 0.2967, 0.14834
# Harmony ChannelID order on the two exports the screen has seen
D28_IDS = {"DAPI": 1, "Alexa 488": 2, "CF568": 3, "Alexa 647": 4}
W48_IDS = {"Alexa 488": 1, "CF568": 2, "Alexa 647": 3, "DAPI": 4}


def _meta(ids, res_x=PX_20X):
    return pd.DataFrame({"Channel_name": list(ids), "ChannelID": list(ids.values()),
                         "res_x": res_x})


class _Rec:
    def log(self, *a, **k):
        pass


# --- run_utils helpers -----------------------------------------------------

class TestPixelSize:
    def test_reads_res_x(self):
        assert ru.pixel_size_um(_meta(W48_IDS, PX_40X)) == PX_40X

    def test_override_wins(self):
        assert ru.pixel_size_um(_meta(W48_IDS, PX_40X), override=0.5) == 0.5

    def test_missing_column_is_none(self):
        assert ru.pixel_size_um(pd.DataFrame({"Channel_name": ["DAPI"]})) is None

    def test_mixed_res_x_raises(self):
        m = pd.concat([_meta(D28_IDS, PX_20X), _meta(D28_IDS, PX_40X)])
        with pytest.raises(ValueError, match="mixed res_x"):
            ru.pixel_size_um(m)


class TestIcNpzKey:
    @pytest.mark.parametrize("ids,channel,key", [
        (D28_IDS, "DAPI", "ch1"), (D28_IDS, "CF568", "ch3"),      # old hardcoded values hold on D28
        (W48_IDS, "DAPI", "ch4"), (W48_IDS, "CF568", "ch2"),      # and are wrong on WARD00048
    ])
    def test_key_follows_channel_id(self, ids, channel, key):
        assert ru.ic_npz_key(_meta(ids), channel) == key

    def test_override_wins(self):
        assert ru.ic_npz_key(_meta(W48_IDS), "DAPI", override="ch9") == "ch9"

    def test_absent_channel_raises(self):
        with pytest.raises(ValueError, match="TagFP"):
            ru.ic_npz_key(_meta(W48_IDS), "TagFP")

    def test_load_ic_field(self):
        npz = {"ch4": np.ones((2, 2)), "bg_ch4": np.array(112.0), "ch2": np.ones((2, 2))}
        field, bg = ru.load_ic_field(npz, "ch4")
        assert field.shape == (2, 2) and bg == 112.0
        assert ru.load_ic_field(npz, "ch2")[1] is None
        with pytest.raises(KeyError, match="ch1"):
            ru.load_ic_field(npz, "ch1")


# --- Step 1: debris floor --------------------------------------------------

class TestMinArea:
    def test_d28_keeps_400(self):
        assert _seg.min_area_px(PX_20X) == 400 == _seg.MIN_AREA

    def test_40x_scales_to_physical_floor(self):
        assert _seg.min_area_px(PX_40X) == 1600

    def test_unknown_pixel_size_falls_back(self):
        assert _seg.min_area_px(None) == _seg.MIN_AREA


def _blob_field(shape=(256, 256)):
    """Background + a small (r=15, ~700 px²) and a large (r=35, ~3,850 px²) nucleus."""
    img = np.full(shape, 100, np.uint16)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    img[(yy - 60) ** 2 + (xx - 60) ** 2 <= 15 ** 2] = 3000
    img[(yy - 170) ** 2 + (xx - 170) ** 2 <= 35 ** 2] = 3000
    return img


def _step1_inputs(tmp_path, res_x, ids):
    tif = tmp_path / "r01c01f01p01-ch4t01.tiff"
    tifffile.imwrite(str(tif), _blob_field())
    meta = pd.DataFrame({"filepath": [str(tif)], "filename": [tif.name], "subdirectory": ["."],
                         "Row": [1], "Column": [1], "Frame": [1], "Plane": [1],
                         "Channel_name": ["DAPI"], "ChannelID": [ids["DAPI"]], "res_x": [res_x],
                         "Experiment_name": ["exp"]})
    pq = tmp_path / "exp_metadata.parquet"
    meta.to_parquet(pq, index=False)
    out = tmp_path / "out"; out.mkdir()
    return pq, out


@pytest.mark.parametrize("res_x,ids,n_expected,floor,key", [
    (PX_20X, D28_IDS, 2, 400, "ch1"),     # D28: both blobs clear 400 px²
    (PX_40X, W48_IDS, 1, 1600, "ch4"),    # 40x: the ~700 px² blob is debris
])
def test_step1_floor_and_key_from_metadata(tmp_path, res_x, ids, n_expected, floor, key):
    pq, out = _step1_inputs(tmp_path, res_x, ids)
    npz = {f"ch{i}": np.ones((256, 256)) for i in range(1, 5)}
    npz.update({f"bg_ch{i}": np.array(0.0) for i in range(1, 5)})
    stats = _seg.process_metadata_parquet(pq, out, "synchronous", _Rec(), npz=npz)
    assert (stats["min_area_px"], stats["ic_npz_key"]) == (floor, key)
    nuclei = pd.read_parquet(out / "exp_nuclei_features.parquet")
    assert len(nuclei) == n_expected
    assert nuclei["area"].min() >= floor


def test_step1_pixel_size_override(tmp_path):
    pq, out = _step1_inputs(tmp_path, PX_20X, D28_IDS)
    stats = _seg.process_metadata_parquet(pq, out, "synchronous", _Rec(), pixel_size=PX_40X)
    assert stats["min_area_px"] == 1600


# --- Step 3: radii + flatfield key ----------------------------------------

def test_step3_radii_and_key_from_metadata(tmp_path, monkeypatch):
    seen = {}

    @delayed
    def fake_site(paths, nuc_site, r_nuc, r_adj, r_cyto, ic_field=None, darkfield=None):
        seen.update(r=(r_nuc, r_adj, r_cyto), ic_mean=float(np.mean(ic_field)), bg=darkfield)
        return pd.DataFrame({"NC_ratio": [1.0]})

    monkeypatch.setattr(_s3, "process_site", fake_site)
    meta_dir = tmp_path / "meta"; meta_dir.mkdir()
    pd.DataFrame({"filepath": ["x.tiff"], "Row": [1], "Column": [1], "Frame": [1], "Plane": [1],
                  "Channel_name": ["CF568"], "ChannelID": [W48_IDS["CF568"]], "res_x": [PX_40X]}
                 ).to_parquet(meta_dir / "exp_metadata.parquet", index=False)
    nuc_path = tmp_path / "exp_nuclei_features.parquet"
    pd.DataFrame({"Row": [1], "Column": [1], "Frame": [1]}).to_parquet(nuc_path, index=False)
    npz = {"ch2": np.full((4, 4), 2.0), "bg_ch2": np.array(7.0), "ch3": np.full((4, 4), 9.0),
           "bg_ch3": np.array(9.0)}

    feats = _s3.process_experiment(nuc_path, meta_dir, tmp_path, "20260101", "synchronous", npz=npz)

    assert seen["r"] == pytest.approx((4.5 / PX_40X, 9.0 / PX_40X, 18.0 / PX_40X))
    assert (seen["ic_mean"], seen["bg"]) == (2.0, 7.0)          # ch2, not the old ch3
    assert feats["pixel_size_um"].iloc[0] == PX_40X


# --- Step 5: radii from the Step 3 scale, BFP key from R1 metadata ---------

def test_step5_radii_and_key(tmp_path, monkeypatch):
    seen = {}

    @delayed
    def fake_measure(paths, shift, ncc, status, nuc_site, r_nuc, r_adj, r_cyto, ic_field=None):
        seen.update(r_nuc=r_nuc, ic_mean=float(np.mean(ic_field)))
        return pd.DataFrame({"bfp_cell_mean": [100.0, 4000.0], "Row": 1, "Column": 1,
                             "Frame": 1, "Nucleus_ID": [0, 1]})

    reg = tmp_path / "reg.parquet"; tdp = tmp_path / "tdp.parquet"; meta = tmp_path / "r1.parquet"
    pd.DataFrame({"Row": [1], "Column": [1], "Frame": [1], "shift_y": [0.0], "shift_x": [0.0],
                  "ncc": [0.9], "reg_status": ["ok"], "Round1_UUID": ["r1uuid"]}).to_parquet(reg)
    pd.DataFrame({"Row": [1], "Column": [1], "Frame": [1], "pixel_size_um": [PX_40X]}).to_parquet(tdp)
    pd.DataFrame({"filepath": ["b.tiff"], "Row": [1], "Column": [1], "Frame": [1], "Plane": [1],
                  "Channel_name": ["DAPI"], "ChannelID": [4], "res_x": [PX_40X]}).to_parquet(meta)
    monkeypatch.setattr(_s5, "measure_bfp_site", fake_measure)
    monkeypatch.setattr(_s5, "_find_reg", lambda u, d: reg)
    monkeypatch.setattr(_s5, "_find_tdp", lambda u, d: tdp)
    monkeypatch.setattr(_s5, "_find_r1_metadata", lambda u, d=None: meta)
    npz = {"ch1": np.full((4, 4), 9.0), "ch4": np.full((4, 4), 3.0)}

    _s5.process_plate("r2uuid", tmp_path, "20260101", "synchronous", tmp_path, tmp_path,
                      meta_dir=tmp_path, npz=npz)

    assert seen["r_nuc"] == pytest.approx(4.5 / PX_40X)
    assert seen["ic_mean"] == 3.0                                # ch4 = R1 BFP's ChannelID, not ch1
