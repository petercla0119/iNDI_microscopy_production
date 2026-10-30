"""Tests for the saturation QC in scripts/0_img_metadata.py.

    python -m pytest tests/test_img_metadata.py -v
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

_SCRIPTS = Path(__file__).parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("img_metadata", _SCRIPTS / "0_img_metadata.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def _tiff(path, img):
    tifffile.imwrite(str(path), img)
    return path


def test_saturation_stats_counts_ceiling_pixels(tmp_path):
    img = np.full((100, 100), 500, dtype=np.uint16)
    img[:10, :10] = 65535                                  # 1% of pixels clipped
    mx, frac = _m.saturation_stats(_tiff(tmp_path / "a.tiff", img))
    assert mx == 65535 and frac == 0.01


def test_saturation_stats_clean_image_is_zero(tmp_path):
    img = np.full((50, 50), 65534, dtype=np.uint16)        # bright but NOT at ceiling
    assert _m.saturation_stats(_tiff(tmp_path / "b.tiff", img)) == (65534, 0.0)


def test_summary_flags_only_clipped_channel(tmp_path):
    good = _tiff(tmp_path / "g.tiff", np.full((50, 50), 1000, dtype=np.uint16))
    bad = np.full((50, 50), 1000, dtype=np.uint16); bad[:5, :] = 65535   # 10% clipped
    bad = _tiff(tmp_path / "bad.tiff", bad)
    df = pd.DataFrame({"filepath": [good, good, bad, bad],
                       "Channel_name": ["CF568", "CF568", "DAPI", "DAPI"]})
    df = _m.add_saturation(df, n_workers=1)
    assert set(df.columns) >= {"img_max", "sat_frac"}
    table, flagged = _m.saturation_summary(df)
    assert flagged == ["DAPI"]
    assert table.loc["CF568", "flag"] == False and table.loc["CF568", "frac_images_clipped"] == 0.0  # noqa: E712
    # the summary must survive the JSON dump StageRecorder.finish does
    json.dumps({"saturation": table.round(5).to_dict(orient="index"), "overexposed_channels": flagged})


def test_threshold_edge_below_warn_not_flagged(tmp_path):
    img = np.full((100, 100), 100, dtype=np.uint16); img[0, :5] = 65535   # 0.05% < 0.1%
    p = _tiff(tmp_path / "e.tiff", img)
    df = _m.add_saturation(pd.DataFrame({"filepath": [p], "Channel_name": ["DAPI"]}), n_workers=1)
    _, flagged = _m.saturation_summary(df)
    assert flagged == []
