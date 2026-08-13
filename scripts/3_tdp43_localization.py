"""Step 3 — TDP-43 three-zone localization (Round 2 only, no cross-round alignment).

Answers JH's milestone: per cell, measure TDP-43 (CF568 / 546 nm) intensity in
three concentric compartments around each Step-1 nucleus centroid —

    nucleus (mislocalized-in-nucleus) · nuclear-adjacent ring · cytoplasm annulus

and derive the nuclear-to-cytoplasmic (N:C) ratio, the primary mislocalization
readout (high = healthy nuclear TDP-43; low = cytoplasmic mislocalization).

This is a fork of Organelle_segmentation/Panel_1/organelle_seg_batch_20260511.py
::process_site — it keeps that file's per-nucleus bbox + circular-ROI + radial
idioms but replaces organelle segmentation with fixed concentric compartment
masks and raw per-compartment intensity.

ponytail: concentric disks centered on the centroid are a first-pass proxy for
the true nuclear footprint (nuclei aren't circles). Upgrade path = load a real
DAPI nuclear mask (dilate/erode the Step-1 label image) instead of R_NUCLEUS
disks. Everything downstream (compartment intensities, N:C ratio) is unchanged
by that swap — only build_compartment_masks() changes.

Z-handling (data is a widefield stack: 10 z-planes, ~2 um spacing, ~20 um deep):
  - Nucleus detection: Step 1 segments the DAPI *MIP over all 10 z-planes* (MIP
    collapses best-in-focus nuclear signal — good for finding nuclei). We reuse
    only its centroids here.
  - Compartment masks (nucleus/adjacent/cytoplasm): 2D concentric disks/annuli
    around each centroid — z-agnostic geometry, not segmented from pixels.
  - TDP-43 intensity: read from the SINGLE best-focus z-plane per field
    (pick_best_focus_plane, max variance-of-Laplacian). NOT the MIP (max-proj
    inflates intensity) and NOT a z-sum. One plane is chosen per field, not per
    cell, so cells off that plane read softer — a known noise source in N:C.

Inputs (both produced upstream, TIFF-based, so nuclei match Step 1 exactly):
  - Step 0 metadata parquet  (outputs/image_metadata/<uuid>_metadata_*.parquet)
  - Step 1 nuclei features   (outputs/nuclei_features/<uuid>_nuclei_features_*.parquet)

Run:
  python 3_tdp43_localization.py                     # all experiments
  python 3_tdp43_localization.py -e 5c6bf125         # one experiment prefix
  python 3_tdp43_localization.py --selfcheck         # synthetic-data unit check
"""

import argparse
from datetime import datetime
from pathlib import Path

import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import tifffile
from skimage.filters import laplace

# --- Configuration ---------------------------------------------------------

PIXEL_SIZE_UM = 0.2967          # from index.xml; NOT the 64.5 µm TIFF-tag lie
TDP43_CHANNEL = "CF568"         # 546 nm; XML channel name (see intake note)

# Concentric compartment radii (µm). Centroid-relative. Tunable knobs — the
# physical scale of an iNeuron nucleus/soma; recalibrate against QC montages.
R_NUCLEUS_UM = 4.5              # inner disk  ≈ nuclear region
R_ADJ_UM     = 9.0             # outer bound of the nuclear-adjacent ring
R_CYTO_UM    = 18.0            # outer bound of the cytoplasm annulus

BG_PERCENTILE = 5              # per-field background estimate for the TDP channel

DEFAULT_META_DIR = Path("/Users/pmihack/claire/hs-array/outputs/image_metadata")
DEFAULT_NUC_DIR  = Path("/Users/pmihack/claire/hs-array/outputs/nuclei_features")
DEFAULT_OUT_DIR  = Path("/Users/pmihack/claire/hs-array/outputs/tdp43_localization")


def _um_to_px(um):
    return um / PIXEL_SIZE_UM


# --- Core measurement ------------------------------------------------------

def pick_best_focus_plane(planes):
    """Index of the sharpest z-plane (max variance of Laplacian). Widefield z
    stacks are mostly out of focus; MIP inflates intensity, so quantify on the
    single best-focus plane (2026-08-03 decision)."""
    if len(planes) == 1:
        return 0
    return int(np.argmax([np.var(laplace(p.astype(np.float32))) for p in planes]))


def build_compartment_masks(cy, cx, others_yx, r_nuc, r_adj, r_cyto,
                            y0, y1, x0, x1):
    """Three concentric masks for one cell within a bbox, excluding the footprint
    of neighbouring nuclei from the ring and annulus (JH: rings must not swallow
    adjacent nuclei)."""
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2

    nucleus = d2 <= r_nuc ** 2
    adjacent = (d2 > r_nuc ** 2) & (d2 <= r_adj ** 2)
    cytoplasm = (d2 > r_adj ** 2) & (d2 <= r_cyto ** 2)

    # Punch out neighbour nuclei (their R_NUCLEUS disks) from ring + annulus.
    if len(others_yx):
        other = np.zeros_like(nucleus)
        for oy, ox in others_yx:
            other |= (yy - oy) ** 2 + (xx - ox) ** 2 <= r_nuc ** 2
        adjacent &= ~other
        cytoplasm &= ~other

    return nucleus, adjacent, cytoplasm


def _stats(vals, bg):
    """mean/median/sum/std/n for a compartment, plus background-subtracted mean."""
    if vals.size == 0:
        return dict(mean=np.nan, median=np.nan, sum=0.0, std=np.nan, n=0, mean_bg=np.nan)
    mean = float(vals.mean())
    return dict(
        mean=mean, median=float(np.median(vals)), sum=float(vals.sum()),
        std=float(vals.std()), n=int(vals.size), mean_bg=mean - bg,
    )


def measure_cell(tdp, cy, cx, others_yx, r_nuc, r_adj, r_cyto, bg):
    """Per-compartment TDP-43 stats + N:C ratio for one nucleus."""
    H, W = tdp.shape
    yi, xi = int(round(cy)), int(round(cx))
    R = int(np.ceil(r_cyto))
    y0, y1 = max(yi - R, 0), min(yi + R + 1, H)
    x0, x1 = max(xi - R, 0), min(xi + R + 1, W)
    tile = tdp[y0:y1, x0:x1]

    nuc_m, adj_m, cyto_m = build_compartment_masks(
        cy, cx, others_yx, r_nuc, r_adj, r_cyto, y0, y1, x0, x1)

    nuc = _stats(tile[nuc_m], bg)
    adj = _stats(tile[adj_m], bg)
    cyto = _stats(tile[cyto_m], bg)

    # N:C ratio on background-subtracted means (the mislocalization readout).
    denom = cyto["mean_bg"]
    nc = (nuc["mean_bg"] / denom) if (denom is not None and denom > 1e-6) else np.nan
    adj_nuc = (adj["mean_bg"] / nuc["mean_bg"]) if (nuc.get("mean_bg") and nuc["mean_bg"] > 1e-6) else np.nan

    row = {"NC_ratio": nc, "adj_nuc_ratio": adj_nuc, "bg_tdp": bg}
    for name, s in (("nuc", nuc), ("adj", adj), ("cyto", cyto)):
        for k, v in s.items():
            row[f"tdp_{name}_{k}"] = v
    return row


@delayed
def process_site(tdp_plane_paths, nuc_site, r_nuc, r_adj, r_cyto):
    """One (well, frame) site: best-focus TDP plane, then measure every nucleus."""
    planes = [tifffile.imread(p) for p in tdp_plane_paths]
    bf = pick_best_focus_plane(planes)
    tdp = planes[bf].astype(np.float32)
    bg = float(np.percentile(tdp, BG_PERCENTILE))

    ys = nuc_site["centroid-0"].to_numpy(float)
    xs = nuc_site["centroid-1"].to_numpy(float)
    labels = nuc_site["label"].to_numpy(int)
    centroids = np.column_stack([ys, xs])

    rows = []
    for i, (cy, cx, lab) in enumerate(zip(ys, xs, labels)):
        others = np.delete(centroids, i, axis=0)
        # keep only neighbours that can reach this cell's annulus
        if len(others):
            near = np.abs(others - [cy, cx]).max(axis=1) <= (r_cyto + r_nuc)
            others = others[near]
        m = measure_cell(tdp, cy, cx, others, r_nuc, r_adj, r_cyto, bg)
        m.update({
            "Row": nuc_site["Row"].iloc[0], "Column": nuc_site["Column"].iloc[0],
            "Frame": nuc_site["Frame"].iloc[0], "Nucleus_ID": int(lab),
            "centroid-0": cy, "centroid-1": cx, "best_focus_plane": bf,
        })
        rows.append(m)
    return pd.DataFrame(rows)


# --- IO / orchestration ----------------------------------------------------

def _experiment_name(nuc_path):
    stem = nuc_path.stem
    # handles both old "uuid_nuclei_features_YYYYMMDD" and new "uuid_nuclei_features"
    return stem.split("_nuclei_features")[0] if "_nuclei_features" in stem else stem


def _find_metadata(meta_dir, exp_name):
    hits = sorted(meta_dir.glob(f"{exp_name}*metadata*.parquet"))
    return hits[0] if hits else None


def process_experiment(nuc_path, meta_dir, out_dir, today, scheduler):
    exp = _experiment_name(nuc_path)
    meta_path = _find_metadata(meta_dir, exp)
    if meta_path is None:
        print(f"[warning] no metadata parquet for {exp}, skipping.")
        return None

    meta = pd.read_parquet(meta_path)
    tdp = meta[meta["Channel_name"] == TDP43_CHANNEL].copy()
    if tdp.empty:
        avail = sorted(meta["Channel_name"].dropna().unique())
        print(f"[error] channel {TDP43_CHANNEL!r} not in {exp}; available: {avail}")
        return None

    nuc = pd.read_parquet(nuc_path)
    nuc_by_site = dict(tuple(nuc.groupby(["Row", "Column", "Frame"])))
    tdp_grouped = tdp.sort_values("Plane").groupby(["Row", "Column", "Frame"])

    r_nuc, r_adj, r_cyto = _um_to_px(R_NUCLEUS_UM), _um_to_px(R_ADJ_UM), _um_to_px(R_CYTO_UM)
    tasks = [
        process_site(list(grp["filepath"]), nuc_by_site[site], r_nuc, r_adj, r_cyto)
        for site, grp in tdp_grouped if site in nuc_by_site
    ]
    print(f"{exp}: {len(tasks)} sites with nuclei (TDP-43 {TDP43_CHANNEL}, "
          f"radii px: nuc={r_nuc:.1f} adj={r_adj:.1f} cyto={r_cyto:.1f})")
    with ProgressBar():
        dfs = dask.compute(*tasks, scheduler=scheduler)

    feats = pd.concat(dfs, ignore_index=True)
    feats["Measurement_ID"] = nuc.get("Measurement_ID", pd.Series([exp])).iloc[0] \
        if "Measurement_ID" in nuc.columns else exp

    out_path = out_dir / f"{exp}_tdp43_localization_{today}.parquet"
    feats.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}  ({len(feats)} cells, "
          f"median N:C = {feats['NC_ratio'].median():.2f})")
    return feats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("nuc_dir", nargs="?", type=Path, default=DEFAULT_NUC_DIR,
                   help=f"Step-1 nuclei-feature parquet dir (default: {DEFAULT_NUC_DIR})")
    p.add_argument("-m", "--meta-dir", type=Path, default=DEFAULT_META_DIR,
                   help=f"Step-0 metadata parquet dir (default: {DEFAULT_META_DIR})")
    p.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("-e", "--experiments", nargs="+", default=None,
                   metavar="NAME", help="experiment-name prefixes to process")
    p.add_argument("-s", "--scheduler", default="processes",
                   choices=["processes", "threads", "single-threaded", "synchronous"])
    p.add_argument("--selfcheck", action="store_true", help="run synthetic unit check and exit")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selfcheck:
        return _selfcheck()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    nuc_parquets = sorted(p for p in args.nuc_dir.glob("*.parquet")
                          if "_nuclei_features" in p.name and "_image_qc" not in p.name)
    if args.experiments:
        nuc_parquets = [p for p in nuc_parquets
                        if any(p.name.startswith(e) for e in args.experiments)]
    print(f"Found {len(nuc_parquets)} nuclei-feature parquet(s).")
    for nuc_path in nuc_parquets:
        process_experiment(nuc_path, args.meta_dir, args.output_dir, today, args.scheduler)


# --- Self-check ------------------------------------------------------------

def _selfcheck():
    """Synthetic field: a healthy (nuclear TDP-43) cell must score N:C > 1;
    a mislocalized (cytoplasmic TDP-43) cell must score N:C < 1."""
    r_nuc, r_adj, r_cyto = _um_to_px(R_NUCLEUS_UM), _um_to_px(R_ADJ_UM), _um_to_px(R_CYTO_UM)
    H = W = 200
    yy, xx = np.ogrid[:H, :W]

    def field(nuc_val, cyto_val, cy=100, cx=100):
        img = np.full((H, W), 100.0, np.float32)   # background
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        img[d2 <= r_nuc ** 2] = nuc_val
        img[(d2 > r_adj ** 2) & (d2 <= r_cyto ** 2)] = cyto_val
        return img

    bg = 100.0
    healthy = measure_cell(field(1000, 200), 100, 100, np.empty((0, 2)),
                           r_nuc, r_adj, r_cyto, bg)
    mislocal = measure_cell(field(200, 1000), 100, 100, np.empty((0, 2)),
                            r_nuc, r_adj, r_cyto, bg)

    assert healthy["NC_ratio"] > 1.5, f"healthy N:C should be high, got {healthy['NC_ratio']}"
    assert mislocal["NC_ratio"] < 0.7, f"mislocalized N:C should be low, got {mislocal['NC_ratio']}"
    assert healthy["tdp_nuc_n"] > 0 and healthy["tdp_cyto_n"] > 0, "empty compartments"

    # neighbour exclusion: a neighbour nucleus disk must shrink the annulus
    others = np.array([[100, 100 + int(r_adj + 5)]])
    with_nb = measure_cell(field(500, 500), 100, 100, others, r_nuc, r_adj, r_cyto, bg)
    without = measure_cell(field(500, 500), 100, 100, np.empty((0, 2)), r_nuc, r_adj, r_cyto, bg)
    assert with_nb["tdp_cyto_n"] < without["tdp_cyto_n"], "neighbour exclusion did nothing"

    print(f"selfcheck OK — healthy N:C={healthy['NC_ratio']:.2f}, "
          f"mislocalized N:C={mislocal['NC_ratio']:.2f}, "
          f"neighbour exclusion removed {without['tdp_cyto_n'] - with_nb['tdp_cyto_n']} px")


if __name__ == "__main__":
    main()
