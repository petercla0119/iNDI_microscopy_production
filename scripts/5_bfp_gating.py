"""Step 5 — BFP whole-cell intensity + BFP+/- gate (Phase F).

Attaches a per-cell BFP (guide-delivery) measurement to every Round-2-segmented
nucleus. Round 1 carries the 405/BFP guide-delivery readout; Round 2 is the
analysed round (TDP-43 N:C). Using the per-field R1->R2 transform from Step 4
(4_cross_round_registration.py), we warp the R1 BFP plane into R2 pixel space,
measure BFP over the whole-cell region around each R2 nucleus centroid, and call
each cell BFP+ / BFP- from the per-plate bimodal (untransduced vs transduced)
intensity distribution. BFP- cells within transduced wells are the ideal
within-well internal control (Phase G's 2x2: (NT vs gene) x (BFP- vs BFP+)).

Channel facts (load-bearing; select by Channel_name string, never index):
  - BFP is Channel_name == "DAPI" in the R1 metadata (Harmony labels the 375/456
    nm BFP channel "DAPI"). In R2, "DAPI" is the real nuclear stain — not used
    here. We read BFP from the R1 parquet's "DAPI" rows only.

Reuses Step-3 machinery verbatim (single source of truth): pick_best_focus_plane,
build_compartment_masks, _stats, _um_to_px, and the R_* radii / BG_PERCENTILE.
The only genuinely new code is: warp the BFP plane (scipy.ndimage.shift with the
Step-4 shift), measure the mask UNION (nucleus | adjacent | cytoplasm) instead of
three separate compartments, and the per-plate Otsu-on-log gate.

ponytail: gate = Otsu on log10(whole-cell mean BFP). scikit-learn is NOT in the
.venv, so the plan's 2-component GaussianMixture path is unavailable — Otsu-on-log
is the plan's own recommended first cut anyway. Both write the same bfp_pos +
continuous columns, so a later GMM swap is one function (gate_bfp). Upgrade when
the histogram shows Otsu misplacing the antimode.

ponytail: whole-cell region = the concentric disk-union (r_cyto = 18 um), the
same proxy Step 3 uses. It over-counts crowded fields, but a binary BFP call is
robust to modest contamination. Inherits the real cell mask automatically when
build_compartment_masks() is upgraded — one swap, both TDP-43 and BFP benefit.

WARNING: flatfield is still unapplied (Phase A open). Vignetting makes edge cells
read dim -> biases both bfp_integrated and the gate. Fine for a first qualitative
pass; apply flatfield / regress out field-XY before quantitative hit calling.

Inputs (all on disk, TIFF-based so fields line up with Steps 1/3/4):
  - Step-4 transform parquet: outputs/cross_round_registration/<r2uuid>_registration_*.parquet
  - Step-3 single-cell table:  outputs/tdp43_localization/<r2uuid>_tdp43_localization_*.parquet
                               (R2 nucleus centroids + the join target)
  - R1 metadata parquet:       outputs/run_*/image_metadata/<r1uuid>_metadata*.parquet
                               (BFP planes = Channel_name == "DAPI")
  R1 UUID is read from the transform parquet's Round1_UUID column.

Run:
  python 5_bfp_gating.py                 # all R2 plates with a transform parquet
  python 5_bfp_gating.py -e 5c6bf125     # one R2 UUID prefix
  python 5_bfp_gating.py --selfcheck     # synthetic-data unit check
"""

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import shift as nd_shift
from skimage.filters import threshold_otsu

# --- Reuse Step-3 helpers (single source of truth) -------------------------

def _load_step3():
    """Import 3_tdp43_localization.py by path (module name starts with a digit)."""
    src = Path(__file__).with_name("3_tdp43_localization.py")
    spec = importlib.util.spec_from_file_location("_step3", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_s3 = _load_step3()
pick_best_focus_plane = _s3.pick_best_focus_plane
build_compartment_masks = _s3.build_compartment_masks
_stats = _s3._stats
_um_to_px = _s3._um_to_px
R_NUCLEUS_UM, R_ADJ_UM, R_CYTO_UM = _s3.R_NUCLEUS_UM, _s3.R_ADJ_UM, _s3.R_CYTO_UM
BG_PERCENTILE = _s3.BG_PERCENTILE

# --- Configuration ---------------------------------------------------------

BFP_CHANNEL = "DAPI"         # BFP is labelled "DAPI" in the R1 metadata (Harmony)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO / "outputs" / "bfp_gating"
REG_DIR = REPO / "outputs" / "cross_round_registration"
TDP_DIR = REPO / "outputs" / "tdp43_localization"

R2_UUIDS = [
    "5c6bf125-6e0f-41be-a447-b03ff294c9cc",  # WARD00034
    "eb8c365b-0031-47ee-b7c6-a049d0c6c2cd",  # WARD00026
    "70a85209-e6ea-4d2a-93d5-2d38050dec82",  # WARD00027
]


# --- Core measurement ------------------------------------------------------

def _resolve(fp):
    """Metadata stores filepaths relative to the parent repo; resolve against REPO
    so the script works from any cwd (Step 3/4 relied on cwd == parent)."""
    p = Path(fp)
    return p if p.is_absolute() else REPO / p


def warp_bfp(plane, shift_y, shift_x):
    """Warp an R1 BFP plane into R2 pixel space (translation-only, Step-4 shift)."""
    return nd_shift(plane.astype(np.float32), (shift_y, shift_x),
                    order=1, mode="constant", cval=0.0)


def _valid_bg(warped, shift_y, shift_x):
    """Background (BG_PERCENTILE) over the interior the shift did not pad with 0."""
    sy, sx = int(np.ceil(abs(shift_y))), int(np.ceil(abs(shift_x)))
    ys = slice(sy, warped.shape[0] - sy) if sy else slice(None)
    xs = slice(sx, warped.shape[1] - sx) if sx else slice(None)
    interior = warped[ys, xs]
    if interior.size == 0:
        interior = warped
    return float(np.percentile(interior, BG_PERCENTILE))


def measure_bfp_cell(bfp, cy, cx, others_yx, r_nuc, r_adj, r_cyto, bg):
    """Whole-cell (nucleus | adjacent | cytoplasm) + nucleus-only BFP for one cell.

    Near-clone of Step-3 measure_cell but measures the mask UNION (BFP fills the
    whole soma) instead of three compartments, on the (already warped) BFP plane."""
    H, W = bfp.shape
    yi, xi = int(round(cy)), int(round(cx))
    R = int(np.ceil(r_cyto))
    y0, y1 = max(yi - R, 0), min(yi + R + 1, H)
    x0, x1 = max(xi - R, 0), min(xi + R + 1, W)
    tile = bfp[y0:y1, x0:x1]

    nuc_m, adj_m, cyto_m = build_compartment_masks(
        cy, cx, others_yx, r_nuc, r_adj, r_cyto, y0, y1, x0, x1)
    whole_m = nuc_m | adj_m | cyto_m       # neighbour nuclei already punched out

    cell = _stats(tile[whole_m], bg)
    nucv = _stats(tile[nuc_m], bg)

    row = {
        "bfp_cell_mean": cell["mean"], "bfp_cell_median": cell["median"],
        "bfp_cell_sum": cell["sum"], "bfp_cell_std": cell["std"],
        "bfp_cell_n": cell["n"], "bfp_cell_mean_bg": cell["mean_bg"],
        "bfp_integrated": cell["mean_bg"] * cell["n"] if cell["n"] else np.nan,
        "bfp_nuc_mean": nucv["mean"], "bfp_nuc_sum": nucv["sum"],
        "bfp_bg": bg,
    }
    return row


@delayed
def measure_bfp_site(bfp_plane_paths, shift_yx, reg_ncc, reg_status,
                     nuc_site, r_nuc, r_adj, r_cyto):
    """One field: warp best-focus BFP into R2 space, then measure every R2 nucleus.

    Mirrors Step-3 process_site (same centroids / neighbour pre-filter / tile crop)."""
    planes = [tifffile.imread(_resolve(p)) for p in bfp_plane_paths]
    bf = pick_best_focus_plane(planes)
    shift_y, shift_x = shift_yx
    bfp = warp_bfp(planes[bf], shift_y, shift_x)
    bg = _valid_bg(bfp, shift_y, shift_x)

    ys = nuc_site["centroid-0"].to_numpy(float)
    xs = nuc_site["centroid-1"].to_numpy(float)
    labels = nuc_site["Nucleus_ID"].to_numpy(int)
    centroids = np.column_stack([ys, xs])

    rows = []
    for i, (cy, cx, lab) in enumerate(zip(ys, xs, labels)):
        others = np.delete(centroids, i, axis=0)
        if len(others):
            near = np.abs(others - [cy, cx]).max(axis=1) <= (r_cyto + r_nuc)
            others = others[near]
        m = measure_bfp_cell(bfp, cy, cx, others, r_nuc, r_adj, r_cyto, bg)
        m.update({
            "Row": nuc_site["Row"].iloc[0], "Column": nuc_site["Column"].iloc[0],
            "Frame": nuc_site["Frame"].iloc[0], "Nucleus_ID": int(lab),
            "bfp_best_focus_plane": bf, "reg_ncc": reg_ncc, "reg_status": reg_status,
        })
        rows.append(m)
    return pd.DataFrame(rows)


# --- Per-plate gate --------------------------------------------------------

def gate_bfp(df, method="otsu"):
    """Add bfp_pos / bfp_gmm_prob to a plate's cells via per-plate Otsu on
    log10(whole-cell mean BFP). Raw mean is strictly positive and bimodal
    (untransduced autofluorescence vs transduced); a per-plate cut absorbs the
    (roughly constant within a plate) exposure/vignetting offset.

    ponytail: bfp_gmm_prob left NaN (Otsu path); populate when a GMM is added."""
    df = df.copy()
    df["bfp_gmm_prob"] = np.nan
    feat = df["bfp_cell_mean"].to_numpy(float)
    good = np.isfinite(feat) & (feat > 0)
    if good.sum() < 2:
        df["bfp_pos"] = False
        return df, np.nan
    x = np.log10(feat[good])
    thr = float(threshold_otsu(x))
    pos = np.zeros(len(df), dtype=bool)
    pos[np.where(good)[0]] = x > thr
    df["bfp_pos"] = pos
    return df, 10.0 ** thr


# --- IO / orchestration ----------------------------------------------------

def _newest(paths):
    return sorted(paths)[-1] if paths else None


def _find_reg(uuid, reg_dir):
    return _newest(list(Path(reg_dir).glob(f"{uuid}*registration*.parquet")))


def _find_tdp(uuid, tdp_dir):
    # circle default only — skip the ellipse sensitivity variant
    hits = [p for p in Path(tdp_dir).glob(f"{uuid}*tdp43_localization*.parquet")
            if "ellipse" not in p.name]
    return _newest(hits)


def _find_r1_metadata(r1_uuid, meta_dir=None):
    """Locate the R1 Step-0 metadata parquet (explicit dir, else run_*/image_metadata/)."""
    bases = [meta_dir] if meta_dir else []
    bases += sorted((REPO / "outputs").glob("run_*/image_metadata"))
    for base in bases:
        hit = next(Path(base).glob(f"{r1_uuid}*metadata*.parquet"), None)
        if hit:
            return hit
    raise FileNotFoundError(f"no R1 metadata parquet for {r1_uuid} under outputs/")


def process_plate(r2_uuid, out_dir, today, scheduler, reg_dir, tdp_dir,
                  meta_dir=None, num_workers=None):
    reg_path = _find_reg(r2_uuid, reg_dir)
    tdp_path = _find_tdp(r2_uuid, tdp_dir)
    if reg_path is None or tdp_path is None:
        print(f"[warning] missing reg={reg_path} or tdp={tdp_path} for {r2_uuid[:8]}, skipping.")
        return None

    reg = pd.read_parquet(reg_path)
    r1_uuid = str(reg["Round1_UUID"].iloc[0])
    meta = pd.read_parquet(_find_r1_metadata(r1_uuid, meta_dir))
    bfp = meta[meta["Channel_name"] == BFP_CHANNEL]
    if bfp.empty:
        avail = sorted(meta["Channel_name"].dropna().unique())
        print(f"[error] BFP channel {BFP_CHANNEL!r} absent in R1 {r1_uuid[:8]}; have {avail}")
        return None

    tdp = pd.read_parquet(tdp_path)
    nuc_by_site = dict(tuple(tdp.groupby(["Row", "Column", "Frame"])))
    bfp_by_site = {site: list(g.sort_values("Plane")["filepath"])
                   for site, g in bfp.groupby(["Row", "Column", "Frame"])}
    reg_by_site = {(int(r.Row), int(r.Column), int(r.Frame)): r
                   for r in reg.itertuples(index=False)}

    r_nuc, r_adj, r_cyto = _um_to_px(R_NUCLEUS_UM), _um_to_px(R_ADJ_UM), _um_to_px(R_CYTO_UM)

    tasks, dropped = [], 0
    for site, nuc_site in nuc_by_site.items():
        r = reg_by_site.get(site)
        if r is None or site not in bfp_by_site:
            dropped += len(nuc_site)                    # no transform or no BFP plane
            continue
        tasks.append(measure_bfp_site(
            bfp_by_site[site], (float(r.shift_y), float(r.shift_x)),
            float(r.ncc), str(r.reg_status), nuc_site, r_nuc, r_adj, r_cyto))

    print(f"{r2_uuid[:8]} (R2) <- {r1_uuid[:8]} (R1/BFP): {len(tasks)} fields "
          f"(dropped {dropped} cells with no transform/BFP)", flush=True)
    if not tasks:
        print("  [warning] no measurable fields, skipping.")
        return None

    kw = {"num_workers": num_workers} if num_workers else {}
    with ProgressBar():
        dfs = dask.compute(*tasks, scheduler=scheduler, **kw)
    feats = pd.concat(dfs, ignore_index=True)

    feats, thr = gate_bfp(feats)
    feats.insert(0, "Round2_UUID", r2_uuid)
    feats.insert(0, "Round1_UUID", r1_uuid)

    out_path = Path(out_dir) / f"{r2_uuid}_bfp_{today}.parquet"
    feats.to_parquet(out_path, index=False)
    frac = feats["bfp_pos"].mean()
    print(f"Wrote {out_path}  ({len(feats)} cells; BFP+ {100 * frac:.0f}% "
          f"@ mean-intensity thr={thr:.1f}; flatfield UNAPPLIED)")
    if frac < 0.02 or frac > 0.98:
        print("  [warning] BFP+ fraction near 0/1 — gate may be unimodal; check the histogram.")
    return feats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--experiments", nargs="+", default=None, metavar="R2UUID",
                   help="R2 UUID prefixes to process (default: all three plates)")
    p.add_argument("-m", "--meta-dir", type=Path, default=None,
                   help="explicit R1 Step-0 metadata dir (default: search outputs/run_*/)")
    p.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--reg-dir", type=Path, default=REG_DIR)
    p.add_argument("--tdp-dir", type=Path, default=TDP_DIR)
    p.add_argument("-s", "--scheduler", default="processes",
                   choices=["processes", "threads", "single-threaded", "synchronous"])
    p.add_argument("-w", "--workers", type=int, default=4,
                   help="max parallel workers (default 4; bounds warp/mask memory)")
    p.add_argument("--selfcheck", action="store_true", help="synthetic unit check and exit")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selfcheck:
        return _selfcheck()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    uuids = R2_UUIDS
    if args.experiments:
        uuids = [u for u in R2_UUIDS if any(u.startswith(e) for e in args.experiments)]
    print(f"BFP gating for {len(uuids)} plate(s) with {args.workers} workers.", flush=True)
    for r2 in uuids:
        process_plate(r2, args.output_dir, today, args.scheduler,
                      args.reg_dir, args.tdp_dir, args.meta_dir, num_workers=args.workers)


# --- Self-check ------------------------------------------------------------

def _selfcheck():
    """(a) Warp: an R1 BFP soma offset by (dy,dx) lands back on its R2 centroid
    after warping by (-dy,-dx), so the whole-cell mean reads high — and a
    background cell reads low. (b) Gate: on a bimodal set of whole-cell means the
    per-plate Otsu cut sends the bright soma BFP+ and the background cell BFP-.
    BFP is a whole-soma signal, so the synthetic fills the soma to r_cyto (not a
    nucleus-only blob, which the large disk-union would dilute below background)."""
    r_nuc, r_adj, r_cyto = _um_to_px(R_NUCLEUS_UM), _um_to_px(R_ADJ_UM), _um_to_px(R_CYTO_UM)
    H = W = 400
    yy, xx = np.ogrid[:H, :W]
    bright_c, dark_c = (120, 120), (280, 280)   # two well-separated R2 centroids

    # R1 BFP plane offset by (dy,dx); warping by (-dy,-dx) restores R2 space.
    dy, dx = 6, -4
    r1 = np.full((H, W), 100.0, np.float32)      # background autofluorescence
    r1[(yy - (bright_c[0] + dy)) ** 2 + (xx - (bright_c[1] + dx)) ** 2 <= r_cyto ** 2] = 4000.0
    warped = warp_bfp(r1, -dy, -dx)
    bg = _valid_bg(warped, -dy, -dx)

    bright = measure_bfp_cell(warped, *bright_c, np.array([dark_c], float),
                              r_nuc, r_adj, r_cyto, bg)
    dark = measure_bfp_cell(warped, *dark_c, np.array([bright_c], float),
                            r_nuc, r_adj, r_cyto, bg)
    assert bright["bfp_cell_mean"] > 5 * dark["bfp_cell_mean"], (bright, dark)

    # Gate: a bimodal plate — 4 background cells + the warped bright soma + 3 more bright.
    means = [dark["bfp_cell_mean"], 110.0, 130.0, 150.0,
             bright["bfp_cell_mean"], 3600.0, 3800.0, 3400.0]
    df = pd.DataFrame({"bfp_cell_mean": means})
    df["Row"], df["Column"], df["Frame"] = 1, 1, 1
    df["Nucleus_ID"] = range(len(df))
    gated, thr = gate_bfp(df)
    assert bool(gated.loc[0, "bfp_pos"]) is False, "background cell should gate BFP-"
    assert bool(gated.loc[4, "bfp_pos"]) is True, "bright soma should gate BFP+"

    print(f"selfcheck OK — warp restored bright soma (whole-cell mean "
          f"{bright['bfp_cell_mean']:.0f} vs dark {dark['bfp_cell_mean']:.0f}); "
          f"Otsu thr={thr:.0f}, BFP+ {int(gated['bfp_pos'].sum())}/{len(gated)}")


if __name__ == "__main__":
    main()
