import os
import sys
import argparse
from pathlib import Path

import dask
from dask.diagnostics import ProgressBar
from dask import delayed
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.stats import norm
from skimage import filters, morphology
from skimage.measure import regionprops_table

# --- run_utils bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_utils as ru  # noqa: E402

# --- Configuration ---------------------------------------------------------

# Segmentation parameters
INTENSITY_SCALING_PARAM = [1, 7]
BLUR_SIGMA = 1
# Debris floor. Defined in µm² as exactly 400 px² at D28 Rep1's 0.2967 µm/px
# (≈ 35.2 µm²) so D28 runs keep their 400 px floor; scaled per experiment from
# Step 0's res_x (40x 0.1483 µm/px → 1,600 px²). MIN_AREA is the fallback when
# the metadata carries no pixel size.
MIN_AREA_UM2 = 400 * 0.2967 ** 2
MIN_AREA = 400

# Which channel to segment on.
CHANNEL_NAME = "DAPI"

# Default input/output dirs (chain from the metadata script).
DEFAULT_INPUT_DIR = Path("/Users/pmihack/claire/hs-array/outputs/image_metadata")
DEFAULT_OUTPUT_DIR = Path("/Users/pmihack/claire/hs-array/outputs/nuclei_features")

# Full-frame contrast gate (informational; not applied in MIP mode).
CONTRAST_CUTOFF = 100

# Stage name + the sub-dirs this stage reads from / writes to inside a run.
STAGE = "nucleus_segmentation"
INPUT_STAGE_DIR = "image_metadata"
OUTPUT_STAGE_DIR = "nuclei_features"

REGIONPROPS = (
    "label", "area", "intensity_mean", "intensity_max", "intensity_min",
    "intensity_std", "centroid", "eccentricity", "solidity", "perimeter",
    "feret_diameter_max", "orientation", "axis_major_length", "axis_minor_length",
)


def collect_params():
    """Tuning constants recorded in run_metadata.json for this stage."""
    return {
        "intensity_scaling_param": INTENSITY_SCALING_PARAM,
        "blur_sigma": BLUR_SIGMA,
        "min_area_um2": MIN_AREA_UM2,          # resolved px² per experiment in summary.per_experiment
        "min_area_fallback_px": MIN_AREA,
        "channel_name": CHANNEL_NAME,
        "contrast_cutoff": CONTRAST_CUTOFF,
        "regionprops": list(REGIONPROPS),
    }


# --- Segmentation ----------------------------------------------------------

def min_area_px(pixel_size_um):
    """Debris floor in px² for a pixel size; MIN_AREA when the size is unknown."""
    if pixel_size_um is None:
        return MIN_AREA
    return max(1, round(MIN_AREA_UM2 / pixel_size_um ** 2))


@delayed
def process_nucleus_site(plane_paths, row, column, frame, ic_field=None, darkfield=None,
                         min_area=None):
    """MIP across z-planes for one (well, frame) site, then segment nuclei.

    Returns (features_df, status_dict).
    ic_field/darkfield: passed to ru.load_field for optional flatfield correction.
    """
    floor = MIN_AREA if min_area is None else min_area   # explicit arg reaches dask workers
    planes = [ru.load_field(p, ic_field, darkfield) for p in plane_paths]
    # ponytail: MIP used for segmentation only; intensity features here come
    # from the projection. Use best-focus plane for organelle quantification.
    nuc = np.max(np.stack(planes), axis=0)

    # Normalize via a fitted-Gaussian contrast stretch
    m, s = norm.fit(nuc.flatten())
    stretch_min = max(m - INTENSITY_SCALING_PARAM[0] * s, nuc.min())
    stretch_max = min(m + INTENSITY_SCALING_PARAM[1] * s, nuc.max())
    nuc_stretch = np.clip(nuc, stretch_min, stretch_max)
    image_norm = (nuc_stretch - stretch_min) / (stretch_max - stretch_min)

    blurred = filters.gaussian(image_norm, sigma=BLUR_SIGMA)

    # Step 1: low-level threshold
    triangle_cutoff = filters.threshold_triangle(blurred)
    global_median_cutoff = np.percentile(blurred, 50)
    th_low_cutoff = (triangle_cutoff + global_median_cutoff) / 2
    img_low_level = blurred > th_low_cutoff
    # ponytail: min_size deprecated in skimage 0.26; max_size has inverted
    # semantics (removes <=, not <), so MIN_AREA-1 preserves old min_size=MIN_AREA
    img_low_level = morphology.remove_small_objects(
        img_low_level.astype(bool), max_size=floor - 1
    )
    img_low_level = morphology.dilation(img_low_level, footprint=morphology.disk(2))

    # Step 2: per-object high-level threshold
    otsu_cutoff = 0.333 * filters.threshold_otsu(blurred)
    img_high_level = np.zeros_like(img_low_level)
    lab_low, num_obj = morphology.label(img_low_level, return_num=True)
    for idx in range(num_obj):
        single_obj = lab_low == (idx + 1)
        local_otsu = filters.threshold_otsu(blurred[single_obj])
        if local_otsu > otsu_cutoff:
            img_high_level[np.logical_and(blurred > 0.98 * local_otsu, single_obj)] = 1

    filled = ndi.binary_fill_holes(img_high_level)
    filled = morphology.dilation(filled, footprint=morphology.disk(2))
    filled_clean = morphology.remove_small_objects(filled.astype(bool), max_size=floor - 1)
    nuc_seg = morphology.label(filled_clean)

    # Extract features
    props = regionprops_table(
        label_image=nuc_seg,
        intensity_image=nuc,
        properties=REGIONPROPS,
    )
    df = pd.DataFrame(props)
    df["Row"] = row
    df["Column"] = column
    df["Frame"] = frame
    df["mip_n_planes"] = len(plane_paths)

    status = {
        "Row": row,
        "Column": column,
        "Frame": frame,
        "n_planes": len(plane_paths),
        "n_nuclei": len(df),
        "filter_status": "pass" if len(df) > 0 else "no_nuclei",
        "contrast_check": "pass" if len(df) > 0 else "fail",
    }
    return df, status


# --- helpers ------------------------------------------------------------

def experiment_name_from_parquet(parquet_path, df):
    """Best-effort experiment name: prefer the column, fall back to filename."""
    if "Experiment_name" in df.columns and df["Experiment_name"].notna().any():
        return str(df["Experiment_name"].iloc[0])
    stem = parquet_path.stem
    return stem.split("_metadata")[0] if "_metadata" in stem else stem

def discover_parquets(input_dir, selected=None):
    """Return metadata parquet paths under input_dir.

    If `selected` is given, only parquets whose filename starts with one of
    those experiment names are returned.
    """
    all_parquets = sorted(input_dir.glob("*.parquet"))
    if selected is None:
        return all_parquets

    chosen = []
    for name in selected:
        matches = [p for p in all_parquets if p.name.startswith(name)]
        if matches:
            chosen.extend(matches)
        else:
            print(f"[warning] no parquet found for experiment: {name}")
    return chosen


def process_metadata_parquet(parquet_path, output_dir, scheduler, rec,
                             ic_field=None, darkfield=None, npz=None, npz_key=None,
                             pixel_size=None):
    """Run segmentation for every DAPI image in one metadata parquet.

    Pixel size and the flatfield key are resolved per experiment from Step 0's
    metadata (res_x, ChannelID); `pixel_size` / `npz_key` override them.
    """
    meta = pd.read_parquet(parquet_path)

    if "Channel_name" not in meta.columns:
        print(f"[warning] {parquet_path.name} has no Channel_name column, skipping.")
        return None

    samples = meta[meta["Channel_name"] == CHANNEL_NAME].copy()
    if samples.empty:
        print(f"[warning] no {CHANNEL_NAME} images in {parquet_path.name}, skipping.")
        return None

    exp_name = experiment_name_from_parquet(parquet_path, meta)

    px = ru.pixel_size_um(meta, pixel_size)
    min_area = min_area_px(px)
    if px is None:
        print(f"[warning] {exp_name}: no res_x in metadata; MIN_AREA falls back to {MIN_AREA} px²")
    ic_key = None
    if npz is not None:
        ic_key = ru.ic_npz_key(meta, CHANNEL_NAME, npz_key)
        ic_field, darkfield = ru.load_ic_field(npz, ic_key)
    print(f"{exp_name}: pixel size {px} µm/px → MIN_AREA {min_area} px²"
          + (f"; flatfield key {ic_key}" if ic_key else ""))

    # Group z-planes by site (well + frame); one MIP task per site.
    grouped = samples.sort_values("Plane").groupby(["Row", "Column", "Frame"])
    print(f"\n{exp_name}: segmenting {len(grouped)} sites (MIP over z-planes)...")

    tasks = [
        process_nucleus_site(list(grp["filepath"]), row, col, frame, ic_field, darkfield,
                             min_area=min_area)
        for (row, col, frame), grp in grouped
    ]
    with ProgressBar():
        results = dask.compute(*tasks, scheduler=scheduler)

    dfs = [r[0] for r in results]
    statuses = [r[1] for r in results]

    features_df = pd.concat(dfs, ignore_index=True)

    # One metadata row per site (drop plane-specific columns before dedup).
    plane_cols = [c for c in ("Plane", "filepath", "filename", "subdirectory") if c in samples.columns]
    site_meta = (
        samples.drop(columns=plane_cols)
        .drop_duplicates(["Row", "Column", "Frame"])
        .reset_index(drop=True)
    )
    features_df = features_df.merge(site_meta, on=["Row", "Column", "Frame"], how="left")

    qc_df = pd.DataFrame(statuses)
    # ponytail: MIP mode — QC keyed by site (Row/Column/Frame), not filepath
    site_cols = [c for c in samples.columns if c not in ("Plane", "filepath", "filename", "subdirectory")]
    qc_df = qc_df.merge(
        samples[site_cols].drop_duplicates(["Row", "Column", "Frame"]),
        on=["Row", "Column", "Frame"],
        how="left",
    )

    out_path = output_dir / f"{exp_name}_nuclei_features.parquet"
    features_df.to_parquet(out_path, index=False)

    qc_path = output_dir / f"{exp_name}_image_qc.parquet"
    qc_df.to_parquet(qc_path, index=False)

    # --- Per-experiment summary ---
    total_images = len(qc_df)
    passed_contrast = int((qc_df["contrast_check"] == "pass").sum())
    total_nuclei = len(features_df)

    print(f"\n--- {exp_name} summary ---")
    print(f"  Total images:              {total_images}")
    print(f"  Passed contrast check:     {passed_contrast}")
    print(f"  Total nuclei detected:     {total_nuclei}")
    print(f"  Wrote features -> {out_path}")
    print(f"  Wrote QC       -> {qc_path}")
    rec.log(f"{exp_name}: {total_nuclei} nuclei from {passed_contrast}/"
            f"{total_images} frames -> {out_path.name}")

    return {
        "exp_name": exp_name,
        "total_images": total_images,
        "passed_contrast": passed_contrast,
        "total_nuclei": total_nuclei,
        "pixel_size_um": px,
        "min_area_px": int(min_area),
        "ic_npz_key": ic_key,
    }


# --- Main ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment nuclei from metadata parquet files and write "
                    "per-experiment nuclei-feature parquets."
    )
    parser.add_argument(
        "-e", "--experiments",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Specific experiment names to process (matched against parquet "
             "filename prefixes). If omitted, all parquets are processed.",
    )
    parser.add_argument(
        "-s", "--scheduler",
        default="processes",
        choices=["processes", "threads", "single-threaded", "synchronous"],
        help="Dask scheduler to use (default: processes).",
    )
    parser.add_argument(
        "--ic-npz", type=Path, default=None, metavar="PATH",
        help="NPZ file with flatfield fields. If omitted, no correction applied.",
    )
    parser.add_argument(
        "--ic-npz-key", default=None, metavar="KEY",
        help="Override the flatfield key (default: ch<ChannelID> of DAPI from the "
             "Step 0 metadata, e.g. ch1 on D28 Rep1, ch4 on WARD00048).",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=None, metavar="UM",
        help="Override the pixel size used to scale MIN_AREA (default: res_x from the "
             "Step 0 metadata; 0.2967 → 400 px², 0.1483 → 1,600 px²).",
    )
    # --output-root + --run-id (required here: reuse the run minted by 0_).
    ru.add_run_args(parser, mints_run_id=False)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve the existing run dir (errors clearly if the run ID is wrong).
    run_dir = ru.resolve_run_dir(args.output_root, args.run_id)
    input_dir = ru.stage_dir(run_dir, INPUT_STAGE_DIR)
    output_dir = ru.stage_dir(run_dir, OUTPUT_STAGE_DIR)

    rec = ru.StageRecorder(
        run_dir, stage=STAGE, run_id=args.run_id,
        params=collect_params(),
        inputs={
            "input_dir": str(input_dir),
            "scheduler": args.scheduler,
            "experiments_requested": args.experiments or "ALL",
        },
    )
    print(f"\n=== RUN ID: {args.run_id} ===")
    print(f"=== run dir: {run_dir} ===\n")

    # Optional flatfield correction
    npz = None
    if args.ic_npz:
        import numpy as np
        npz = np.load(args.ic_npz)
        print(f"Flatfield: {args.ic_npz} (key resolved per experiment)")

    parquets = discover_parquets(input_dir, args.experiments)
    print(f"Found {len(parquets)} metadata parquet(s) to process.")
    rec.log(f"found {len(parquets)} metadata parquet(s) in {input_dir}")

    all_stats = []
    for parquet_path in parquets:
        stats = process_metadata_parquet(parquet_path, output_dir, args.scheduler, rec,
                                         npz=npz, npz_key=args.ic_npz_key,
                                         pixel_size=args.pixel_size)
        if stats is not None:
            all_stats.append(stats)

    # --- Grand total across all experiments ---
    total_images = sum(s["total_images"] for s in all_stats)
    passed_contrast = sum(s["passed_contrast"] for s in all_stats)
    total_nuclei = sum(s["total_nuclei"] for s in all_stats)

    if all_stats:
        print("\n" + "=" * 40)
        print("OVERALL SUMMARY")
        print(f"  Experiments processed:     {len(all_stats)}")
        print(f"  Total images:              {total_images}")
        print(f"  Passed contrast check:     {passed_contrast}")
        print(f"  Total nuclei detected:     {total_nuclei}")
        print("=" * 40)

    rec.finish(
        outputs={"nuclei_features_dir": str(output_dir)},
        summary={
            "experiments_processed": len(all_stats),
            "total_images": total_images,
            "passed_contrast": passed_contrast,
            "total_nuclei": total_nuclei,
            "per_experiment": all_stats,
        },
    )

    print(f"\n=== RUN ID: {args.run_id} (pass to downstream stages) ===")


if __name__ == "__main__":
    main()