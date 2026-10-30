import re
import sys
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import LinearSegmentedColormap

# --- run_utils bootstrap ---------------------------------------------------
# Make `import run_utils` work regardless of where the script is launched from
# (repo root, scripts/, or a SLURM job with an arbitrary cwd).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_utils as ru  # noqa: E402

# --- Configuration ---------------------------------------------------------

# Defaults; can be overridden on the command line (see parse_args).
DEFAULT_METADATA_DIR = Path("/Users/pmihack/claire/hs-array/resources/plate_maps")
DEFAULT_BASE_PATH = Path("/Users/pmihack/claire/hs-array/data/D28_AB_Rep1")
DEFAULT_OUTPUT_DIR = Path("/Users/pmihack/claire/hs-array/outputs/image_metadata")

# HS array D28 AB Rep1 UUIDs
# Round 2 only (DAPI-based, authoritative for analysis):
#   WARD00034-R2: 5c6bf125-6e0f-41be-a447-b03ff294c9cc
#   WARD00026-R2: eb8c365b-0031-47ee-b7c6-a049d0c6c2cd
#   WARD00027-R2: 70a85209-e6ea-4d2a-93d5-2d38050dec82
# Round 1 only (BFP/HDGFL2-CE/TDP-43/TGN-46):
#   WARD00034-R1: ec4cf983-56d0-4933-9b99-d9bddb374047
#   WARD00026-R1: 96e67a25-7c3b-4d8b-acc7-bd27e93c30a6
#   WARD00027-R1: d9412df5-9d66-4e1c-ac4f-45c784370eaa
# Pass --experiments <uuid> [<uuid> ...] to select a subset.

# Harmony XML namespace (consistent across experiments)
NS = {"h": "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"}

# Pseudocolor mapping rules (used to generate images on the fly)
PSEUDOCOLOR_MAP = {
    "DAPI": "blue",
    "Brightfield": "gray",
    "Alexa 488": "green",
    "CF568": "red",
    "Alexa 647": "magenta",
}

MPL_COLORMAPS = {
    "blue": LinearSegmentedColormap.from_list("black_blue", [(0, 0, 0), (0, 0, 1)]),
    "green": LinearSegmentedColormap.from_list("black_green", [(0, 0, 0), (0, 1, 0)]),
    "red": LinearSegmentedColormap.from_list("black_red", [(0, 0, 0), (1, 0, 0)]),
    "magenta": LinearSegmentedColormap.from_list("black_magenta", [(0, 0, 0), (1, 0, 1)]),
    "gray": LinearSegmentedColormap.from_list("black_gray", [(0, 0, 0), (1, 1, 1)]),
}

FILENAME_RE = re.compile(r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)t(\d+)")

# Saturation QC: a channel is flagged when the MEDIAN image has more than this
# fraction of pixels at the dtype ceiling (65535). 0.1% of a 2160² field is
# ~4,700 px — about one clipped nucleus core per field; a well-exposed channel
# sits at 0. Found 2026-10-29 on the 40x optimizationplate: DAPI p99.9 = 65535
# on all 81 sites × 5 planes, which flattens nuclei and breaks Step 1's Otsu.
SAT_WARN_FRAC = 1e-3
N_WORKERS = max(1, cpu_count() - 2)

# Stage name used in run metadata / logs.
STAGE = "image_metadata"


def collect_params():
    """Tuning constants recorded in run_metadata.json for this stage."""
    return {
        "pseudocolor_map": PSEUDOCOLOR_MAP,
        "harmony_namespace": NS["h"],
        "filename_regex": FILENAME_RE.pattern,
        "sat_warn_frac": SAT_WARN_FRAC,
    }


# --- Helpers ---------------------------------------------------------------

def find_text(element, path, cast=None):
    """Return the text of the first matching element, optionally cast."""
    node = element.find(path, NS)
    if node is None or node.text is None:
        return None
    return cast(node.text) if cast else node.text


def parse_filename(name):
    """Extract (Row, Column, Frame, Plane, ChannelID, Time) from a filename."""
    match = FILENAME_RE.match(name)
    if match:
        return [int(g) for g in match.groups()]
    return [None] * 6


def saturation_stats(filepath):
    """(img_max, sat_frac) for one TIFF; sat_frac = fraction of pixels at the dtype ceiling."""
    img = tifffile.imread(str(filepath))
    ceiling = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else img.max()
    return int(img.max()), float((img >= ceiling).mean())


def add_saturation(df, n_workers=N_WORKERS):
    """Add img_max / sat_frac columns (one TIFF read per row, in a process pool)."""
    paths = df["filepath"].tolist()
    if n_workers == 1:                     # also the test path: spec-loaded module can't pickle to spawn workers
        stats = list(map(saturation_stats, paths))
    else:
        with Pool(n_workers) as pool:
            stats = pool.map(saturation_stats, paths, chunksize=16)
    df = df.copy()
    df["img_max"], df["sat_frac"] = zip(*stats)
    return df


def saturation_summary(df):
    """Per-channel saturation table + list of flagged channels (median image clipped)."""
    g = df.groupby("Channel_name")["sat_frac"]
    table = pd.DataFrame({
        "median_sat_frac": g.median(),
        "max_sat_frac": g.max(),
        "frac_images_clipped": g.apply(lambda s: (s > 0).mean()),
    })
    table["flag"] = table["median_sat_frac"] > SAT_WARN_FRAC
    return table, table.index[table["flag"]].tolist()


def parse_channels(index_root):
    """Extract channel metadata from the index XML."""
    for map_el in index_root.findall(".//h:Map", NS):
        first_entry = map_el.find("h:Entry", NS)
        if first_entry is None or first_entry.find("h:ChannelName", NS) is None:
            continue

        channels = []
        for entry in map_el.findall("h:Entry", NS):
            ch_id = entry.attrib.get("ChannelID")
            channels.append({
                "ChannelID": int(ch_id) if ch_id is not None else None,
                "Channel_name": find_text(entry, "h:ChannelName"),
                "Type": find_text(entry, "h:ChannelType"),
                "Excitation_nm": find_text(entry, "h:MainExcitationWavelength"),
                "Emission_nm": find_text(entry, "h:MainEmissionWavelength"),
            })
        return channels
    return []


def discover_experiments(base_path, selected=None):
    """Return a list of experiment directories under base_path.

    If `selected` is given, only those folder names are returned (preserving
    the given order); a warning is printed for any that are missing.
    """
    if selected is not None:
        experiments = []
        for name in selected:
            path = base_path / name
            if path.is_dir():
                experiments.append(path)
            else:
                print(f"[warning] experiment folder not found, skipping: {path}")
        return experiments

    # Default: every subdirectory that contains an index folder with an XML.
    experiments = []
    for path in sorted(p for p in base_path.iterdir() if p.is_dir()):
        if next((path / "index").glob("*.xml"), None) is not None:
            experiments.append(path)
    return experiments


def process_experiment(experiment_path):
    """Parse one experiment folder and return its merged metadata DataFrame.

    Returns None if required files are missing.
    """
    experiment_img_dir = experiment_path / "images"
    experiment_xml_file = next(experiment_path.glob("*.xml"), None)
    index_xml = next((experiment_path / "index").glob("*.xml"), None)

    if experiment_xml_file is None or index_xml is None:
        print(f"[warning] missing XML files in {experiment_path.name}, skipping.")
        return None

    # Experiment-level XML
    experiment_root = ET.parse(experiment_xml_file).getroot()
    measurement_id = find_text(experiment_root, "h:MeasurementID")
    date = find_text(experiment_root, "h:Date")
    plate = find_text(experiment_root, "h:InitialPlateName")

    # Index XML
    index_root = ET.parse(index_xml).getroot()
    plate_id = find_text(index_root, ".//h:PlateID")
    x_res = find_text(index_root, ".//h:ImageResolutionX", float) * 1e6
    y_res = find_text(index_root, ".//h:ImageResolutionY", float) * 1e6

    # Channel metadata
    channel_df = (
        pd.DataFrame(parse_channels(index_root))
        .sort_values("ChannelID")
        .reset_index(drop=True)
    )
    channel_df["Pseudocolor"] = channel_df["Channel_name"].map(PSEUDOCOLOR_MAP).fillna("gray")
    channel_df["MPL_colormap"] = channel_df["Pseudocolor"].str.lower().map(MPL_COLORMAPS)
    channel_df["Measurement_ID"] = measurement_id
    channel_df["Measurement_date"] = date
    channel_df["Plate_ID"] = plate_id
    channel_df["res_x"] = x_res
    channel_df["res_y"] = y_res

    # Image files
    files = sorted(f for f in experiment_img_dir.rglob("*") if f.suffix.lower() == ".tiff")
    if not files:
        print(f"[warning] no .tiff files in {experiment_path.name}, skipping.")
        return None

    df = pd.DataFrame({
        "filepath": files,
        "filename": [f.name for f in files],
        "subdirectory": [f.parent.relative_to(experiment_img_dir) for f in files],
    })
    df[["Row", "Column", "Frame", "Plane", "ChannelID", "Time"]] = df["filename"].apply(
        lambda x: pd.Series(parse_filename(x))
    )

    merged_df = pd.merge(df, channel_df, on="ChannelID")
    merged_df["Experiment_name"] = experiment_path.name
    merged_df = add_saturation(merged_df)
    sat_table, flagged = saturation_summary(merged_df)
    print("Saturation per channel (fraction of pixels at 65535):")
    print(sat_table.to_string(float_format=lambda x: f"{x:.4f}"))
    for ch in flagged:
        print(f"[WARNING] {experiment_path.name}: channel {ch!r} is overexposed — "
              f"median image has {sat_table.loc[ch, 'median_sat_frac']:.2%} clipped pixels. "
              f"Lower exposure/power on re-acquisition; segmentation on a clipped channel is unreliable.")

    # Per-experiment summary
    summary = {
        "wells": merged_df[["Row", "Column"]].drop_duplicates().shape[0],
        "channels": merged_df["ChannelID"].nunique(),
        "z_planes": merged_df["Plane"].nunique(),
        "frames": merged_df["Frame"].nunique(),
        "timepoints": merged_df["Time"].nunique(),
    }

    print(f"""
Experiment ID: {measurement_id}
Experiment folder: {experiment_path.name}
Plate ID: {plate}
Wells imaged: {summary["wells"]}
Frames per well: {summary["frames"]}
Channels per image: {summary["channels"]}
Z-slices per image: {summary["z_planes"]}
Timepoints per image: {summary["timepoints"]}
""")

    return merged_df


# --- Main ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract imaging metadata for one or more experiments."
    )
    parser.add_argument(
        "base_path",
        nargs="?",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Directory containing experiment folders "
             f"(default: {DEFAULT_BASE_PATH}).",
    )
    parser.add_argument(
        "-e", "--experiments",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Specific experiment folder names to process. "
             "If omitted, all experiments under base_path are processed.",
    )
    parser.add_argument(
        "-m", "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help=f"Directory containing the plate metadata CSVs "
             f"(default: {DEFAULT_METADATA_DIR}).",
    )
    # --output-root + --run-id (optional here: stage 0_ mints the run ID).
    ru.add_run_args(parser, mints_run_id=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.base_path.is_dir():
        raise SystemExit(f"[error] base path is not a directory: {args.base_path}")

    # Mint (or accept) the run ID and create the run directory.
    run_id = args.run_id or ru.new_run_id()
    run_dir = ru.create_run_dir(args.output_root, run_id)
    out_dir = ru.stage_dir(run_dir, STAGE)

    rec = ru.StageRecorder(
        run_dir, stage=STAGE, run_id=run_id,
        params=collect_params(),
        inputs={
            "base_path": str(args.base_path),
            "metadata_dir": str(args.metadata_dir),
            "experiments_requested": args.experiments or "ALL",
        },
    )
    # Make the run ID obvious in stdout — downstream stages need it.
    print(f"\n=== RUN ID: {run_id} ===")
    print(f"=== run dir: {run_dir} ===\n")
    rec.log(f"run id {run_id}")
    experiments = discover_experiments(args.base_path, args.experiments)
    rec.log(f"found {len(experiments)} experiment(s) to process")
    print(f"Found {len(experiments)} experiment(s) to process.\n")

    all_metadata = {}
    per_experiment = []
    for experiment_path in experiments:
        merged_df = process_experiment(experiment_path)
        if merged_df is None:
            per_experiment.append({
                "experiment": experiment_path.name,
                "status": "skipped",
                "n_images": 0,
            })
            continue
        all_metadata[experiment_path.name] = merged_df

        out_path = out_dir / f"{experiment_path.name}_metadata.parquet"
        # Drop the colormap objects (not serializable) and cast Path
        # columns to str so pyarrow can write them.
        out_df = merged_df.drop(columns=["MPL_colormap"], errors="ignore").copy()
        for col in ("filepath", "subdirectory"):
            if col in out_df.columns:
                out_df[col] = out_df[col].astype(str)
        out_df.to_parquet(out_path, index=False)
        rec.log(f"{experiment_path.name}: wrote {len(out_df)} rows -> {out_path.name}")
        print(f"Wrote {out_path}")

        sat_table, flagged = saturation_summary(merged_df)
        for ch in flagged:
            rec.log(f"{experiment_path.name}: WARNING channel {ch!r} overexposed "
                    f"(median sat_frac {sat_table.loc[ch, 'median_sat_frac']:.4f})")
        per_experiment.append({
            "experiment": experiment_path.name,
            "status": "ok",
            "n_images": int(len(out_df)),
            "n_dapi": int((merged_df["Channel_name"] == "DAPI").sum()),
            "saturation": sat_table.round(5).to_dict(orient="index"),
            "overexposed_channels": flagged,
        })

    # Combine everything into one big table (with an Experiment_name column).
    if all_metadata:
        combined_df = pd.concat(all_metadata.values(), ignore_index=True)
    else:
        combined_df = pd.DataFrame()
        print("No experiments were successfully processed.")

    # DAPI samples across all processed experiments.
    n_dapi_total = 0
    if not combined_df.empty:
        dapi_samples = combined_df[combined_df["Channel_name"] == "DAPI"].copy()
        filepaths = dapi_samples["filepath"].tolist()
        n_dapi_total = len(filepaths)

        print(f"\nFound {n_dapi_total} DAPI images across all experiments.")
        print("Sample filepaths:")
        for filepath in filepaths[:5]:
            print(f" - {filepath}")

    # One overall stage record, with per-experiment summaries nested inside.
    n_ok = sum(1 for e in per_experiment if e["status"] == "ok")
    rec.finish(
        outputs={"image_metadata_dir": str(out_dir)},
        summary={
            "experiments_processed": n_ok,
            "experiments_skipped": len(per_experiment) - n_ok,
            "total_dapi_images": int(n_dapi_total),
            "per_experiment": per_experiment,
        },
    )

    print(f"\n=== RUN ID: {run_id} (pass to downstream stages with --run-id) ===")
    return combined_df


if __name__ == "__main__":
    combined_df = main()