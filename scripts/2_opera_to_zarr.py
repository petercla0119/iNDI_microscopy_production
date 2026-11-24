"""Convert an Opera Phenix TIFF export to an OME-Zarr v3 HCS plate store.

Reads the standard Opera Phenix directory layout (experiment XML, index XML,
image.index.txt, flatfield XML, TIFFs) and writes a single OME-Zarr v3
plate store that preserves all acquisition metadata.

Usage:
    python 2_opera_to_zarr.py <experiment_dir> [--output-dir <dir>]

The experiment directory must contain:
  - An experiment XML at top level
  - index/<name>.xml
  - images/image.index.txt
  - images/**/*.tiff

Output is written to <output_dir>/<plate_name>.zarr (default: parent of
experiment_dir).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import zarr
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from tifffile import imread as tiff_imread

N_WORKERS = max(1, os.cpu_count() - 2)
ZARR_FORMAT = 3
NS = {"h": "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"}
CHANNEL_COLORS = ["0000FF", "00FF00", "FF0000", "FF00FF"]


# ---------------------------------------------------------------------------
# Metadata parsers
# ---------------------------------------------------------------------------


def _find_text(el, tag):
    m = el.find(tag, NS)
    return m.text if m is not None else None


def parse_experiment_xml(experiment_dir: Path) -> dict:
    xml_file = next(
        (f for f in experiment_dir.glob("*.xml") if f.parent == experiment_dir), None
    )
    if xml_file is None:
        raise FileNotFoundError(f"No experiment XML in {experiment_dir}")

    root = ET.parse(xml_file).getroot()
    meta = {
        "measurement_id": _find_text(root, "h:MeasurementID"),
        "date": _find_text(root, "h:Date"),
        "plate_name": _find_text(root, "h:InitialPlateName"),
        "serial": _find_text(root, "h:Serial"),
        "user": _find_text(root, "h:UserName"),
    }

    inst = root.find("h:InstrumentDescription", NS)
    if inst is not None:
        meta["instrument"] = {
            "type": _find_text(inst, "h:Type"),
            "software_version": _find_text(inst, "h:SoftwareVersion"),
            "serial": _find_text(inst, "h:Serial"),
        }
        objectives = []
        for obj in inst.findall(".//h:Objective", NS):
            objectives.append({
                "name": _find_text(obj, "h:Name"),
                "magnification": _find_text(obj, "h:Magnification"),
                "na": _find_text(obj, "h:NumAperture"),
                "working_distance_m": _find_text(obj, "h:WorkingDistance"),
            })
        meta["objectives"] = objectives

    return meta


def parse_index_xml(experiment_dir: Path) -> dict:
    index_dir = experiment_dir / "index"
    xml_file = next(index_dir.glob("*.xml"), None)
    if xml_file is None:
        raise FileNotFoundError(f"No index XML in {index_dir}")

    root = ET.parse(xml_file).getroot()
    channels = []
    for map_el in root.findall(".//h:Map", NS):
        first = map_el.find("h:Entry", NS)
        if first is None or first.find("h:ChannelName", NS) is None:
            continue
        for entry in map_el.findall("h:Entry", NS):
            ch_id = entry.attrib.get("ChannelID")
            ch = {
                "channel_id": int(ch_id) if ch_id else None,
                "name": _find_text(entry, "h:ChannelName"),
                "type": _find_text(entry, "h:ChannelType"),
                "excitation_nm": _find_text(entry, "h:MainExcitationWavelength"),
                "emission_nm": _find_text(entry, "h:MainEmissionWavelength"),
                "image_size_x": _find_text(entry, "h:ImageSizeX"),
                "image_size_y": _find_text(entry, "h:ImageSizeY"),
                "exposure_s": _find_text(entry, "h:ExposureTime"),
                "camera": _find_text(entry, "h:CameraType"),
                "objective_mag": _find_text(entry, "h:ObjectiveMagnification"),
                "objective_na": _find_text(entry, "h:ObjectiveNA"),
            }
            res_x = _find_text(entry, "h:ImageResolutionX")
            if res_x:
                ch["pixel_size_x_um"] = float(res_x) * 1e6
            res_y = _find_text(entry, "h:ImageResolutionY")
            if res_y:
                ch["pixel_size_y_um"] = float(res_y) * 1e6
            channels.append(ch)
        break

    return {
        "plate_id": _find_text(root, ".//h:PlateID"),
        "plate_type": _find_text(root, ".//h:PlateTypeName"),
        "plate_rows": _find_text(root, ".//h:PlateRows"),
        "plate_cols": _find_text(root, ".//h:PlateColumns"),
        "channels": channels,
    }


def parse_kw_txt(experiment_dir: Path) -> dict:
    kw_file = next(experiment_dir.glob("*.kw.txt"), None)
    if kw_file is None:
        return {}
    text = kw_file.read_text()
    start, end = text.index("{"), text.rindex("}") + 1
    return json.loads(text[start:end])


def parse_image_index(experiment_dir: Path) -> pd.DataFrame:
    idx_file = experiment_dir / "images" / "image.index.txt"
    df = pd.read_csv(idx_file, sep="\t", skiprows=[0, 1])
    for col in ("Row", "Column", "Field", "Plane", "Channel", "Timepoint"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.dropna(subset=["Row", "Column", "Field", "Plane", "Channel"])


def parse_flatfield_xml(experiment_dir: Path) -> list[dict]:
    ff_dir = experiment_dir / "flatfieldcorrection"
    if not ff_dir.exists():
        return []
    xml_file = next(ff_dir.glob("*.xml"), None)
    if xml_file is None:
        return []
    root = ET.parse(xml_file).getroot()
    profiles = []
    for entry in root.findall(".//h:Entry", NS):
        ch_id = entry.attrib.get("ChannelID")
        profile_text = _find_text(entry, "h:FlatfieldProfile")
        if profile_text:
            profiles.append({"channel_id": int(ch_id), "profile": profile_text})
    return profiles


# ---------------------------------------------------------------------------
# Row/col helpers
# ---------------------------------------------------------------------------


def row_num_to_letter(row_num: int) -> str:
    result = ""
    while row_num > 0:
        row_num, rem = divmod(row_num - 1, 26)
        result = chr(ord("A") + rem) + result
    return result


# ---------------------------------------------------------------------------
# Zarr writers
# ---------------------------------------------------------------------------


def write_field_zarr(
    field_path: Path,
    image_data: np.ndarray,
    channel_names: list[str],
    pixel_size_xy_um: float,
    pixel_size_z_um: float,
):
    """Write a single TCZYX field as OME-Zarr v3."""
    field_path.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(field_path), mode="w", zarr_format=ZARR_FORMAT)

    shape = image_data.shape
    chunks = list(shape)
    chunks[-2] = min(shape[-2], 1024)
    chunks[-1] = min(shape[-1], 1024)
    dask_data = da.from_array(image_data, chunks=tuple(chunks))

    axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    coord_transforms = [[{
        "type": "scale",
        "scale": [1.0, 1.0, pixel_size_z_um, pixel_size_xy_um, pixel_size_xy_um],
    }]]
    dtype_max = float(np.iinfo(image_data.dtype).max)
    omero_channels = [
        {
            "label": name,
            "active": True,
            "color": CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
            "window": {"start": 0.0, "end": dtype_max, "min": 0.0, "max": dtype_max},
        }
        for i, name in enumerate(channel_names)
    ]

    write_image(
        image=dask_data,
        group=root,
        axes=axes,
        coordinate_transformations=coord_transforms,
        scaler=Scaler(method="gaussian", downscale=2, max_layer=0, labeled=False),
        omero={"channels": omero_channels},
    )

    ome_attrs = dict(root.attrs.get("ome", {}))
    ome_attrs["omero"] = {"channels": omero_channels}
    root.attrs["ome"] = ome_attrs


def write_plate_metadata(
    plate_path: Path,
    wells_by_row_col: set,
    field_count: int,
    experiment_meta: dict,
    index_meta: dict,
    kw_meta: dict,
    flatfield: list,
):
    sorted_rows = sorted(set(r for r, _ in wells_by_row_col))
    sorted_cols = sorted(set(c for _, c in wells_by_row_col), key=int)
    meta = {
        "zarr_format": ZARR_FORMAT,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "plate": {
                    "version": "0.5",
                    "name": experiment_meta.get("plate_name", plate_path.stem),
                    "field_count": field_count,
                    "acquisitions": [{"id": 0}],
                    "columns": [{"name": c} for c in sorted_cols],
                    "rows": [{"name": r} for r in sorted_rows],
                    "wells": sorted(
                        [
                            {
                                "path": f"{r}/{c}",
                                "rowIndex": sorted_rows.index(r),
                                "columnIndex": sorted_cols.index(c),
                            }
                            for r, c in wells_by_row_col
                        ],
                        key=lambda w: w["path"],
                    ),
                },
            },
            "experiment_metadata": experiment_meta,
            "index_metadata": index_meta,
            "instrument_summary": {k: v for k, v in kw_meta.items() if k != "__URL"},
        },
    }
    if flatfield:
        meta["attributes"]["flatfield_profiles"] = flatfield
    with open(plate_path / "zarr.json", "w") as f:
        json.dump(meta, f, indent=2)


def write_well_metadata(well_path: Path, field_indices: list[int]):
    well_path.mkdir(parents=True, exist_ok=True)
    meta = {
        "zarr_format": ZARR_FORMAT,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "well": {
                    "version": "0.5",
                    "images": [{"path": str(i), "acquisition": 0} for i in field_indices],
                },
            }
        },
    }
    with open(well_path / "zarr.json", "w") as f:
        json.dump(meta, f, indent=2)


def write_group_metadata(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    meta = {"zarr_format": ZARR_FORMAT, "node_type": "group", "attributes": {}}
    with open(path / "zarr.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Per-field worker
# ---------------------------------------------------------------------------


def _process_field(args):
    (field_path, img_dir, n_channels, n_planes, img_size_y, img_size_x,
     channel_names, px_um, z_spacing_um, acq_rows) = args
    img_dir = Path(img_dir)

    stack = np.zeros((1, n_channels, n_planes, img_size_y, img_size_x), dtype=np.uint16)
    acq_meta = []

    for ch_idx, plane_idx, url, acq in acq_rows:
        tiff_path = img_dir / url
        if not tiff_path.exists():
            continue
        stack[0, ch_idx, plane_idx] = tiff_imread(str(tiff_path))
        if acq is not None:
            acq_meta.append(acq)

    write_field_zarr(Path(field_path), stack, channel_names, px_um, z_spacing_um)

    zj_path = Path(field_path) / "zarr.json"
    if zj_path.exists() and acq_meta:
        zj = json.loads(zj_path.read_text())
        zj.setdefault("attributes", {})["acquisition_metadata"] = acq_meta
        zj_path.write_text(json.dumps(zj, indent=2))

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def convert(experiment_dir: Path, output_dir: Path | None = None):
    t0 = time.time()
    print(f"Parsing metadata from {experiment_dir}...")
    experiment_meta = parse_experiment_xml(experiment_dir)
    index_meta = parse_index_xml(experiment_dir)
    kw_meta = parse_kw_txt(experiment_dir)
    image_index = parse_image_index(experiment_dir)
    flatfield = parse_flatfield_xml(experiment_dir)

    plate_name = experiment_meta.get("plate_name", "plate")
    if output_dir is None:
        output_dir = experiment_dir.parent
    plate_path = output_dir / f"{plate_name}.zarr"

    channel_names = [ch["name"] for ch in index_meta["channels"]]
    n_channels = len(channel_names)
    n_planes = int(image_index["Plane"].max())
    img_size_x = int(index_meta["channels"][0].get("image_size_x", 2160))
    img_size_y = int(index_meta["channels"][0].get("image_size_y", 2160))
    px_um = index_meta["channels"][0].get("pixel_size_x_um", 0.3)

    # Z spacing from median diff of AbsoluteZ positions
    sample = image_index[
        (image_index["Row"] == image_index["Row"].iloc[0]) &
        (image_index["Column"] == image_index["Column"].iloc[0]) &
        (image_index["Field"] == image_index["Field"].iloc[0]) &
        (image_index["Channel"] == 1)
    ]
    z_positions = sample.sort_values("Plane")["AbsoluteZ"].values
    z_spacing_um = float(np.median(np.diff(z_positions))) * 1e6 if len(z_positions) > 1 else 1.0

    print(f"Pixel size: {px_um:.4f} µm (XY), {z_spacing_um:.2f} µm (Z)")
    print(f"Channels: {channel_names}")
    print(f"Planes: {n_planes}, Fields: {image_index['Field'].max()}")
    print(f"Workers: {N_WORKERS}")

    groups = image_index.groupby(["Row", "Column", "Field"])
    wells_by_row_col: set[tuple[str, str]] = set()
    fields_by_well: dict[tuple[str, str], list[int]] = {}
    img_dir = experiment_dir / "images"
    plate_path.mkdir(parents=True, exist_ok=True)

    work_items = []
    for (row_num, col_num, field_num), grp in groups:
        row_letter = row_num_to_letter(int(row_num))
        col_str = str(int(col_num))
        field_idx = int(field_num) - 1

        wells_by_row_col.add((row_letter, col_str))
        fields_by_well.setdefault((row_letter, col_str), []).append(field_idx)

        field_path = plate_path / row_letter / col_str / str(field_idx)
        acq_rows = []
        for _, img_row in grp.iterrows():
            ch_idx = int(img_row["Channel"]) - 1
            plane_idx = int(img_row["Plane"]) - 1
            url = img_row["__URL"]
            acq = None
            if ch_idx == 0:
                acq = {
                    "plane": int(img_row["Plane"]),
                    "absolute_z_m": float(img_row["AbsoluteZ"]),
                    "temperature_c": float(img_row["Temperature"]),
                    "co2_pct": float(img_row["CO2"]),
                    "date": str(img_row["Date"]),
                }
            acq_rows.append((ch_idx, plane_idx, url, acq))

        work_items.append((
            str(field_path), str(img_dir),
            n_channels, n_planes, img_size_y, img_size_x,
            channel_names, px_um, z_spacing_um, acq_rows,
        ))

    print(f"\nWriting {len(work_items)} fields to {plate_path} ({N_WORKERS} workers)...")
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_process_field, item): item for item in work_items}
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 100 == 0 or done == len(work_items):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(work_items) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(work_items)} fields  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    print("\nWriting HCS plate/well metadata...")
    field_count = max(len(v) for v in fields_by_well.values())
    write_plate_metadata(plate_path, wells_by_row_col, field_count,
                         experiment_meta, index_meta, kw_meta, flatfield)

    for row_letter in sorted(set(r for r, _ in wells_by_row_col)):
        write_group_metadata(plate_path / row_letter)
    for (row_letter, col_str), field_indices in sorted(fields_by_well.items()):
        write_well_metadata(plate_path / row_letter / col_str, sorted(field_indices))

    elapsed = time.time() - t0
    print(f"\nDone. {len(work_items)} fields across {len(wells_by_row_col)} wells → {plate_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Opera Phenix TIFF export to OME-Zarr v3 HCS plate store."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to the Opera Phenix experiment directory (contains experiment XML, "
             "index/, images/, flatfieldcorrection/).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Directory to write <plate_name>.zarr (default: parent of experiment_dir).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.experiment_dir.is_dir():
        sys.exit(f"[error] not a directory: {args.experiment_dir}")
    convert(args.experiment_dir, args.output_dir)
