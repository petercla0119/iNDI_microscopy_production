"""Unit and integration tests for scripts/2_opera_to_zarr.py.

Run unit tests only (fast):
    python -m pytest tests/test_opera_to_zarr.py -v -m "not slow"

Run everything including the integration test:
    python -m pytest tests/test_opera_to_zarr.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load the script module by path (named with leading digit, can't plain import)
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parent.parent / "scripts" / "2_opera_to_zarr.py"
_spec = importlib.util.spec_from_file_location("opera_to_zarr", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

row_num_to_letter = _mod.row_num_to_letter
parse_experiment_xml = _mod.parse_experiment_xml
parse_index_xml = _mod.parse_index_xml
parse_kw_txt = _mod.parse_kw_txt
parse_image_index = _mod.parse_image_index
parse_flatfield_xml = _mod.parse_flatfield_xml
write_group_metadata = _mod.write_group_metadata
write_well_metadata = _mod.write_well_metadata
write_plate_metadata = _mod.write_plate_metadata
write_field_zarr = _mod.write_field_zarr
_process_field = _mod._process_field
convert = _mod.convert

# ---------------------------------------------------------------------------
# XML namespace used throughout
# ---------------------------------------------------------------------------

NS_URI = "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"
NS_DECL = f'xmlns="{NS_URI}"'


# ---------------------------------------------------------------------------
# XML fixture helpers
# ---------------------------------------------------------------------------


def _write_experiment_xml(
    path: Path,
    *,
    plate_name: str = "TESTPLATE",
    measurement_id: str = "uuid-001",
    date: str = "2026-01-01",
    serial: str = "SER001",
    user: str = "testuser",
    include_instrument: bool = True,
) -> None:
    inst_block = ""
    if include_instrument:
        inst_block = f"""
  <InstrumentDescription>
    <Type>Phenix</Type>
    <SoftwareVersion>5.2.0</SoftwareVersion>
    <Serial>INSTR001</Serial>
    <Objectives>
      <Objective Version="1">
        <Name>20x Air, NA 0.4</Name>
        <Magnification>20</Magnification>
        <NumAperture>0.4</NumAperture>
        <WorkingDistance>0.00828</WorkingDistance>
      </Objective>
    </Objectives>
  </InstrumentDescription>"""
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<Measurement {NS_DECL}>
  <MeasurementID>{measurement_id}</MeasurementID>
  <Date>{date}</Date>
  <Serial>{serial}</Serial>
  <UserName>{user}</UserName>
  <InitialPlateName>{plate_name}</InitialPlateName>{inst_block}
</Measurement>"""
    path.write_text(xml, encoding="utf-8")


def _write_index_xml(
    path: Path,
    channels: list[dict],
    *,
    plate_id: str = "TESTPLATE",
    plate_type: str = "384 Test",
    plate_rows: str = "16",
    plate_cols: str = "24",
    img_size: int = 64,
) -> None:
    """Build a minimal index XML with one Map containing channel entries."""
    entries = ""
    for ch in channels:
        ch_id_attr = f' ChannelID="{ch["channel_id"]}"' if ch.get("channel_id") is not None else ""
        ch_name = ch.get("name", "Unknown")
        res_x = ch.get("res_x", "2.9670000E-07")
        res_y = ch.get("res_y", "2.9670000E-07")
        entries += f"""      <Entry{ch_id_attr}>
        <ChannelName>{ch_name}</ChannelName>
        <ChannelType>Fluorescence</ChannelType>
        <MainExcitationWavelength>405</MainExcitationWavelength>
        <MainEmissionWavelength>450</MainEmissionWavelength>
        <ImageSizeX>{img_size}</ImageSizeX>
        <ImageSizeY>{img_size}</ImageSizeY>
        <ExposureTime>0.050</ExposureTime>
        <CameraType>sCMOS</CameraType>
        <ObjectiveMagnification>20</ObjectiveMagnification>
        <ObjectiveNA>0.4</ObjectiveNA>
        <ImageResolutionX>{res_x}</ImageResolutionX>
        <ImageResolutionY>{res_y}</ImageResolutionY>
      </Entry>
"""
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<EvaluationInputData {NS_DECL}>
  <Plates>
    <Plate>
      <PlateID>{plate_id}</PlateID>
      <PlateTypeName>{plate_type}</PlateTypeName>
      <PlateRows>{plate_rows}</PlateRows>
      <PlateColumns>{plate_cols}</PlateColumns>
    </Plate>
  </Plates>
  <Maps>
    <Map>
{entries}    </Map>
  </Maps>
</EvaluationInputData>"""
    path.write_text(xml, encoding="utf-8")


def _write_flatfield_xml(path: Path, channels: list[dict]) -> None:
    """Build a minimal flatfield XML."""
    entries = ""
    for ch in channels:
        entries += f"""    <Entry ChannelID="{ch['channel_id']}">
      <FlatfieldProfile>{ch['profile']}</FlatfieldProfile>
    </Entry>
"""
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<Results {NS_DECL}>
  <Map>
{entries}  </Map>
</Results>"""
    path.write_text(xml, encoding="utf-8")


def _make_experiment_dir(
    tmp_path: Path,
    *,
    channels: list[dict] | None = None,
    n_rows: int = 1,
    n_cols: int = 1,
    n_fields: int = 1,
    n_planes: int = 1,
    img_size: int = 8,
    include_flatfield: bool = False,
    flatfield_channels: list[dict] | None = None,
) -> Path:
    """Create a minimal synthetic experiment directory for convert() tests."""
    if channels is None:
        channels = [{"channel_id": 1, "name": "DAPI"}]

    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    _write_experiment_xml(exp_dir / "experiment.xml", plate_name="SYNTHPLATE")

    index_dir = exp_dir / "index"
    index_dir.mkdir()
    _write_index_xml(index_dir / "index.xml", channels, img_size=img_size)

    img_dir = exp_dir / "images"
    img_dir.mkdir()

    # Write image.index.txt
    from tifffile import imwrite as tiff_imwrite

    header_rows = [
        "Version\t1.0.2",
        "Measurement\ttest-uuid",
    ]
    col_headers = "Row\tColumn\tField\tPlane\tChannel\tTimepoint\tFlags\tAbsoluteZ\tTemperature\tCO2\tDate\t__URL\t__Size\t__Checksum"
    data_rows = []
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            for f in range(1, n_fields + 1):
                for p in range(1, n_planes + 1):
                    for ch_entry in channels:
                        ch_num = ch_entry["channel_id"]
                        url = f"r{r:02d}c{c:02d}/r{r:02d}c{c:02d}f{f:02d}p{p:02d}-ch{ch_num:02d}t01.tiff"
                        tiff_subdir = img_dir / f"r{r:02d}c{c:02d}"
                        tiff_subdir.mkdir(exist_ok=True)
                        tiff_path = img_dir / url
                        tiff_imwrite(str(tiff_path), np.zeros((img_size, img_size), dtype=np.uint16))
                        data_rows.append(
                            f"{r}\t{c}\t{f}\t{p}\t{ch_num}\t1\t1"
                            f"\t0.135{p:04d}\t27.0\t0.06\t2026-01-01T00:00:00Z"
                            f"\t{url}\t100\tabc123"
                        )

    idx_content = "\n".join(header_rows + [col_headers] + data_rows) + "\n"
    (img_dir / "image.index.txt").write_text(idx_content)

    if include_flatfield:
        ff_dir = exp_dir / "flatfieldcorrection"
        ff_dir.mkdir()
        ff_chans = flatfield_channels or [{"channel_id": ch["channel_id"], "profile": "ff_data"} for ch in channels]
        _write_flatfield_xml(ff_dir / "flatfield.xml", ff_chans)

    return exp_dir


# ---------------------------------------------------------------------------
# row_num_to_letter
# ---------------------------------------------------------------------------


class TestRowNumToLetter:
    def test_first_letter(self):
        assert row_num_to_letter(1) == "A"

    def test_second_letter(self):
        assert row_num_to_letter(2) == "B"

    def test_last_single_letter(self):
        assert row_num_to_letter(26) == "Z"

    def test_wraps_to_double_letter(self):
        # 27th row must be "AA" (Excel-style alpha encoding)
        assert row_num_to_letter(27) == "AA"


# ---------------------------------------------------------------------------
# parse_experiment_xml
# ---------------------------------------------------------------------------


class TestParseExperimentXml:
    def test_happy_path(self, tmp_path):
        _write_experiment_xml(tmp_path / "exp.xml", plate_name="WARD00001", measurement_id="m-001")
        result = parse_experiment_xml(tmp_path)
        assert result["plate_name"] == "WARD00001"
        assert result["measurement_id"] == "m-001"
        assert result["user"] == "testuser"

    def test_with_instrument_block(self, tmp_path):
        _write_experiment_xml(tmp_path / "exp.xml", include_instrument=True)
        result = parse_experiment_xml(tmp_path)
        assert "instrument" in result
        assert result["instrument"]["type"] == "Phenix"
        assert len(result["objectives"]) == 1
        assert result["objectives"][0]["magnification"] == "20"

    def test_no_instrument_block(self, tmp_path):
        _write_experiment_xml(tmp_path / "exp.xml", include_instrument=False)
        result = parse_experiment_xml(tmp_path)
        assert "instrument" not in result
        assert result["plate_name"] == "TESTPLATE"

    def test_missing_optional_fields_return_none_not_crash(self, tmp_path):
        xml = f"""<?xml version="1.0"?>
<Measurement {NS_DECL}>
  <MeasurementID>m-partial</MeasurementID>
</Measurement>"""
        (tmp_path / "partial.xml").write_text(xml)
        result = parse_experiment_xml(tmp_path)
        assert result["measurement_id"] == "m-partial"
        assert result["plate_name"] is None
        assert result["date"] is None
        assert result["serial"] is None

    def test_missing_xml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_experiment_xml(tmp_path)


# ---------------------------------------------------------------------------
# parse_index_xml
# ---------------------------------------------------------------------------


class TestParseIndexXml:
    def test_single_channel(self, tmp_path):
        idx_dir = tmp_path / "index"
        idx_dir.mkdir()
        _write_index_xml(idx_dir / "idx.xml", [{"channel_id": 1, "name": "DAPI"}])
        result = parse_index_xml(tmp_path)
        assert len(result["channels"]) == 1
        assert result["channels"][0]["name"] == "DAPI"
        assert result["channels"][0]["channel_id"] == 1
        assert result["plate_id"] == "TESTPLATE"

    def test_multiple_channels(self, tmp_path):
        idx_dir = tmp_path / "index"
        idx_dir.mkdir()
        chs = [
            {"channel_id": 1, "name": "DAPI"},
            {"channel_id": 2, "name": "Alexa 488"},
            {"channel_id": 3, "name": "CF568"},
            {"channel_id": 4, "name": "Alexa 647"},
        ]
        _write_index_xml(idx_dir / "idx.xml", chs)
        result = parse_index_xml(tmp_path)
        assert len(result["channels"]) == 4
        names = [ch["name"] for ch in result["channels"]]
        assert names == ["DAPI", "Alexa 488", "CF568", "Alexa 647"]

    def test_missing_channel_id_attribute(self, tmp_path):
        idx_dir = tmp_path / "index"
        idx_dir.mkdir()
        # Entry with no ChannelID attribute
        xml = f"""<?xml version="1.0"?>
<EvaluationInputData {NS_DECL}>
  <Plates><Plate><PlateID>P1</PlateID><PlateTypeName>T</PlateTypeName>
  <PlateRows>16</PlateRows><PlateColumns>24</PlateColumns></Plate></Plates>
  <Maps>
    <Map>
      <Entry>
        <ChannelName>NoChanID</ChannelName>
        <ImageSizeX>64</ImageSizeX>
        <ImageSizeY>64</ImageSizeY>
      </Entry>
    </Map>
  </Maps>
</EvaluationInputData>"""
        (idx_dir / "idx.xml").write_text(xml)
        result = parse_index_xml(tmp_path)
        assert len(result["channels"]) == 1
        assert result["channels"][0]["channel_id"] is None
        assert result["channels"][0]["name"] == "NoChanID"

    def test_pixel_size_converted_to_microns(self, tmp_path):
        idx_dir = tmp_path / "index"
        idx_dir.mkdir()
        # 0.2967e-6 m → 0.2967 µm
        _write_index_xml(idx_dir / "idx.xml", [{"channel_id": 1, "name": "DAPI", "res_x": "2.967E-07"}])
        result = parse_index_xml(tmp_path)
        assert result["channels"][0]["pixel_size_x_um"] == pytest.approx(0.2967, rel=1e-3)

    def test_missing_index_xml_raises(self, tmp_path):
        (tmp_path / "index").mkdir()
        with pytest.raises(FileNotFoundError):
            parse_index_xml(tmp_path)


# ---------------------------------------------------------------------------
# parse_kw_txt
# ---------------------------------------------------------------------------


class TestParseKwTxt:
    def test_absent_returns_empty_dict(self, tmp_path):
        result = parse_kw_txt(tmp_path)
        assert result == {}

    def test_embedded_json_parsed(self, tmp_path):
        kw = tmp_path / "test.kw.txt"
        kw.write_text('Version\t1.0.1\n{"KEY": "value", "NUM": 42}\n')
        result = parse_kw_txt(tmp_path)
        assert result["KEY"] == "value"
        assert result["NUM"] == 42

    def test_prefix_text_ignored(self, tmp_path):
        kw = tmp_path / "test.kw.txt"
        kw.write_text('Version\t1.0.1\nMeasurement\tsome-uuid\n{"A": 1}\n')
        result = parse_kw_txt(tmp_path)
        assert result == {"A": 1}


# ---------------------------------------------------------------------------
# parse_image_index
# ---------------------------------------------------------------------------


class TestParseImageIndex:
    def _write_index(self, path: Path, rows: list[str]) -> None:
        header = [
            "Version\t1.0.2",
            "Measurement\ttest",
            "Row\tColumn\tField\tPlane\tChannel\tTimepoint\tFlags\tAbsoluteZ\tTemperature\tCO2\tDate\t__URL\t__Size\t__Checksum",
        ]
        (path / "images").mkdir(exist_ok=True)
        (path / "images" / "image.index.txt").write_text("\n".join(header + rows) + "\n")

    def test_happy_path(self, tmp_path):
        self._write_index(
            tmp_path,
            ["3\t3\t1\t1\t1\t1\t1\t0.135\t27.0\t0.06\t2026-01-01T00:00:00Z\tr03c03/img.tiff\t100\tabc"]
        )
        df = parse_image_index(tmp_path)
        assert len(df) == 1
        assert int(df.iloc[0]["Row"]) == 3
        assert int(df.iloc[0]["Column"]) == 3
        assert str(df.iloc[0]["__URL"]) == "r03c03/img.tiff"

    def test_non_numeric_row_col_field_dropped(self, tmp_path):
        self._write_index(
            tmp_path,
            [
                "3\t3\t1\t1\t1\t1\t1\t0.135\t27.0\t0.06\t2026-01-01T00:00:00Z\tr03c03/img.tiff\t100\tabc",
                "bad\t3\t1\t1\t1\t1\t1\t0.135\t27.0\t0.06\t2026-01-01T00:00:00Z\tr03c03/bad.tiff\t100\tabc",
            ],
        )
        df = parse_image_index(tmp_path)
        # Row='bad' → NaN → dropped; only the valid row survives
        assert len(df) == 1
        assert int(df.iloc[0]["Row"]) == 3

    def test_non_numeric_column_dropped(self, tmp_path):
        self._write_index(
            tmp_path,
            ["3\tX\t1\t1\t1\t1\t1\t0.135\t27.0\t0.06\t2026-01-01T00:00:00Z\tr03cX/img.tiff\t100\tabc"],
        )
        df = parse_image_index(tmp_path)
        assert len(df) == 0

    def test_multiple_rows_all_numeric(self, tmp_path):
        self._write_index(
            tmp_path,
            [
                "3\t3\t1\t1\t1\t1\t1\t0.135\t27.0\t0.06\t2026-01-01T00:00:00Z\tr03c03/a.tiff\t100\tabc",
                "3\t3\t1\t2\t1\t1\t1\t0.136\t27.0\t0.06\t2026-01-01T00:00:00Z\tr03c03/b.tiff\t100\tdef",
            ],
        )
        df = parse_image_index(tmp_path)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# parse_flatfield_xml
# ---------------------------------------------------------------------------


class TestParseFlatfieldXml:
    def test_dir_absent_returns_empty(self, tmp_path):
        result = parse_flatfield_xml(tmp_path)
        assert result == []

    def test_dir_present_no_xml_returns_empty(self, tmp_path):
        (tmp_path / "flatfieldcorrection").mkdir()
        result = parse_flatfield_xml(tmp_path)
        assert result == []

    def test_two_channels(self, tmp_path):
        ff_dir = tmp_path / "flatfieldcorrection"
        ff_dir.mkdir()
        _write_flatfield_xml(
            ff_dir / "ff.xml",
            [
                {"channel_id": 1, "profile": "profile_ch1"},
                {"channel_id": 2, "profile": "profile_ch2"},
            ],
        )
        result = parse_flatfield_xml(tmp_path)
        assert len(result) == 2
        assert result[0] == {"channel_id": 1, "profile": "profile_ch1"}
        assert result[1] == {"channel_id": 2, "profile": "profile_ch2"}

    def test_single_channel(self, tmp_path):
        ff_dir = tmp_path / "flatfieldcorrection"
        ff_dir.mkdir()
        _write_flatfield_xml(ff_dir / "ff.xml", [{"channel_id": 3, "profile": "only_one"}])
        result = parse_flatfield_xml(tmp_path)
        assert len(result) == 1
        assert result[0]["channel_id"] == 3


# ---------------------------------------------------------------------------
# write_group_metadata
# ---------------------------------------------------------------------------


class TestWriteGroupMetadata:
    def test_creates_zarr_json(self, tmp_path):
        write_group_metadata(tmp_path / "A")
        zj = json.loads((tmp_path / "A" / "zarr.json").read_text())
        assert zj["zarr_format"] == 3
        assert zj["node_type"] == "group"
        assert "attributes" in zj

    def test_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        write_group_metadata(deep)
        assert (deep / "zarr.json").exists()


# ---------------------------------------------------------------------------
# write_well_metadata
# ---------------------------------------------------------------------------


class TestWriteWellMetadata:
    def test_happy_path(self, tmp_path):
        well = tmp_path / "A" / "3"
        write_well_metadata(well, [0, 1, 2])
        zj = json.loads((well / "zarr.json").read_text())
        images = zj["attributes"]["ome"]["well"]["images"]
        assert len(images) == 3
        assert images[0] == {"path": "0", "acquisition": 0}
        assert images[2] == {"path": "2", "acquisition": 0}

    def test_field_indices_written_in_given_order(self, tmp_path):
        # Caller is responsible for sorting; writer preserves the order given.
        well = tmp_path / "B" / "5"
        write_well_metadata(well, [2, 0, 1])
        zj = json.loads((well / "zarr.json").read_text())
        paths = [img["path"] for img in zj["attributes"]["ome"]["well"]["images"]]
        assert paths == ["2", "0", "1"]

    def test_single_field(self, tmp_path):
        write_well_metadata(tmp_path / "C" / "1", [0])
        zj = json.loads((tmp_path / "C" / "1" / "zarr.json").read_text())
        assert len(zj["attributes"]["ome"]["well"]["images"]) == 1


# ---------------------------------------------------------------------------
# write_plate_metadata
# ---------------------------------------------------------------------------


class TestWritePlateMetadata:
    def _call(self, plate_path, wells, flatfield=None):
        plate_path.mkdir(parents=True, exist_ok=True)
        write_plate_metadata(
            plate_path,
            wells,
            field_count=1,
            experiment_meta={"plate_name": "TEST"},
            index_meta={"channels": []},
            kw_meta={},
            flatfield=flatfield or [],
        )
        return json.loads((plate_path / "zarr.json").read_text())

    def test_no_flatfield_key_absent(self, tmp_path):
        zj = self._call(tmp_path / "plate.zarr", {("A", "3")}, flatfield=[])
        assert "flatfield_profiles" not in zj["attributes"]

    def test_with_flatfield_key_present(self, tmp_path):
        ff = [{"channel_id": 1, "profile": "data"}]
        zj = self._call(tmp_path / "plate.zarr", {("A", "3")}, flatfield=ff)
        assert "flatfield_profiles" in zj["attributes"]
        assert zj["attributes"]["flatfield_profiles"] == ff

    def test_wells_sorted_by_path(self, tmp_path):
        wells = {("B", "3"), ("A", "5"), ("A", "3"), ("B", "1")}
        zj = self._call(tmp_path / "plate.zarr", wells)
        paths = [w["path"] for w in zj["attributes"]["ome"]["plate"]["wells"]]
        assert paths == sorted(paths), "wells must be sorted by path string"

    def test_rows_and_cols_lists(self, tmp_path):
        wells = {("A", "3"), ("A", "5"), ("B", "3")}
        zj = self._call(tmp_path / "plate.zarr", wells)
        plate = zj["attributes"]["ome"]["plate"]
        row_names = [r["name"] for r in plate["rows"]]
        col_names = [c["name"] for c in plate["columns"]]
        assert "A" in row_names and "B" in row_names
        # columns sorted numerically
        assert col_names.index("3") < col_names.index("5")

    def test_zarr_format_version(self, tmp_path):
        zj = self._call(tmp_path / "plate.zarr", {("A", "1")})
        assert zj["zarr_format"] == 3
        assert zj["attributes"]["ome"]["plate"]["version"] == "0.5"


# ---------------------------------------------------------------------------
# write_field_zarr
# ---------------------------------------------------------------------------


class TestWriteFieldZarr:
    def _make_array(self, n_channels=2, n_planes=1, h=8, w=8):
        return np.zeros((1, n_channels, n_planes, h, w), dtype=np.uint16)

    def test_zarr_json_exists(self, tmp_path):
        arr = self._make_array()
        write_field_zarr(tmp_path / "field", arr, ["DAPI", "GFP"], 0.297, 1.0)
        assert (tmp_path / "field" / "zarr.json").exists()

    def test_axes_count_is_five(self, tmp_path):
        arr = self._make_array()
        write_field_zarr(tmp_path / "field", arr, ["DAPI", "GFP"], 0.297, 1.0)
        zj = json.loads((tmp_path / "field" / "zarr.json").read_text())
        axes = zj["attributes"]["ome"]["multiscales"][0]["axes"]
        assert len(axes) == 5
        names = [a["name"] for a in axes]
        assert names == ["t", "c", "z", "y", "x"]

    def test_omero_channels_label_and_color(self, tmp_path):
        arr = self._make_array(n_channels=2)
        write_field_zarr(tmp_path / "field", arr, ["DAPI", "Alexa 488"], 0.297, 1.0)
        zj = json.loads((tmp_path / "field" / "zarr.json").read_text())
        channels = zj["attributes"]["ome"]["omero"]["channels"]
        assert len(channels) == 2
        assert channels[0]["label"] == "DAPI"
        assert channels[1]["label"] == "Alexa 488"
        # colors come from CHANNEL_COLORS = ["0000FF", "00FF00", ...]
        assert channels[0]["color"] == "0000FF"
        assert channels[1]["color"] == "00FF00"

    def test_array_dtype_is_uint16(self, tmp_path):
        import zarr as _zarr

        arr = self._make_array()
        write_field_zarr(tmp_path / "field", arr, ["DAPI", "GFP"], 0.297, 1.0)
        grp = _zarr.open_group(str(tmp_path / "field"), mode="r")
        # ome-zarr writes pyramid levels as "s0", "s1", … (not "0", "1")
        arr_back = grp["s0"]
        assert arr_back.dtype == np.uint16

    def test_single_channel(self, tmp_path):
        arr = self._make_array(n_channels=1)
        write_field_zarr(tmp_path / "field", arr, ["DAPI"], 0.297, 1.0)
        zj = json.loads((tmp_path / "field" / "zarr.json").read_text())
        channels = zj["attributes"]["ome"]["omero"]["channels"]
        assert len(channels) == 1
        assert channels[0]["label"] == "DAPI"


# ---------------------------------------------------------------------------
# _process_field
# ---------------------------------------------------------------------------


class TestProcessField:
    def _args(
        self, field_path: Path, img_dir: Path, acq_rows, n_channels=2, n_planes=1, size=8
    ):
        return (
            str(field_path),
            str(img_dir),
            n_channels,
            n_planes,
            size,
            size,
            ["DAPI", "GFP"],
            0.297,
            1.0,
            acq_rows,
        )

    def test_missing_tiff_returns_true_no_crash(self, tmp_path):
        """Slot stays zero when TIFF file does not exist; function returns True."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        acq_rows = [(0, 0, "r01c01/missing.tiff", {"plane": 1, "absolute_z_m": 0.1,
                                                     "temperature_c": 27.0, "co2_pct": 0.05,
                                                     "date": "2026-01-01"})]
        result = _process_field(self._args(tmp_path / "field", img_dir, acq_rows))
        assert result is True
        # Field zarr was written; array at level s0 (ome-zarr naming) should be all zeros
        import zarr as _zarr

        grp = _zarr.open_group(str(tmp_path / "field"), mode="r")
        data = grp["s0"][:]
        assert np.all(data == 0)

    def test_acq_metadata_only_for_ch_idx_zero(self, tmp_path):
        """Acquisition metadata is appended only for ch_idx==0 rows."""
        from tifffile import imwrite as tiff_imwrite

        img_dir = tmp_path / "images" / "r01c01"
        img_dir.mkdir(parents=True)

        ch0_url = "r01c01/ch0.tiff"
        ch1_url = "r01c01/ch1.tiff"
        tiff_imwrite(str(tmp_path / "images" / ch0_url), np.zeros((8, 8), dtype=np.uint16))
        tiff_imwrite(str(tmp_path / "images" / ch1_url), np.zeros((8, 8), dtype=np.uint16))

        acq_dict = {"plane": 1, "absolute_z_m": 0.135, "temperature_c": 27.0,
                    "co2_pct": 0.06, "date": "2026-01-01"}
        acq_rows = [
            (0, 0, ch0_url, acq_dict),   # ch_idx=0 → acq is set
            (1, 0, ch1_url, None),        # ch_idx=1 → acq is None
        ]

        _process_field(self._args(tmp_path / "field", tmp_path / "images", acq_rows))

        zj = json.loads((tmp_path / "field" / "zarr.json").read_text())
        acq_meta = zj.get("attributes", {}).get("acquisition_metadata", [])
        assert len(acq_meta) == 1
        assert acq_meta[0]["plane"] == 1

    def test_returns_true_on_success(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        result = _process_field(self._args(tmp_path / "field", img_dir, []))
        assert result is True


# ---------------------------------------------------------------------------
# convert() — minimal synthetic integration
# ---------------------------------------------------------------------------


class TestConvertSynthetic:
    """Full pipeline on tiny synthetic data — no real microscopy images required.

    ProcessPoolExecutor is replaced by ThreadPoolExecutor so that _process_field
    (loaded via importlib, not importable by name) can run without pickling.
    """

    @pytest.fixture(autouse=True)
    def use_threads(self):
        # ponytail: swap to threads so importlib-loaded _process_field doesn't need pickling
        with patch.object(_mod, "ProcessPoolExecutor", ThreadPoolExecutor):
            yield

    def test_single_well_single_channel(self, tmp_path):
        exp_dir = _make_experiment_dir(
            tmp_path,
            channels=[{"channel_id": 1, "name": "DAPI"}],
            n_rows=1, n_cols=1, n_fields=1, n_planes=2, img_size=8,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        convert(exp_dir, out_dir)

        plate = out_dir / "SYNTHPLATE.zarr"
        assert plate.exists()
        zj = json.loads((plate / "zarr.json").read_text())
        wells = zj["attributes"]["ome"]["plate"]["wells"]
        assert len(wells) >= 1
        # Row 1 → A, Col 1 → 1
        assert wells[0]["path"] == "A/1"
        well_zj = json.loads((plate / "A" / "1" / "zarr.json").read_text())
        assert "well" in well_zj["attributes"]["ome"]

    def test_two_channels_two_wells(self, tmp_path):
        exp_dir = _make_experiment_dir(
            tmp_path,
            channels=[{"channel_id": 1, "name": "DAPI"}, {"channel_id": 2, "name": "GFP"}],
            n_rows=1, n_cols=2, n_fields=1, n_planes=1, img_size=8,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        convert(exp_dir, out_dir)

        plate = out_dir / "SYNTHPLATE.zarr"
        zj = json.loads((plate / "zarr.json").read_text())
        wells = zj["attributes"]["ome"]["plate"]["wells"]
        assert len(wells) == 2

    def test_flatfield_profiles_in_plate_json(self, tmp_path):
        exp_dir = _make_experiment_dir(
            tmp_path,
            channels=[{"channel_id": 1, "name": "DAPI"}],
            include_flatfield=True,
            flatfield_channels=[{"channel_id": 1, "profile": "ff_data_here"}],
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        convert(exp_dir, out_dir)

        zj = json.loads((out_dir / "SYNTHPLATE.zarr" / "zarr.json").read_text())
        assert "flatfield_profiles" in zj["attributes"]
        assert zj["attributes"]["flatfield_profiles"][0]["profile"] == "ff_data_here"

    def test_no_flatfield_key_absent_from_plate_json(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path, include_flatfield=False)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        convert(exp_dir, out_dir)

        zj = json.loads((out_dir / "SYNTHPLATE.zarr" / "zarr.json").read_text())
        assert "flatfield_profiles" not in zj["attributes"]

    def test_field_array_shape_and_dtype(self, tmp_path):
        import zarr as _zarr

        exp_dir = _make_experiment_dir(
            tmp_path,
            channels=[{"channel_id": 1, "name": "DAPI"}, {"channel_id": 2, "name": "GFP"}],
            n_planes=3, img_size=8,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        convert(exp_dir, out_dir)

        # field_idx=0 (field_num=1 → field_idx=0)
        field_grp = _zarr.open_group(str(out_dir / "SYNTHPLATE.zarr" / "A" / "1" / "0"), mode="r")
        # ome-zarr writes pyramid levels as "s0", "s1", … not "0", "1"
        arr = field_grp["s0"]
        # Shape is (1, C, Z, H, W)
        assert arr.ndim == 5
        assert arr.shape[0] == 1      # T
        assert arr.shape[1] == 2      # C
        assert arr.shape[2] == 3      # Z (n_planes)
        assert arr.shape[3] > 0       # H
        assert arr.shape[4] > 0       # W
        assert arr.dtype == np.uint16

    def test_wells_sorted_in_plate_json(self, tmp_path):
        exp_dir = _make_experiment_dir(
            tmp_path,
            channels=[{"channel_id": 1, "name": "DAPI"}],
            n_rows=2, n_cols=3,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        convert(exp_dir, out_dir)

        zj = json.loads((out_dir / "SYNTHPLATE.zarr" / "zarr.json").read_text())
        paths = [w["path"] for w in zj["attributes"]["ome"]["plate"]["wells"]]
        assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# Integration test — real Opera Phenix data
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    "/Users/pmihack/claire/hs-array/data/D28_AB_Rep1"
    "/5c6bf125-6e0f-41be-a447-b03ff294c9cc"
)


@pytest.fixture(scope="session")
def real_data_dir():
    if not DATA_DIR.exists():
        pytest.skip("real data not available")
    return DATA_DIR


@pytest.mark.integration
@pytest.mark.slow
def test_real_data_convert(real_data_dir, tmp_path_factory):
    import zarr as _zarr

    out_dir = tmp_path_factory.mktemp("real_zarr_out")
    convert(real_data_dir, out_dir)

    # Plate zarr exists
    plate_dirs = list(out_dir.glob("*.zarr"))
    assert len(plate_dirs) == 1, "Expected exactly one .zarr output"
    plate_path = plate_dirs[0]

    # Plate zarr.json has wells
    zj = json.loads((plate_path / "zarr.json").read_text())
    wells = zj["attributes"]["ome"]["plate"]["wells"]
    assert len(wells) >= 1

    # flatfield_profiles present (this dataset has flatfield XMLs)
    assert "flatfield_profiles" in zj["attributes"]

    # Open first well's field 0 and check array shape / dtype
    first_well = wells[0]["path"]
    row_letter, col_str = first_well.split("/")
    field_path = plate_path / row_letter / col_str / "0"
    assert (plate_path / row_letter / col_str / "zarr.json").exists(), \
        "well zarr.json must exist"

    grp = _zarr.open_group(str(field_path), mode="r")
    arr = grp["s0"]  # ome-zarr names pyramid levels s0, s1, …
    assert arr.ndim == 5
    t, c, z, h, w = arr.shape
    assert t == 1
    assert c >= 1
    assert z >= 1
    assert h > 0
    assert w > 0
    assert arr.dtype == np.uint16
