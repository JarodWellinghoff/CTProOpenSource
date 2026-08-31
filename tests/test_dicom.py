from pathlib import Path

import numpy as np
import pytest

# ruff: disable[E402]
pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from ctpro.dicom import load_ct_series

# ruff: enable[E402]


def _write_ct_slice(
    path: Path,
    *,
    series_uid: str,
    instance: int,
    z_position: float,
    stored_value: int,
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "CT"
    dataset.InstanceNumber = instance
    dataset.ImagePositionPatient = [0.0, 0.0, z_position]
    dataset.Rows = 4
    dataset.Columns = 4
    dataset.PixelSpacing = [0.5, 0.5]
    dataset.SliceThickness = 1.0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = -1000
    dataset.PixelData = np.full((4, 4), stored_value, dtype=np.int16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def test_load_ct_series_sorts_by_position_and_applies_hu(tmp_path: Path) -> None:
    uid = generate_uid()
    _write_ct_slice(
        tmp_path / "slice10.dcm",
        series_uid=uid,
        instance=10,
        z_position=10.0,
        stored_value=20,
    )
    _write_ct_slice(
        tmp_path / "slice2.dcm",
        series_uid=uid,
        instance=2,
        z_position=2.0,
        stored_value=5,
    )

    series = load_ct_series(tmp_path)
    assert series.images_hu.shape == (2, 4, 4)
    assert series.images_hu[0, 0, 0] == pytest.approx(-990)
    assert series.images_hu[1, 0, 0] == pytest.approx(-960)
    assert series.slice_spacing_mm == pytest.approx(8)
