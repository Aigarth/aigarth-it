"""Regression tests for ``ITUClArithmeticAdditionIntI2x7O8.reflect_many()``.

A bad ``dataset_fpath`` (a missing file, or a file whose metadata header fails
validation) must surface as the library's own ``AigarthITUError``. Before the
fix, ``dataset_meta_csv()`` raised before ``csv_dh`` was bound, so the ``finally``
block's ``csv_dh.close()`` raised ``UnboundLocalError`` and masked the intended
error on a routine user mistake.
"""
from pathlib import Path

import pytest

from aigarth_it.exceptions import AigarthITUError
from aigarth_it.icap.itucl_aai_i2x7o8 import ITUClArithmeticAdditionIntI2x7O8


@pytest.fixture
def itu() -> ITUClArithmeticAdditionIntI2x7O8:
    return ITUClArithmeticAdditionIntI2x7O8()


def test_reflect_many_missing_file_raises_itu_error(itu, tmp_path):
    # Nonexistent dataset path: dataset_meta_csv() raises OSError before the file
    # handle is opened, so the caller must see AigarthITUError, not UnboundLocalError.
    with pytest.raises(AigarthITUError):
        itu.reflect_many(dataset_fpath=tmp_path / "does_not_exist.csv")


def test_reflect_many_malformed_file_raises_itu_error(itu, tmp_path):
    # A file whose metadata header fails validation also raises before the handle
    # is opened; the error must still be the library's own.
    bad = tmp_path / "bad.csv"
    bad.write_text("garbage,only,one,line\n")
    with pytest.raises(AigarthITUError):
        itu.reflect_many(dataset_fpath=bad)
