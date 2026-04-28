import os
from unittest import mock

from synth_nmr.chemical_shifts import ShiftX2Predictor


def test_shiftx2_resolve_path_from_env(mocker):
    """Test that SHIFTX2_DIR environment variable is respected."""
    mocker.patch("shutil.which", side_effect=lambda x: x if "found" in x else None)

    with mock.patch.dict(os.environ, {"SHIFTX2_DIR": "/custom/path"}):
        # Mock os.path.isfile to return True for our fake executable
        mocker.patch("os.path.isfile", side_effect=lambda x: x == "/custom/path/shiftx2.py")
        # mocker.patch("os.access", return_value=True) # which handles this

        predictor = ShiftX2Predictor()
        # Since _resolve_path uses 'which', and we mocked 'which' to return the path if it exists
        # wait, my mocked 'which' returns x if "found" in x.
        # Let's fix the mock.

        mocker.patch("shutil.which", side_effect=lambda x: x if "/custom/path" in x else None)

        predictor = ShiftX2Predictor()
        assert predictor.executable == "/custom/path/shiftx2.py"
        assert predictor.is_available() is True


def test_shiftx2_resolve_path_typical_location(mocker):
    """Test that typical locations are searched."""
    # Ensure SHIFTX2_DIR is NOT set
    with mock.patch.dict(os.environ, {}, clear=False):
        if "SHIFTX2_DIR" in os.environ:
            del os.environ["SHIFTX2_DIR"]

        # Mock which to only find it in a typical location
        typical_path = os.path.expanduser("~/shiftx2/shiftx2.py")
        mocker.patch("shutil.which", side_effect=lambda x: x if x == typical_path else None)

        predictor = ShiftX2Predictor()
        assert predictor.executable == typical_path
        assert predictor.is_available() is True


def test_shiftx2_not_found(mocker):
    """Test behavior when SHIFTX2 is not found anywhere."""
    mocker.patch("shutil.which", return_value=None)
    with mock.patch.dict(os.environ, {}, clear=False):
        if "SHIFTX2_DIR" in os.environ:
            del os.environ["SHIFTX2_DIR"]

        predictor = ShiftX2Predictor()
        assert predictor.executable == "shiftx2.py"
        assert predictor.is_available() is False
