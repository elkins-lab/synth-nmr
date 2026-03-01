import pytest
import numpy as np
from synth_nmr.validation import compare_chemical_shifts, print_validation_report

def test_compare_chemical_shifts_empty():
    assert compare_chemical_shifts({}, {}) == {}

def test_compare_chemical_shifts_mismatched_chain():
    predicted = {"A": {1: {"CA": 50.0}}}
    reference = {"B": {1: {"CA": 50.0}}}
    assert compare_chemical_shifts(predicted, reference) == {}

def test_compare_chemical_shifts_mismatched_res():
    predicted = {"A": {1: {"CA": 50.0}}}
    reference = {"A": {2: {"CA": 50.0}}}
    assert compare_chemical_shifts(predicted, reference) == {}

def test_compare_chemical_shifts_insufficient_data(caplog):
    predicted = {"A": {1: {"CA": 50.0}}}
    reference = {"A": {1: {"CA": 51.0}}}
    
    # We need at least 2 points for a valid pearson correlation
    res = compare_chemical_shifts(predicted, reference, ["CA"])
    assert res == {}
    assert "Insufficient data to compare atom type: CA" in caplog.text

def test_compare_chemical_shifts_valid():
    predicted = {"A": {1: {"CA": 50.0}, 2: {"CA": 60.0}}}
    reference = {"A": {1: {"CA": 51.0}, 2: {"CA": 59.0}}}
    
    res = compare_chemical_shifts(predicted, reference, ["CA"])
    assert "CA" in res
    assert res["CA"]["count"] == 2
    assert res["CA"]["rmse"] == 1.0 # sqrt(((50-51)^2 + (60-59)^2)/2) = sqrt((1+1)/2) = 1.0
    assert "pearson" in res["CA"]

def test_print_validation_report(capsys):
    stats = {
        "CA": {"rmse": 1.0, "pearson": 0.99, "count": 2},
        "HA": {"rmse": 0.5, "pearson": 0.5, "count": 10}
    }
    
    print_validation_report(stats)
    
    captured = capsys.readouterr()
    assert "CA" in captured.out
    assert "1.000" in captured.out
    assert "HA" in captured.out

