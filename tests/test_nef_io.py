from synth_nmr.nef_io import write_nef_chemical_shifts, write_nef_file, write_nef_relaxation


def test_write_nef_file_basic(tmp_path):
    """Test writing a basic NEF file with restraints."""
    output_file = tmp_path / "test.nef"
    sequence = "GA"
    restraints = [
        {
            "chain_1": "A",
            "index_1": 1,
            "res_name_1": "GLY",
            "atom_name_1": "HA2",
            "chain_2": "A",
            "index_2": 2,
            "res_name_2": "ALA",
            "atom_name_2": "HN",
            "distance": 3.5,
            "upper_limit": 4.0,
            "lower_limit": 1.8,
        }
    ]
    write_nef_file(str(output_file), sequence, restraints)

    assert output_file.exists()
    content = output_file.read_text()

    assert "data_synth-nmr-project" in content
    assert "_nef_nmr_meta_data.program_name synth-nmr" in content
    assert "save_nef_sequence" in content
    assert "save_synthetic_noes" in content

    # Check sequence loop content
    assert "A 1 GLY protein" in content
    assert "A 2 ALA protein" in content

    # Check restraint content
    assert "1 1 A 1 GLY HA2 A 2 ALA HN 3.500 4.000 1.800 1.0" in content


def test_write_nef_relaxation(tmp_path):
    """Test writing relaxation data to NEF."""
    output_file = tmp_path / "relax.nef"
    sequence = "GA"

    # Mock relaxation data
    # {res_id: {'R1': ..., 'R2': ..., 'NOE': ...}}
    relaxation_data = {
        1: {"R1": 0.9, "R2": 35.0, "NOE": 0.8},
        2: {"R1": 1.2, "R2": 45.0, "NOE": 0.75},
    }

    write_nef_relaxation(str(output_file), sequence, relaxation_data, field_freq_mhz=600.0)

    assert output_file.exists()
    content = output_file.read_text()

    assert "save_R1_600MHz" in content
    assert "save_R2_600MHz" in content
    assert "save_Heteronuclear_NOE_600MHz" in content

    # Check units
    assert "_nef_nmr_measurement_list.measurement_unit s-1" in content  # for R1/R2
    assert "_nef_nmr_measurement_list.measurement_unit none" in content  # for NOE

    # Check values
    # Glycine (Res 1) R1=0.9
    assert "1 A 1 GLY N 0.9000" in content
    # Alanine (Res 2) R2=45.0
    assert "2 A 2 ALA N 45.0000" in content


def test_nef_sequence_conversion_unknown(tmp_path):
    """Test handling of unknown amino acids or non-standard characters."""
    output_file = tmp_path / "unknown.nef"
    sequence = "X"
    restraints = []

    write_nef_file(str(output_file), sequence, restraints)

    content = output_file.read_text()
    # Expect "UNK" for X
    assert "A 1 UNK protein" in content


def test_write_nef_chemical_shifts(tmp_path):
    """Test writing chemical shift list to NEF."""
    output_file = tmp_path / "shifts.nef"
    sequence = "MA"

    # Mock shift data
    # Dict[chain -> res_id -> atom -> val]
    shifts = {
        "A": {
            1: {"CA": 55.0, "N": 120.0, "H": 8.0, "CB": 30.0},
            2: {
                "CA": 52.0,
                "N": 122.0,
                "H": 8.2,
            },  # No CB for Ala in this mock? (ALA has CB, but just testing IO)
        }
    }

    write_nef_chemical_shifts(str(output_file), sequence, shifts)

    assert output_file.exists()
    content = output_file.read_text()

    assert "save_chemical_shift_list" in content
    assert "_nef_chemical_shift_list.shift_unit ppm" in content

    # Check values
    # Res 1 Met pair
    assert "A 1 MET CA 55.000" in content
    # Res 2 Ala pair
    assert "A 2 ALA N 122.000" in content


def test_write_nef_relaxation_out_of_bounds(tmp_path):
    from synth_nmr.nef_io import write_nef_relaxation

    output_file = tmp_path / "relax_out.nef"
    sequence = "GA"
    # rid 3 is out of bounds, missing metric R2 handles val is None (line 188)
    relaxation_data = {
        1: {"R1": 0.9, "NOE": 0.8},  # missing R2
        3: {"R1": 1.5, "R2": 40.0, "NOE": 0.5},  # out of bounds rid 3
    }
    write_nef_relaxation(str(output_file), sequence, relaxation_data, field_freq_mhz=600.0)
    content = output_file.read_text()
    assert "A 3 UNK N 1.5000" in content


def test_write_nef_chemical_shifts_out_of_bounds(tmp_path):
    from synth_nmr.nef_io import write_nef_chemical_shifts

    output_file = tmp_path / "shifts_out.nef"
    sequence = "G"
    shifts = {"A": {2: {"CA": 55.0}}}  # out of bounds
    write_nef_chemical_shifts(str(output_file), sequence, shifts)
    content = output_file.read_text()
    assert "A 2 UNK CA 55.000" in content


def test_read_nef_restraints_save_close(tmp_path):
    from synth_nmr.nef_io import read_nef_restraints

    nef_content = """
save_nef_distance_restraint_list
   loop_
      _nef_distance_restraint.index
      _nef_distance_restraint.restraint_id
      _nef_distance_restraint.chain_code_1
      _nef_distance_restraint.sequence_code_1
      _nef_distance_restraint.residue_name_1
      _nef_distance_restraint.atom_name_1
      _nef_distance_restraint.chain_code_2
      _nef_distance_restraint.sequence_code_2
      _nef_distance_restraint.residue_name_2
      _nef_distance_restraint.atom_name_2
      _nef_distance_restraint.target_value
      _nef_distance_restraint.target_value_uncertainty
      _nef_distance_restraint.lower_limit
      _nef_distance_restraint.upper_limit
      _nef_distance_restraint.weight
      1 1 A 1 GLY H A 2 ALA HA 5.0 0.0 1.8 5.5 1.0
      2 2 A 1 GLY H A 2 ALA HA 5.0 0.0 1.8 5.5
      3 3 A 1 GLY H A 2 ALA HA nope 0.0 1.8 5.5 1.0
   stop_
save_
"""
    f = tmp_path / "test.nef"
    f.write_text(nef_content)
    r = read_nef_restraints(str(f))
    assert len(r) == 1


def test_read_nef_restraints_file_not_found(tmp_path):
    from synth_nmr.nef_io import read_nef_restraints

    f = tmp_path / "non_existent.nef"
    r = read_nef_restraints(str(f))
    assert r == []
