"""
Data pipeline utility for downloading and parsing experimental BMRB chemical shifts.

This module provides functions to download NMR-STAR files from BMRB and PDB files
from RCSB, and parse them into a format suitable for training the Neural Shift Predictor.
"""

import logging
import os
import re
import urllib.request
from typing import Dict, List, Optional, Tuple

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
import numpy as np

logger = logging.getLogger(__name__)

# A curated list of high-quality, high-resolution matched PDB and BMRB pairs
# typically used in chemical shift prediction benchmarking (e.g., SPARTA+ training set).
# Format: (PDB_ID, BMRB_ID)
TRAINING_PAIRS = [
    ("1D3Z", 17769),  # Ubiquitin
    ("1GB1", 7359),  # Protein G
    ("2LZM", 15844),  # T4 Lysozyme
    ("1BRV", 4005),  # HIV-1 Protease
    ("1A1X", 4057),  # Calmodulin
    ("1C08", 4375),  # Barnase
    ("1P7E", 5387),  # Profilin
    ("1R0B", 6457),  # Ribonuclease
    ("2A3D", 68),  # Basic Pancreatic Trypsin Inhibitor (BPTI)
    ("2HBB", 7111),  # Hemoglobin
    ("1UBQ", 5387),  # Ubiquitin (alt)
]


def ensure_data_dir_exists(data_dir: str = "data") -> None:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def download_bmrb_file(bmrb_id: int, data_dir: str = "data") -> Optional[str]:
    """Downloads an NMR-STAR file from BMRB."""
    ensure_data_dir_exists(data_dir)
    filename = os.path.join(data_dir, f"bmr{bmrb_id}.str")

    if os.path.exists(filename):
        return filename

    url = f"https://bmrb.io/rest/bmrb/{bmrb_id}/nmr-star3"
    try:
        urllib.request.urlretrieve(url, filename)
        logger.info(f"Downloaded BMRB {bmrb_id} to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to download BMRB {bmrb_id}: {e}")
        return None


def download_pdb_file(pdb_id: str, data_dir: str = "data") -> Optional[str]:
    """Downloads a PDB file from RCSB."""
    ensure_data_dir_exists(data_dir)
    filename = os.path.join(data_dir, f"{pdb_id}.pdb")

    if os.path.exists(filename):
        return filename

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, filename)
        logger.info(f"Downloaded PDB {pdb_id} to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to download PDB {pdb_id}: {e}")
        return None


def parse_bmrb_shifts(filepath: str) -> Dict[int, Dict[str, float]]:
    """
    Parses an NMR-STAR file and extracts actual experimental chemical shifts.

    Returns:
        A dictionary mapping: {seq_id: {atom_name: shift_value_ppm}}
    """
    experimental_shifts: Dict[int, Dict[str, float]] = {}

    try:
        with open(filepath) as f:
            lines = f.readlines()
    except OSError:
        logger.error(f"Could not read {filepath}")
        return experimental_shifts

    in_atom_chem_shift_loop = False
    loop_headers: List[str] = []

    # Store all values, then average if there are multiple (e.g. HA2/HA3 -> HA)
    experimental_shifts_collected: Dict[int, Dict[str, List[float]]] = {}

    for line in lines:
        line = line.strip()
        if line == "loop_":
            in_atom_chem_shift_loop = False
            loop_headers = []

        elif line.startswith("_Atom_chem_shift.") and not in_atom_chem_shift_loop:
            loop_headers.append(line.split(".")[-1].strip())

        elif line == "stop_":
            in_atom_chem_shift_loop = False

        elif (
            len(loop_headers) > 0
            and not in_atom_chem_shift_loop
            and not line.startswith("save_")
            and not line.startswith("data_")
        ):
            in_atom_chem_shift_loop = True

        if (
            in_atom_chem_shift_loop
            and not (
                line.startswith("save_")
                or line.startswith("loop_")
                or line.startswith("stop_")
                or line.startswith("data_")
            )
            and len(line) > 0
        ):
            parts = re.split(r"\s+", line)
            if len(parts) == len(loop_headers):
                data = dict(zip(loop_headers, parts))

                try:
                    seq_id = int(data["Seq_ID"])
                    atom_id_bmrb = data["Atom_ID"]
                    atom_element_bmrb = data["Atom_type"]
                    val = float(data["Val"])
                except (ValueError, KeyError):
                    continue

                # Map BMRB atom names to standard NMR predictor names
                common_atom_name = None
                if atom_element_bmrb == "H" and atom_id_bmrb.startswith("HA"):
                    common_atom_name = "HA"
                elif atom_element_bmrb == "C" and atom_id_bmrb == "CA":
                    common_atom_name = "CA"
                elif atom_element_bmrb == "C" and atom_id_bmrb == "CB":
                    common_atom_name = "CB"
                elif atom_element_bmrb == "C" and atom_id_bmrb == "C":
                    common_atom_name = "C"
                elif atom_element_bmrb == "N" and atom_id_bmrb == "N":
                    common_atom_name = "N"
                elif atom_element_bmrb == "H" and atom_id_bmrb == "H":
                    common_atom_name = "H"

                if common_atom_name:
                    if seq_id not in experimental_shifts_collected:
                        experimental_shifts_collected[seq_id] = {}
                    if common_atom_name not in experimental_shifts_collected[seq_id]:
                        experimental_shifts_collected[seq_id][common_atom_name] = []
                    experimental_shifts_collected[seq_id][common_atom_name].append(val)

    # Process collected shifts (average stereopairs like HA2/HA3)
    for seq_id, atom_data in experimental_shifts_collected.items():
        experimental_shifts[seq_id] = {}
        for common_atom_name, values in atom_data.items():
            if values:
                experimental_shifts[seq_id][common_atom_name] = float(np.mean(values))

    return experimental_shifts


def parse_bmrb_j_couplings(filepath: str) -> Dict[int, Dict[str, float]]:
    """
    Parses an NMR-STAR file and extracts actual experimental scalar J-couplings.

    Returns:
        A dictionary mapping: {seq_id: {coupling_code: value_hz}}
        Common codes: '3JHNHA', '3JHAHB', '3JCCG'
    """
    experimental_couplings: Dict[int, Dict[str, float]] = {}

    try:
        with open(filepath) as f:
            lines = f.readlines()
    except OSError:
        logger.error(f"Could not read {filepath}")
        return experimental_couplings

    in_coupling_loop = False
    loop_headers: List[str] = []

    for line in lines:
        line = line.strip()
        if line == "loop_":
            in_coupling_loop = False
            loop_headers = []

        elif line.startswith("_Coupling_constant.") and not in_coupling_loop:
            loop_headers.append(line.split(".")[-1].strip())

        elif line == "stop_":
            in_coupling_loop = False

        elif (
            len(loop_headers) > 0
            and not in_coupling_loop
            and not line.startswith("save_")
            and not line.startswith("data_")
        ):
            in_coupling_loop = True

        if (
            in_coupling_loop
            and not (
                line.startswith("save_")
                or line.startswith("loop_")
                or line.startswith("stop_")
                or line.startswith("data_")
            )
            and len(line) > 0
        ):
            parts = re.split(r"\s+", line)
            if len(parts) == len(loop_headers):
                data = dict(zip(loop_headers, parts))

                try:
                    # Some files use Seq_ID_1, others use Seq_ID
                    seq_id = int(data.get("Seq_ID_1", data.get("Seq_ID", 0)))
                    code = data.get("Code", "UNKNOWN").upper().replace("-", "").replace("_", "")
                    val = float(data["Val"])
                except (ValueError, KeyError):
                    continue

                if seq_id > 0:
                    if seq_id not in experimental_couplings:
                        experimental_couplings[seq_id] = {}
                    experimental_couplings[seq_id][code] = val

    return experimental_couplings


def load_matched_dataset(
    data_dir: str = "data",
) -> List[Tuple[struc.AtomArray, Dict[int, Dict[str, float]]]]:
    """
    Downloads and prepares a list of (Structure, ExperimentalShifts) pairs.
    """
    dataset = []

    for pdb_id, bmrb_id in TRAINING_PAIRS:
        pdb_path = download_pdb_file(pdb_id, data_dir)
        bmrb_path = download_bmrb_file(bmrb_id, data_dir)

        if not pdb_path or not bmrb_path:
            logger.warning(f"Skipping pair ({pdb_id}, {bmrb_id}) due to download failure.")
            continue

        try:
            struct = pdb_io.PDBFile.read(pdb_path).get_structure(model=1)
            # Pre-filter for protein only
            struct = struct[struc.filter_amino_acids(struct)]

            shifts = parse_bmrb_shifts(bmrb_path)

            if len(struct) > 0 and len(shifts) > 0:
                dataset.append((struct, shifts))
                logger.info(
                    f"Loaded {pdb_id}/{bmrb_id}: {struc.get_residue_count(struct)} residues, {len(shifts)} shift assignments"
                )
        except Exception as e:
            logger.warning(f"Error parsing pair ({pdb_id}, {bmrb_id}): {e}")

    return dataset
