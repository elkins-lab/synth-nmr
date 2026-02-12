"""
NEF (NMR Exchange Format) I/O module for synth-pdb.
Handles writing synthetic NMR data to valid NEF files.
"""

import logging
import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)

def write_nef_file(
    filename: str,
    sequence: str,
    restraints: List[Dict],
    system_name: str = "synth-pdb-project"
) -> None:
    """
    Write a minimal valid NEF file containing sequence and distance restraints.
    
    Args:
        filename: Output filepath.
        sequence: Amino acid sequence string (1-letter code).
        restraints: List of restraint dicts from nmr.calculate_synthetic_noes.
        system_name: Name for the NEF saveframe.
    """
    logger.info(f"Writing NEF file to {filename}...")
    
    # NEF uses NMR-STAR syntax.
    # We will construct it manually to avoid dependencies, as our scope is limited.
    
    # 1. Header
    nc = "data_" + system_name + "\n"
    nc += "\n"
    nc += "_nef_nmr_meta_data.nef_format_version 1.1\n"
    nc += f"_nef_nmr_meta_data.creation_date {datetime.datetime.now().isoformat()}\n"
    nc += "_nef_nmr_meta_data.program_name synth-pdb\n"
    nc += "\n"
    
    # 2. Sequence (nef_sequence)
    # We need to expand 1-letter code to residues
    nc += "save_nef_sequence\n"
    nc += "   _nef_sequence.sf_category nef_sequence\n"
    nc += "   _nef_sequence.sf_framecode nef_sequence\n"
    nc += "\n"
    nc += "   loop_\n"
    nc += "      _nef_sequence.chain_code\n"
    nc += "      _nef_sequence.sequence_code\n"
    nc += "      _nef_sequence.residue_name\n"
    nc += "      _nef_sequence.residue_type\n" # protein
    
    # Mapping 1-letter to 3-letter (reuse data.py if possible, or simple map)
    # We'll do a quick local map or import proper one.
    # Importing from data for robustness
    from .data import STANDARD_AMINO_ACIDS, ONE_TO_THREE_LETTER_CODE
    
    # Invert mapping for 1->3
    # Actually ONE_TO_THREE_LETTER_CODE is {1: 3}
    one_to_three = ONE_TO_THREE_LETTER_CODE
    
    for i, char in enumerate(sequence):
        res_num = i + 1
        res_name = one_to_three.get(char, "UNK")
        nc += f"      A {res_num} {res_name} protein\n"
        
    nc += "   stop_\n"
    nc += "save_\n"
    nc += "\n"
    
    # 3. Distance Restraints (nef_distance_restraint_list)
    nc += "save_synthetic_noes\n"
    nc += "   _nef_distance_restraint_list.sf_category nef_distance_restraint_list\n"
    nc += "   _nef_distance_restraint_list.sf_framecode synthetic_noes\n"
    nc += "   _nef_distance_restraint_list.restraint_origin synthetic\n"
    nc += "\n"
    nc += "   loop_\n"
    nc += "      _nef_distance_restraint.index\n"
    nc += "      _nef_distance_restraint.restraint_id\n" # ID within list
    nc += "      _nef_distance_restraint.chain_code_1\n"
    nc += "      _nef_distance_restraint.sequence_code_1\n"
    nc += "      _nef_distance_restraint.residue_name_1\n"
    nc += "      _nef_distance_restraint.atom_name_1\n"
    nc += "      _nef_distance_restraint.chain_code_2\n"
    nc += "      _nef_distance_restraint.sequence_code_2\n"
    nc += "      _nef_distance_restraint.residue_name_2\n"
    nc += "      _nef_distance_restraint.atom_name_2\n"
    nc += "      _nef_distance_restraint.target_value\n"
    nc += "      _nef_distance_restraint.upper_limit\n"
    nc += "      _nef_distance_restraint.lower_limit\n"
    nc += "      _nef_distance_restraint.weight\n"
    
    for i, r in enumerate(restraints):
        idx = i + 1
        # Convert atom names to NEF standard?
        # NEF usually follows IUPAC. Biotite should output IUPAC-ish.
        # Naming is tricky (HB2 vs 2HB). We output what we have.
        
        row = f"      {idx} {idx} "
        row += f"{r['chain_1']} {r['residue_index_1']} {r['res_name_1']} {r['atom_name_1']} "
        row += f"{r['chain_2']} {r['residue_index_2']} {r['res_name_2']} {r['atom_name_2']} "
        row += f"{r['actual_distance']:.3f} {r['upper_limit']:.3f} {r['lower_limit']:.3f} 1.0\n"
        nc += row

    nc += "   stop_\n"
    nc += "save_\n"
    
    with open(filename, "w") as f:
        f.write(nc)
    
    logger.info(f"Successfully wrote {len(restraints)} restraints to {filename}.")

def write_nef_relaxation(
    filename: str,
    sequence: str,
    relaxation_data: Dict[int, Dict[str, float]],
    field_freq_mhz: float = 600.0,
    system_name: str = "synth-pdb-project"
) -> None:
    """
    Append Relaxation Data (R1, R2, NOE) to a NEF file.
    
    Args:
        filename: Path to EXISTING or NEW NEF file. (Currently overwrites or appends? 
                  Better to write all at once usually, but here we might append).
                  Actually, NEF stricture requires `data_` block. Appending is tricky.
                  For simplicity, this function assumes it's WRITING a new file solely for relaxation
                  OR the user should have called a master writer.
                  
                  Let's make this standalone for "Relaxation Only" or refactor `write_nef_file` later.
                  For Phase 8, let's just write a new file or support "append" mode carefully.
    """
    # ... Implementation ...
    # For simplicity in this iteration, we will just support writing a NEW file 
    # that contains Sequence + Relaxation. 
    # (Merging with NOEs would require a unified "write_all_nef" function).
    
    logger.info(f"Writing NEF relaxation data to {filename}...")
    
    nc = "data_" + system_name + "\n\n"
    nc += "_nef_nmr_meta_data.nef_format_version 1.1\n"
    nc += f"_nef_nmr_meta_data.creation_date {datetime.datetime.now().isoformat()}\n"
    nc += "_nef_nmr_meta_data.program_name synth-pdb\n\n"
    
    # Write Sequence (Required)
    nc += "save_nef_sequence\n"
    nc += "   _nef_sequence.sf_category nef_sequence\n"
    nc += "   _nef_sequence.sf_framecode nef_sequence\n\n"
    nc += "   loop_\n"
    nc += "      _nef_sequence.chain_code\n"
    nc += "      _nef_sequence.sequence_code\n"
    nc += "      _nef_sequence.residue_name\n"
    nc += "      _nef_sequence.residue_type\n"
    
    from .data import ONE_TO_THREE_LETTER_CODE
    for i, char in enumerate(sequence):
        res_num = i + 1
        res_name = ONE_TO_THREE_LETTER_CODE.get(char, "UNK")
        nc += f"      A {res_num} {res_name} protein\n"
    nc += "   stop_\n"
    nc += "save_\n\n"
    
    # Helper to write a measurement list
    def write_list(metric_name, nef_name, unit):
        block = f"save_{nef_name}_{int(field_freq_mhz)}MHz\n"
        block += f"   _nef_nmr_measurement_list.sf_category nef_nmr_measurement_list\n"
        block += f"   _nef_nmr_measurement_list.sf_framecode {nef_name}_{int(field_freq_mhz)}MHz\n"
        block += f"   _nef_nmr_measurement_list.experiment_classification {nef_name}\n"
        block += f"   _nef_nmr_measurement_list.measurement_unit {unit}\n"
        block += "\n"
        block += "   loop_\n"
        block += "      _nef_nmr_measurement.index\n"
        block += "      _nef_nmr_measurement.chain_code\n"
        block += "      _nef_nmr_measurement.sequence_code\n"
        block += "      _nef_nmr_measurement.residue_name\n"
        block += "      _nef_nmr_measurement.atom_name\n"
        block += "      _nef_nmr_measurement.value\n"
        block += "      _nef_nmr_measurement.value_uncertainty\n"
        
        count = 0
        sorted_ids = sorted(relaxation_data.keys())
        for rid in sorted_ids:
            # We assume Amide N usually
            # NEF requires defining the atom. Usually N.
            # R1/R2/NOE are 15N relaxation.
            
            val = relaxation_data[rid].get(metric_name)
            if val is None: continue
            
            # Map rid to res_name?
            # We only have sequence string. 'rid' is 1-based index?
            # 'rid' in relaxation.py comes from struct.res_id.
            # Assuming struct.res_id matches 1..N of sequence.
            # Check bounds
            if rid < 1 or rid > len(sequence):
                res_n = "UNK"
            else:
                s_char = sequence[rid-1]
                res_n = ONE_TO_THREE_LETTER_CODE.get(s_char, "UNK")
            
            count += 1
            # Uncertainty dummy 5%
            err = abs(val * 0.05)
            block += f"      {count} A {rid} {res_n} N {val:.4f} {err:.4f}\n"
            
        block += "   stop_\n"
        block += "save_\n\n"
        return block

    # T1 (R1 in s^-1, but NEF usually stores T1 or R1? "R1" classification exists)
    # The classification is "R1", unit "s-1".
    nc += write_list('R1', 'R1', 's-1')
    
    # T2 (R2)
    nc += write_list('R2', 'R2', 's-1')
    
    # NOE (Heteronuclear NOE)
    # Unit: none (dimensionless)
    nc += write_list('NOE', 'Heteronuclear_NOE', 'none')
    
    with open(filename, "w") as f:
        f.write(nc)
    
    logger.info(f"Wrote relaxation data to {filename}")

def write_nef_chemical_shifts(
    filename: str,
    sequence: str,
    shift_data: Dict[str, Dict[int, Dict[str, float]]],
    system_name: str = "synth-pdb-project"
) -> None:
    """
    write a NEF file with Chemical Shift List.
    
    EDUCATIONAL NOTE - NEF Chemical Shift Format:
    =============================================
    The NMR Exchange Format (NEF) standardizes how chemical shifts are stored.
    Unlike older formats (like NMR-STAR 3.1), NEF simplifies the tags.
    
    Key Elements:
    - save_chemical_shift_list: The saveframe container.
    - _nef_chemical_shift_list.shift_reference_type: Usually "DSS" or "TSP" (internal standards).
    - Loop containing:
      - chain_code / sequence_code / residue_name: Identity
      - atom_name: IUPAC name (e.g., CA, HB2)
      - value: The shift in ppm.
    
    Compatibility:
    This format is native to CCPNMR Analysis V3 and supported by CYANA/XPLOR-NIH.
    
    Args:
        filename: Output filename
        sequence: Sequence string
        shift_data: Dict[chain -> res_id -> atom_name -> shift]
    """
    logger.info(f"Writing NEF chemical shifts to {filename}...")
    
    nc = "data_" + system_name + "\n\n"
    # Metadata
    nc += "_nef_nmr_meta_data.nef_format_version 1.1\n"
    nc += f"_nef_nmr_meta_data.creation_date {datetime.datetime.now().isoformat()}\n"
    nc += "_nef_nmr_meta_data.program_name synth-pdb\n\n"
    
    # Write Sequence (Required)
    nc += "save_nef_sequence\n"
    nc += "   _nef_sequence.sf_category nef_sequence\n"
    nc += "   _nef_sequence.sf_framecode nef_sequence\n\n"
    nc += "   loop_\n"
    nc += "      _nef_sequence.chain_code\n"
    nc += "      _nef_sequence.sequence_code\n"
    nc += "      _nef_sequence.residue_name\n"
    nc += "      _nef_sequence.residue_type\n"
    
    from .data import ONE_TO_THREE_LETTER_CODE
    for i, char in enumerate(sequence):
        res_num = i + 1
        res_name = ONE_TO_THREE_LETTER_CODE.get(char, "UNK")
        nc += f"      A {res_num} {res_name} protein\n"
    nc += "   stop_\n"
    nc += "save_\n\n"
    
    # Write Chemical Shifts
    nc += "save_chemical_shift_list\n"
    nc += "   _nef_chemical_shift_list.sf_category nef_chemical_shift_list\n"
    nc += "   _nef_chemical_shift_list.sf_framecode chemical_shift_list\n"
    nc += "   _nef_chemical_shift_list.shift_reference_type DSS\n"
    nc += "   _nef_chemical_shift_list.shift_unit ppm\n\n"
    
    nc += "   loop_\n"
    nc += "      _nef_chemical_shift.index\n"
    nc += "      _nef_chemical_shift.chain_code\n"
    nc += "      _nef_chemical_shift.sequence_code\n"
    nc += "      _nef_chemical_shift.residue_name\n"
    nc += "      _nef_chemical_shift.atom_name\n"
    nc += "      _nef_chemical_shift.value\n"
    nc += "      _nef_chemical_shift.value_uncertainty\n"
    
    idx = 0
    # Process chain A only for now as generator is single chain
    chain_shifts = shift_data.get('A', {})
    
    sorted_res = sorted(chain_shifts.keys())
    for res_id in sorted_res:
        # Determine Residue Name
        if 1 <= res_id <= len(sequence):
            aa_char = sequence[res_id-1]
            res_name = ONE_TO_THREE_LETTER_CODE.get(aa_char, "UNK")
        else:
            res_name = "UNK"
            
        atoms = chain_shifts[res_id]
        # Sort atoms for clean output (N, CA, CB, C, H, HA)
        # Custom sort order:
        order = {"N":1, "H":2, "CA":3, "HA":4, "CB":5, "C":6}
        sorted_atoms = sorted(atoms.keys(), key=lambda x: order.get(x, 99))
        
        for atom in sorted_atoms:
            val = atoms[atom]
            idx += 1
            # Uncertainty dummy 0.1 ppm
            nc += f"      {idx} A {res_id} {res_name} {atom} {val:.3f} 0.100\n"
            
    nc += "   stop_\n"
    nc += "save_\n"
    
    with open(filename, "w") as f:
        f.write(nc)
    logger.info(f"Wrote chemical shifts to {filename}")

def read_nef_restraints(filename: str) -> List[Dict]:
    """
    Read distance restraints from a NEF file.
    
    Parses 'nef_distance_restraint_list' saveframes.
    
    Args:
        filename: Path to NEF file.
        
    Returns:
        List of dictionaries containing parsed restraint data:
        [{
            'chain_1': 'A', 'seq_1': 1, 'atom_1': 'H',
            'chain_2': 'A', 'seq_2': 4, 'atom_2': 'HA',
            'dist': 5.0, ...
        }, ...]
    """
    restraints = []
    
    # State machine for parsing
    in_restraint_saveframe = False
    in_loop = False
    headers = []
    
    # Header indices mapped to keys
    col_map = {}
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Detect saveframe start
            # We relax the check to enter any saveframe, then validation happens by column headers
            if line.startswith('save_'):
                in_restraint_saveframe = True
                continue
                
            # Detect saveframe end
            if line == 'save_' and in_restraint_saveframe:
                in_restraint_saveframe = False
                in_loop = False
                headers = []
                col_map = {}
                continue
                
            if not in_restraint_saveframe:
                continue
                
            # Inside restraint saveframe
            if line == 'loop_':
                in_loop = True
                headers = []
                col_map = {}
                continue
            
            if line == 'stop_':
                in_loop = False
                headers = []
                continue
                
            if in_loop:
                # Reading headers or data
                if line.startswith('_nef_distance_restraint.'):
                    # It's a header column definition
                    col_name = line.split('.')[1]
                    headers.append(col_name)
                    # Map simplified name to index
                    # e.g. chain_code_1 -> index 2
                    col_map[col_name] = len(headers) - 1
                else:
                    # It's a data line
                    parts = line.split()
                    if len(parts) != len(headers):
                        # Skip malformed lines or comment handling
                        continue
                        
                    # Extract data using the map
                    try:
                        r = {
                            'chain_1': parts[col_map['chain_code_1']],
                            'seq_1': int(parts[col_map['sequence_code_1']]),
                            'res_1': parts[col_map['residue_name_1']],
                            'atom_1': parts[col_map['atom_name_1']],
                            
                            'chain_2': parts[col_map['chain_code_2']],
                            'seq_2': int(parts[col_map['sequence_code_2']]),
                            'res_2': parts[col_map['residue_name_2']],
                            'atom_2': parts[col_map['atom_name_2']],
                            
                            'dist': float(parts[col_map['target_value']])
                        }
                        restraints.append(r)
                    except (KeyError, ValueError) as e:
                        # Log parsing error but continue
                        # logger.warning(f"Skipping malformed restraint line: {line} - Error: {e}")
                        pass
                        
    except FileNotFoundError:
        logger.error(f"NEF file not found: {filename}")
        return []
        
    logger.info(f"Read {len(restraints)} restraints from {filename}")
    return restraints
