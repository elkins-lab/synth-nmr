#!/usr/bin/env python

#
# Copyright (C) 2024 - All Rights Reserved
#
# This file is part of the synth-nmr project.
#
# synth-nmr is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# synth-nmr is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with synth-nmr. If not, see <https://www.gnu.org/licenses/>.

"""A command-line interface for synth-nmr."""

import sys
from typing import Dict, List, Optional

import biotite.structure as struc
import biotite.structure.io.pdb as pdb

from synth_nmr.chemical_shifts import predict_chemical_shifts
from synth_nmr.data_pipeline import download_bmrb_file, parse_bmrb_restraints, parse_bmrb_shifts
from synth_nmr.j_coupling import (
    calculate_c_cg_coupling,
    calculate_ha_hb_coupling,
    calculate_hn_ha_coupling,
)
from synth_nmr.nmr import calculate_synthetic_noes
from synth_nmr.rdc import calculate_rdcs
from synth_nmr.structure_utils import calculate_c_beta_deviations, get_residue_info
from synth_nmr.trajectory import (
    TrajectoryEnsemble,
    compute_s2_from_trajectory,
    ensemble_average_j_couplings,
    ensemble_average_noes,
    ensemble_average_rdcs,
    ensemble_average_shifts,
    load_trajectory,
)
from synth_nmr.validation import (
    calculate_cs_r_factor,
    calculate_dp_score,
    calculate_rpf_scores,
    compare_chemical_shifts,
)

structure: Optional[struc.AtomArray] = None
ensemble: Optional[TrajectoryEnsemble] = None


def main() -> None:
    """The main function for the synth-nmr CLI."""
    if len(sys.argv) > 1:
        # Non-interactive mode
        process_commands(sys.argv[1:])
    else:
        # Interactive mode
        interactive_mode()


def process_commands(args: List[str]) -> None:
    """Process a list of commands."""
    # Group arguments into commands
    # This is a bit tricky because some commands have variable number of arguments
    # Let's use a simple heuristic: split by known top-level commands
    commands = []
    current_cmd: List[str] = []
    top_level = {
        "read",
        "load",
        "ensemble",
        "calculate",
        "predict",
        "validate",
        "export",
        "help",
        "exit",
    }

    for arg in args:
        if arg.lower() in top_level and current_cmd:
            commands.append(" ".join(current_cmd))
            current_cmd = [arg]
        else:
            current_cmd.append(arg)
    if current_cmd:
        commands.append(" ".join(current_cmd))

    for cmd_line in commands:
        handle_interactive_command(cmd_line)


def handle_interactive_command(line: str) -> bool:
    """
    Handle a single line of input in interactive mode.

    Args:
        line: The command line string.

    Returns:
        True if the session should continue, False if 'exit' was received.
    """
    global structure, ensemble
    parts = line.split()
    if not parts:
        return True
    command = parts[0].lower()

    try:
        if command == "exit":
            return False
        elif command == "help":
            print("Commands:")
            print("  read pdb <filename>          Read a PDB file.")
            print("  load trajectory <pdb1> ...   Load a trajectory ensemble.")
            print("  ensemble shifts              Ensemble-average chemical shifts.")
            print("  ensemble noes [cutoff]       Ensemble-average NOE distances.")
            print("  ensemble rdcs [Da] [R]       Ensemble-average RDCs.")
            print("  ensemble j-coupling          Ensemble-average J-couplings.")
            print("  ensemble s2                  Compute S² order parameters.")
            print("  calculate rdc [Da] [R]       Calculate RDCs for single structure.")
            print("  predict shifts               Predict chemical shifts for single structure.")
            print("  calculate j-coupling         Calculate J-couplings for single structure.")
            print("  validate shifts <bmrb_id>    Validate shifts against BMRB.")
            print("  validate noes <bmrb_id>      Validate NOEs against BMRB (RPF scores).")
            print("  validate structure           Check C-beta deviations.")
            print("  export nef <filename>        Export structure and data to NEF.")
            print("  export shifts <filename>     Export chemical shifts to CSV.")
            print("  exit                         Exit the CLI.")
        elif command == "validate" and len(parts) > 1:
            sub = parts[1].lower()
            if structure is None:
                print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                return True

            if sub == "shifts" and len(parts) > 2:
                bmrb_id = parts[2]
                bmrb_path = download_bmrb_file(int(bmrb_id))
                if bmrb_path:
                    exp_shifts = parse_bmrb_shifts(bmrb_path)
                    pred_shifts = predict_chemical_shifts(structure)
                    # Align exp_shifts (residue-based) to predicted (chain-based)
                    # For simplicity, assume chain A
                    ref_dict = {"A": exp_shifts}
                    stats = compare_chemical_shifts(pred_shifts, ref_dict)
                    r_cs = calculate_cs_r_factor(pred_shifts, ref_dict, atom="CA")

                    print(f"\nValidation against BMRB {bmrb_id}:")
                    for atom, m in stats.items():
                        print(f"  {atom:4s}: RMSE={m['rmse']:.3f}, Pearson R={m['pearson']:.3f}")
                    print(f"  Chemical Shift R-factor (CA): {r_cs:.4f}")

            elif sub == "noes" and len(parts) > 2:
                bmrb_id = parts[2]
                bmrb_path = download_bmrb_file(int(bmrb_id))
                if bmrb_path:
                    exp_noes = parse_bmrb_restraints(bmrb_path)
                    pred_noes = calculate_synthetic_noes(structure, cutoff=5.0)
                    rpf = calculate_rpf_scores(pred_noes, exp_noes)
                    dp = calculate_dp_score(rpf)

                    print(f"\nNOE Validation (RPF) against BMRB {bmrb_id}:")
                    print(f"  Recall:    {rpf['recall']:.3f}")
                    print(f"  Precision: {rpf['precision']:.3f}")
                    print(f"  F-measure: {rpf['f_measure']:.3f}")
                    print(f"  DP-score:  {dp:.3f}")

            elif sub == "structure":
                deviations = calculate_c_beta_deviations(structure)
                outliers = {rid: dev for rid, dev in deviations.items() if dev > 0.25}
                print("\nStructural Validation (C-beta deviations):")
                print(f"  Total residues checked: {len(deviations)}")
                print(f"  Outliers (> 0.25 Å):    {len(outliers)}")
                for rid, dev in sorted(outliers.items()):
                    print(f"    ResID {rid:4d}: {dev:.3f} Å")
            else:
                print(f"Error: Unknown validate subcommand: {sub}")

        elif command == "read" and len(parts) > 2 and parts[1].lower() == "pdb":
            path = parts[2]
            try:
                pdb_file = pdb.PDBFile.read(path)
                structure = pdb_file.get_structure()
                if isinstance(structure, struc.AtomArrayStack):
                    structure = structure[0]
                print(f"Read PDB file: {path}")
            except Exception as e:
                print(f"Error: Could not read '{path}': {e}")
        elif command == "load" and len(parts) > 2 and parts[1].lower() == "trajectory":
            paths = parts[2:]
            frames = []
            for path in paths:
                try:
                    pdb_file = pdb.PDBFile.read(path)
                    frame = pdb_file.get_structure()
                    if isinstance(frame, struc.AtomArrayStack):
                        frame = frame[0]
                    frames.append(frame)
                except Exception as e:
                    print(f"Warning: Could not read '{path}': {e}")
            if frames:
                ensemble = load_trajectory(frames)
                print(f"Loaded trajectory ensemble with {len(ensemble)} frames.")
            else:
                print("Error: No frames could be loaded.")
        elif command == "ensemble" and len(parts) > 1:
            sub = parts[1].lower()
            if ensemble is None:
                print("Error: No trajectory loaded. Use 'load trajectory <pdb1> ...' first.")
                return True

            if sub == "shifts":
                per_frame = []
                for frame in ensemble:
                    try:
                        raw = predict_chemical_shifts(frame)
                        merged: Dict[int, Dict[str, float]] = {}
                        for method_dict in raw.values():
                            for res_id, atoms in method_dict.items():
                                if res_id not in merged:
                                    merged[res_id] = {}
                                merged[res_id].update(atoms)
                        per_frame.append(merged)
                    except Exception as e:
                        print(f"Warning: shift prediction failed for a frame: {e}")
                if per_frame:
                    avg = ensemble_average_shifts(per_frame)
                    for res_id, nucleus_dict in sorted(avg.items()):
                        for atom_name, shift in sorted(nucleus_dict.items()):
                            print(f"ResID {res_id:4d}  {atom_name:<4s}  {shift:.3f} ppm")

            elif sub == "noes":
                cutoff = 5.0
                if len(parts) > 2:
                    try:
                        cutoff = float(parts[2])
                    except ValueError:
                        pass
                per_frame_n = []
                for frame in ensemble:
                    try:
                        noe_list = calculate_synthetic_noes(frame, cutoff=cutoff)
                        flat = {}
                        for restraint in noe_list:
                            ri = int(restraint["index_1"])
                            rj = int(restraint["index_2"])
                            dist = float(restraint["distance"])
                            flat[(ri, rj)] = dist
                        per_frame_n.append(flat)
                    except Exception as e:
                        print(f"Warning: NOE calculation failed for a frame: {e}")
                if per_frame_n:
                    avg_noes = ensemble_average_noes(per_frame_n)
                    for (ri, rj), r_eff in sorted(avg_noes.items()):
                        print(f"Res {ri:4d} — Res {rj:4d}  r_eff = {r_eff:.3f} Å")

            elif sub == "rdcs":
                Da = 10.0
                R = 0.5
                if len(parts) > 2:
                    try:
                        Da = float(parts[2])
                    except ValueError:
                        pass
                if len(parts) > 3:
                    try:
                        R = float(parts[3])
                    except ValueError:
                        pass
                per_frame_r = [calculate_rdcs(f, Da=Da, R=R) for f in ensemble]
                avg_rdcs = ensemble_average_rdcs(per_frame_r)
                for res_id, rdc in sorted(avg_rdcs.items()):
                    print(f"ResID {res_id:4d}  D_NH = {rdc:.3f} Hz")

            elif sub == "j-coupling":
                per_frame_j = [calculate_hn_ha_coupling(f) for f in ensemble]
                avg_j = ensemble_average_j_couplings(per_frame_j)
                for chain_id, res_couplings in sorted(avg_j.items()):
                    for res_id, j_val in sorted(res_couplings.items()):
                        print(f"Chain {chain_id} ResID {res_id:4d}  3J_HNHa = {j_val:.3f} Hz")

            elif sub == "s2":
                s2_map = compute_s2_from_trajectory(ensemble)
                for res_id, s2_val in sorted(s2_map.items()):
                    print(f"ResID {res_id:4d}  S² = {s2_val:.4f}")
            else:
                print(f"Error: Unknown ensemble subcommand: {sub}")

        elif command == "calculate" and len(parts) > 1 and parts[1].lower() == "rdc":
            if structure is None:
                print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                return True
            Da = 10.0
            R = 0.5
            if len(parts) > 2:
                try:
                    Da = float(parts[2])
                except ValueError:
                    print("Error: Invalid value for Da. Must be a float.")
                    return True
            if len(parts) > 3:
                try:
                    R = float(parts[3])
                except ValueError:
                    print("Error: Invalid value for R. Must be a float.")
                    return True
            rdcs = calculate_rdcs(structure, Da=Da, R=R)
            for res_id, rdc in rdcs.items():
                print(f"ResID: {res_id}, RDC: {rdc}")
        elif command == "predict" and len(parts) > 1 and parts[1].lower() == "shifts":
            if structure is None:
                print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                return True
            shifts = predict_chemical_shifts(structure)
            for chain_id, res_shifts in shifts.items():
                for res_id, atom_shifts in res_shifts.items():
                    print(f"Chain: {chain_id}, ResID: {res_id}")
                    for atom_name, shift_val in atom_shifts.items():
                        print(f"  {atom_name}: {shift_val}")
        elif command == "calculate" and len(parts) > 1 and parts[1].lower() == "j-coupling":
            if structure is None:
                print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                return True
            j_couplings = calculate_hn_ha_coupling(structure)
            j_hahb = calculate_ha_hb_coupling(structure)
            j_ccg = calculate_c_cg_coupling(structure)
            for chain_id, res_couplings in sorted(j_couplings.items()):
                for res_id, coupling in sorted(res_couplings.items()):
                    print(f"Chain {chain_id} ResID {res_id:4d}  3J_HNHa = {coupling:.3f} Hz")
                    if chain_id in j_hahb and res_id in j_hahb[chain_id]:
                        print(
                            f"Chain {chain_id} ResID {res_id:4d}  3J_HaHb = {j_hahb[chain_id][res_id]:.3f} Hz"
                        )
                    if chain_id in j_ccg and res_id in j_ccg[chain_id]:
                        print(
                            f"Chain {chain_id} ResID {res_id:4d}  3J_C'Cg = {j_ccg[chain_id][res_id]:.3f} Hz"
                        )

        elif command == "export" and len(parts) > 1:
            if structure is None:
                print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                return True
            sub = parts[1].lower()
            if sub == "nef" and len(parts) > 2:
                from synth_nmr.nef_io import write_nef_file

                filename = parts[2]

                # Extract sequence and restraints
                _, _, res_names, _ = get_residue_info(structure)
                map3to1 = {
                    "ALA": "A",
                    "CYS": "C",
                    "ASP": "D",
                    "GLU": "E",
                    "PHE": "F",
                    "GLY": "G",
                    "HIS": "H",
                    "ILE": "I",
                    "LYS": "K",
                    "LEU": "L",
                    "MET": "M",
                    "ASN": "N",
                    "PRO": "P",
                    "GLN": "Q",
                    "ARG": "R",
                    "SER": "S",
                    "THR": "T",
                    "VAL": "V",
                    "TRP": "W",
                    "TYR": "Y",
                }
                seq = "".join([map3to1.get(n, "X") for n in res_names])

                noes = calculate_synthetic_noes(structure, cutoff=5.0)

                write_nef_file(filename, seq, noes)
                print(f"Exported data to {filename}")
            elif sub == "shifts" and len(parts) > 2:
                filename = parts[2]
                shifts = predict_chemical_shifts(structure)
                with open(filename, "w") as f:
                    f.write("Chain,ResID,Atom,Shift\n")
                    for cid, res_shifts in shifts.items():
                        for rid, atoms in res_shifts.items():
                            for atom, val in atoms.items():
                                f.write(f"{cid},{rid},{atom},{val:.3f}\n")
                print(f"Exported chemical shifts to {filename}")
            else:
                print(f"Error: Unknown export subcommand: {sub}")

        else:
            print(f"Error: Unknown command: {command}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return True


def interactive_mode() -> None:
    """Run the CLI in interactive mode."""
    print("Welcome to the synth-nmr CLI!")
    print("Enter 'help' for a list of commands.")
    while True:
        try:
            sys.stdout.write("SynthNMR> ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if not handle_interactive_command(line):
                break
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
