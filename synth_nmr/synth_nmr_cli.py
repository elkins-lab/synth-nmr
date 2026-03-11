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
from typing import Dict, List, Optional, Tuple
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from synth_nmr.rdc import calculate_rdcs
from synth_nmr.chemical_shifts import predict_chemical_shifts
from synth_nmr.trajectory import (
    TrajectoryEnsemble,
    load_trajectory,
    ensemble_average_shifts,
    ensemble_average_noes,
    ensemble_average_rdcs,
    ensemble_average_j_couplings,
    compute_s2_from_trajectory,
)
from synth_nmr.nmr import calculate_synthetic_noes
from synth_nmr.j_coupling import (
    calculate_hn_ha_coupling,
    calculate_ha_hb_coupling,
    calculate_c_cg_coupling,
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
    global structure, ensemble
    i = 0
    while i < len(args):
        command = args[i].lower()
        if command == "read" and i + 2 < len(args) and args[i + 1].lower() == "pdb":
            filename = args[i + 2]
            try:
                pdb_file = pdb.PDBFile.read(filename)
                structure = pdb_file.get_structure()
                if isinstance(structure, struc.AtomArrayStack):
                    structure = structure[0]
                print(f"Read PDB file: {filename}")
            except FileNotFoundError:
                print(f"Error: File not found: {filename}")
            except Exception as e:
                print(f"Error: Failed to read PDB file: {e}")
            i += 3

        elif command == "load" and i + 1 < len(args) and args[i + 1].lower() == "trajectory":
            # Collect all PDB file paths that follow the 'load trajectory' keyword
            # Usage: load trajectory frame1.pdb frame2.pdb [...]
            frame_paths = []
            j = i + 2
            while j < len(args) and args[j].lower() not in (
                "load",
                "ensemble",
                "read",
                "calculate",
                "predict",
            ):
                frame_paths.append(args[j])
                j += 1
            if not frame_paths:
                print("Error: Provide at least one PDB file path after 'load trajectory'.")
                i = j
                continue
            frames: List[struc.AtomArray] = []
            for path in frame_paths:
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
            i = j

        elif command == "ensemble" and i + 1 < len(args):
            sub = args[i + 1].lower()
            if ensemble is None:
                print("Error: No trajectory loaded. Use 'load trajectory <pdb1> ...' first.")
                i += 2
                continue

            if sub == "shifts":
                # Ensemble-average chemical shifts across all frames.
                # predict_chemical_shifts returns {method: {res_id: {atom: shift}}},
                # so we merge the inner dicts to get the flat {res_id: {atom: shift}}
                # format that ensemble_average_shifts expects.
                per_frame: List[Dict[int, Dict[str, float]]] = []
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
                i += 2

            elif sub == "noes":
                cutoff = 5.0
                if i + 2 < len(args):
                    try:
                        cutoff = float(args[i + 2])
                        i += 1
                    except ValueError:
                        pass
                per_frame_n: List[Dict[Tuple[int, int], float]] = []
                for frame in ensemble:
                    try:
                        noe_dict = calculate_synthetic_noes(frame, cutoff=cutoff)
                        flat: Dict[Tuple[int, int], float] = {}
                        for ri, peers in noe_dict.items():  # type: ignore[attr-defined]
                            for rj, dist in peers.items():
                                flat[(ri, rj)] = dist
                        per_frame_n.append(flat)
                    except Exception as e:
                        print(f"Warning: NOE calculation failed for a frame: {e}")
                if per_frame_n:
                    avg_noes = ensemble_average_noes(per_frame_n)
                    for (ri, rj), r_eff in sorted(avg_noes.items()):
                        print(f"Res {ri:4d} — Res {rj:4d}  r_eff = {r_eff:.3f} Å")
                i += 2

            elif sub == "rdcs":
                Da = 10.0
                R = 0.5
                if i + 2 < len(args):
                    try:
                        Da = float(args[i + 2])
                        i += 1
                    except ValueError:
                        pass
                if i + 2 < len(args):
                    try:
                        R = float(args[i + 2])
                        i += 1
                    except ValueError:
                        pass
                per_frame_r = [calculate_rdcs(f, Da=Da, R=R) for f in ensemble]
                avg_rdcs = ensemble_average_rdcs(per_frame_r)
                for res_id, rdc in sorted(avg_rdcs.items()):
                    print(f"ResID {res_id:4d}  D_NH = {rdc:.3f} Hz")
                i += 2

            elif sub == "j-coupling":
                per_frame_j = [calculate_hn_ha_coupling(f) for f in ensemble]
                avg_j = ensemble_average_j_couplings(per_frame_j)
                for chain_id, res_couplings in sorted(avg_j.items()):
                    for res_id, j_val in sorted(res_couplings.items()):
                        print(f"Chain {chain_id} ResID {res_id:4d}  3J_HNHa = {j_val:.3f} Hz")
                i += 2

            elif sub == "s2":
                s2_map = compute_s2_from_trajectory(ensemble)
                for res_id, s2_val in sorted(s2_map.items()):
                    print(f"ResID {res_id:4d}  S² = {s2_val:.4f}")
                i += 2

            else:
                print(f"Error: Unknown ensemble subcommand: {sub}")
                i += 2

        elif command == "calculate" and i + 1 < len(args) and args[i + 1].lower() == "rdc":
            if structure is None:
                print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                i += 2
                continue
            Da = 10.0
            R = 0.5
            if i + 2 < len(args) and not args[i + 2].isalpha():
                try:
                    Da = float(args[i + 2])
                    i += 1
                except ValueError:
                    print("Error: Invalid value for Da. Must be a float.")
                    i += 1
                    continue
            if i + 2 < len(args) and not args[i + 2].isalpha():
                try:
                    R = float(args[i + 2])
                    i += 1
                except ValueError:
                    print("Error: Invalid value for R. Must be a float.")
                    i += 1
                    continue
            rdcs = calculate_rdcs(structure, Da=Da, R=R)
            for res_id, rdc in rdcs.items():
                print(f"ResID: {res_id}, RDC: {rdc}")
            i += 2
        elif command == "predict" and i + 1 < len(args) and args[i + 1].lower() == "shifts":
            if structure is None:
                print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                i += 2
                continue
            shifts = predict_chemical_shifts(structure)
            for chain_id, res_shifts in shifts.items():
                for res_id, atom_shifts in res_shifts.items():
                    print(f"Chain: {chain_id}, ResID: {res_id}")
                    for atom_name, shift_val in atom_shifts.items():
                        print(f"  {atom_name}: {shift_val}")
            i += 2
        elif command == "calculate" and i + 1 < len(args) and args[i + 1].lower() == "j-coupling":
            if structure is None:
                print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                i += 2
                continue

            # Backbone
            j_couplings = calculate_hn_ha_coupling(structure)
            # Side-chains
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
            i += 2
        else:
            print(f"Error: Unknown command: {command}")
            i += 1


def interactive_mode() -> None:
    """Run the CLI in interactive mode."""
    global structure, ensemble
    print("Welcome to the synth-nmr CLI!")
    print("Enter 'help' for a list of commands.")
    while True:
        try:
            sys.stdout.write("SynthNMR> ")
            sys.stdout.flush()
            line = sys.stdin.readline().strip()
            if not line:
                continue

            parts = line.split()
            command = parts[0].lower()

            if command == "exit":
                break
            elif command == "help":
                print("Commands:")
                print("  read pdb <filename>")
                print("  load trajectory <pdb1> [pdb2 ...]")
                print("  ensemble shifts")
                print("  ensemble noes [cutoff_angstrom]")
                print("  ensemble rdcs [Da] [R]")
                print("  ensemble s2")
                print("  calculate rdc [Da] [R]")
                print("  predict shifts")
                print("  calculate j-coupling")
                print("  exit")
            elif command == "read" and len(parts) > 1 and parts[1].lower() == "pdb":
                if len(parts) < 3:
                    print("Usage: read pdb <filename>")
                    continue
                filename = " ".join(parts[2:])
                try:
                    pdb_file = pdb.PDBFile.read(filename)
                    structure = pdb_file.get_structure()
                    if isinstance(structure, struc.AtomArrayStack):
                        structure = structure[0]
                    print(f"Read PDB file: {filename}")
                except FileNotFoundError:
                    print(f"Error: File not found: {filename}")
                except Exception as e:
                    print(f"Error: Failed to read PDB file: {e}")

            elif command == "load" and len(parts) > 1 and parts[1].lower() == "trajectory":
                frame_paths = parts[2:]
                if not frame_paths:
                    print("Usage: load trajectory <pdb1> [pdb2 ...]")
                    continue
                frames: List[struc.AtomArray] = []
                for path in frame_paths:
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
                    continue

                if sub == "shifts":
                    per_frame2: List[Dict[int, Dict[str, float]]] = []
                    for frame in ensemble:
                        try:
                            raw = predict_chemical_shifts(frame)
                            merged2: Dict[int, Dict[str, float]] = {}
                            for method_dict in raw.values():
                                for res_id, atoms in method_dict.items():
                                    if res_id not in merged2:
                                        merged2[res_id] = {}
                                    merged2[res_id].update(atoms)
                            per_frame2.append(merged2)
                        except Exception as e:
                            print(f"Warning: shift prediction failed for a frame: {e}")
                    if per_frame2:
                        avg = ensemble_average_shifts(per_frame2)
                        for res_id, nucleus_dict in sorted(avg.items()):
                            for atom_name, shift in sorted(nucleus_dict.items()):
                                print(f"ResID {res_id:4d}  {atom_name:<4s}  {shift:.3f} ppm")

                elif sub == "noes":
                    cutoff = float(parts[2]) if len(parts) > 2 else 5.0
                    per_frame_n2: List[Dict[Tuple[int, int], float]] = []
                    for frame in ensemble:
                        try:
                            noe_dict = calculate_synthetic_noes(frame, cutoff=cutoff)
                            flat2: Dict[Tuple[int, int], float] = {}
                            for ri, peers in noe_dict.items():  # type: ignore[attr-defined]
                                for rj, dist in peers.items():
                                    flat2[(ri, rj)] = dist
                            per_frame_n2.append(flat2)
                        except Exception as e:
                            print(f"Warning: NOE calculation failed for a frame: {e}")
                    if per_frame_n2:
                        avg_noes = ensemble_average_noes(per_frame_n2)
                        for (ri, rj), r_eff in sorted(avg_noes.items()):
                            print(f"Res {ri:4d} — Res {rj:4d}  r_eff = {r_eff:.3f} Å")

                elif sub == "rdcs":
                    Da = float(parts[2]) if len(parts) > 2 else 10.0
                    R = float(parts[3]) if len(parts) > 3 else 0.5
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
                    continue
                Da = 10.0
                R = 0.5
                if len(parts) > 2:
                    try:
                        Da = float(parts[2])
                    except ValueError:
                        print("Error: Invalid value for Da. Must be a float.")
                        continue
                if len(parts) > 3:
                    try:
                        R = float(parts[3])
                    except ValueError:
                        print("Error: Invalid value for R. Must be a float.")
                        continue
                rdcs = calculate_rdcs(structure, Da=Da, R=R)
                for res_id, rdc in rdcs.items():
                    print(f"ResID: {res_id}, RDC: {rdc}")
            elif command == "predict" and len(parts) > 1 and parts[1].lower() == "shifts":
                if structure is None:
                    print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                    continue
                shifts = predict_chemical_shifts(structure)
                for chain_id, res_shifts in shifts.items():
                    for res_id, atom_shifts in res_shifts.items():
                        print(f"Chain: {chain_id}, ResID: {res_id}")
                        for atom_name, shift_val in atom_shifts.items():
                            print(f"  {atom_name}: {shift_val}")
            elif command == "calculate" and len(parts) > 1 and parts[1].lower() == "j-coupling":
                if structure is None:
                    print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                    continue
                # Backbone
                j_couplings = calculate_hn_ha_coupling(structure)
                # Side-chains
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
            else:
                print(f"Error: Unknown command: {command}")
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
