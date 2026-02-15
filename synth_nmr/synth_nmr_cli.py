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
import os
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from synth_nmr.rdc import calculate_rdcs
from synth_nmr.chemical_shifts import predict_chemical_shifts
from synth_nmr.j_coupling import calculate_hn_ha_coupling

structure = None

def main():
    """The main function for the synth-nmr CLI."""
    if len(sys.argv) > 1:
        # Non-interactive mode
        process_commands(sys.argv[1:])
    else:
        # Interactive mode
        interactive_mode()

def process_commands(args):
    """Process a list of commands."""
    global structure
    i = 0
    while i < len(args):
        command = args[i].lower()
        if command == "read" and i + 2 < len(args) and args[i+1].lower() == "pdb":
            filename = args[i+2]
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
        elif command == "calculate" and i + 1 < len(args) and args[i+1].lower() == "rdc":
            if structure is None:
                print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                i += 2
                continue
            Da = 10.0
            R = 0.5
            if i + 2 < len(args) and not args[i+2].isalpha():
                try:
                    Da = float(args[i+2])
                    i += 1
                except ValueError:
                    print("Error: Invalid value for Da. Must be a float.")
                    i += 1
                    continue
            if i + 2 < len(args) and not args[i+2].isalpha():
                try:
                    R = float(args[i+2])
                    i += 1
                except ValueError:
                    print("Error: Invalid value for R. Must be a float.")
                    i += 1
                    continue
            rdcs = calculate_rdcs(structure, Da=Da, R=R)
            for res_id, rdc in rdcs.items():
                print(f"ResID: {res_id}, RDC: {rdc}")
            i += 2
        elif command == "predict" and i + 1 < len(args) and args[i+1].lower() == "shifts":
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
        elif command == "calculate" and i + 1 < len(args) and args[i+1].lower() == "j-coupling":
            if structure is None:
                print("Error: No PDB file loaded. Use 'read pdb <filename>' first.")
                i += 2
                continue
            j_couplings = calculate_hn_ha_coupling(structure)
            for chain_id, res_couplings in j_couplings.items():
                for res_id, coupling in res_couplings.items():
                    print(f"Chain: {chain_id}, ResID: {res_id}, J-coupling: {coupling}")
            i += 2
        else:
            print(f"Error: Unknown command: {command}")
            i += 1

def interactive_mode():
    """Run the CLI in interactive mode."""
    global structure
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
                j_couplings = calculate_hn_ha_coupling(structure)
                for chain_id, res_couplings in j_couplings.items():
                    for res_id, coupling in res_couplings.items():
                        print(f"Chain: {chain_id}, ResID: {res_id}, J-coupling: {coupling}")
            else:
                print(f"Error: Unknown command: {command}")
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"An error occurred: {e}")



if __name__ == "__main__":
    main()
