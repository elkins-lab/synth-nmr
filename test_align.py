import biotite.structure.io.pdb as pdb
from synth_nmr.chemical_shifts import predict_empirical_shifts, ShiftX2Predictor

s = pdb.PDBFile.read('/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/1UBQ.pdb').get_structure(model=1)
synth_preds = predict_empirical_shifts(s)
shiftx_preds = ShiftX2Predictor(executable='/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/shiftx2.py').predict(s)

aligned_data = {"CA": ([], [])}

for chain_id, res_dict in synth_preds.items():
    print(f"Synth chain: {repr(chain_id)}")
    if chain_id not in shiftx_preds:
        print(f"  -> chain {repr(chain_id)} missing from shiftx_preds! Available: {list(shiftx_preds.keys())}")
        continue
    print(f"  -> Chain {repr(chain_id)} found.")
        
    for res_id, synth_atom_shifts in res_dict.items():
        if res_id not in shiftx_preds[chain_id]:
            print(f"  -> res_id {repr(res_id)} missing from shiftx_preds[chain]! Available: {list(shiftx_preds[chain_id].keys())[:5]}")
            continue
            
        shiftx_atom_shifts = shiftx_preds[chain_id][res_id]
        
        for atom_type in aligned_data.keys():
            if atom_type in synth_atom_shifts and atom_type in shiftx_atom_shifts:
                aligned_data[atom_type][0].append(synth_atom_shifts[atom_type])
                aligned_data[atom_type][1].append(shiftx_atom_shifts[atom_type])

print(f"Aligned CA elements: {len(aligned_data['CA'][0])}")
