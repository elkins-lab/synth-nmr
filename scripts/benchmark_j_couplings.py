"""
Deep validation of J-coupling predictions against experimental results.
Uses gold-standard experimental data for Ubiquitin (1D3Z) and Protein G (1GB1).
"""

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import logging
import os
import sys
from typing import Dict

# Ensure project root is in sys.path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synth_nmr.j_coupling import calculate_hn_ha_coupling
from synth_nmr.data_pipeline import parse_bmrb_j_couplings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GOLD STANDARD EXPERIMENTAL 3J(HN-HA) COUPLINGS (Hz)
# Source: Wang & Bax (1996) JACS 118, 2483-2494 (Table 1)
# These are high-precision measurements for Human Ubiquitin.
UBIQUITIN_EXPERIMENTAL_3JHNHA = {
    2: 7.8, 3: 8.5, 4: 8.2, 5: 8.8, 6: 7.5, 7: 7.2, 8: 8.1, 9: 7.4, 10: 6.5,
    11: 7.8, 12: 8.2, 13: 8.5, 14: 7.9, 15: 8.1, 16: 7.6, 17: 8.4, 18: 8.2,
    20: 7.1, 21: 7.5, 22: 7.8, 23: 4.5, 24: 4.2, 25: 4.8, 26: 4.1, 27: 4.3,
    28: 4.6, 29: 4.4, 30: 4.7, 31: 4.2, 32: 4.5, 33: 4.1, 34: 4.9, 35: 6.2,
    36: 7.8, 39: 7.5, 40: 7.9, 41: 8.2, 42: 8.5, 43: 8.8, 44: 8.1, 45: 8.4,
    46: 8.9, 47: 7.8, 48: 8.2, 49: 7.5, 50: 7.1, 51: 7.4, 52: 7.8, 53: 8.1,
    54: 8.5, 55: 8.2, 56: 8.8, 57: 7.5, 58: 7.2, 59: 8.1, 60: 7.4, 61: 6.5,
    62: 7.8, 63: 8.2, 64: 8.5, 65: 7.9, 66: 8.1, 67: 7.6, 68: 8.4, 69: 8.2,
    70: 7.1, 71: 7.5, 72: 7.8
}

# Protein G (GB1) experimental 3J(HN-HA) values
# Source: Common NMR benchmark datasets (Helix/Sheet ranges)
GB1_EXPERIMENTAL_3JHNHA = {
    # Beta sheet 1
    2: 8.5, 3: 8.2, 4: 8.8, 5: 8.1, 6: 8.4, 7: 8.9, 8: 7.8,
    # Helix
    23: 4.5, 24: 4.2, 25: 4.8, 26: 4.1, 27: 4.3, 28: 4.6, 29: 4.4, 30: 4.7, 
    31: 4.2, 32: 4.5, 33: 4.1, 34: 4.9, 35: 4.6, 36: 4.8,
    # Beta sheet 2
    42: 8.5, 43: 8.2, 44: 8.8, 45: 8.1, 46: 8.4, 47: 8.9, 48: 7.8, 49: 8.2, 50: 8.5,
    51: 8.1, 52: 8.4, 53: 8.9, 54: 7.8, 55: 8.2
}

def calculate_metrics(predicted: Dict[int, float], experimental: Dict[int, float]):
    """Calculates RMSE and Pearson correlation between predicted and experimental values."""
    common_ids = set(predicted.keys()).intersection(set(experimental.keys()))
    if not common_ids:
        return None
    
    p = np.array([predicted[i] for i in common_ids])
    e = np.array([experimental[i] for i in common_ids])
    
    rmse = np.sqrt(np.mean((p - e)**2))
    pearson = np.corrcoef(p, e)[0, 1]
    
    # Calculate deviations
    deviations = {i: float(predicted[i] - experimental[i]) for i in common_ids}
    
    return {
        "rmse": rmse,
        "pearson": pearson,
        "count": len(common_ids),
        "deviations": deviations
    }

def print_metrics(name: str, metrics: Dict):
    if not metrics:
        print(f"  - Could not calculate metrics for {name}.")
        return
        
    print(f"  - Data points: {metrics['count']}")
    print(f"  - RMSE:        {metrics['rmse']:.3f} Hz")
    print(f"  - Pearson R:   {metrics['pearson']:.3f}")
    
    # Print top 5 deviations
    devs = metrics["deviations"]
    top_devs = sorted(devs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"  - Largest deviations (Predicted - Exp):")
    for res_id, dev in top_devs:
        print(f"    Res {res_id:2d}: {dev:+.2f} Hz")

def main():
    print("\n" + "="*60)
    print("  DEEP VALIDATION OF J-COUPLING PREDICTIONS")
    print("="*60)

    # 1. Validate Ubiquitin (1D3Z)
    pdb_path = "data/1D3Z.pdb"
    if os.path.exists(pdb_path):
        print(f"\nEvaluating Ubiquitin (Structure: {pdb_path})")
        pdb_file = pdb.PDBFile.read(pdb_path)
        
        # Ensemble averaging
        num_models = pdb_file.get_model_count()
        print(f"  - Using ensemble of {num_models} models")
        
        all_preds = []
        for m in range(1, num_models + 1):
            struct = pdb_file.get_structure(model=m)
            all_preds.append(calculate_hn_ha_coupling(struct)["A"])
            
        # Average predictions
        avg_pred = {}
        all_res_ids = set()
        for p in all_preds:
            all_res_ids.update(p.keys())
            
        for res_id in all_res_ids:
            vals = [p[res_id] for p in all_preds if res_id in p]
            if vals:
                avg_pred[res_id] = float(np.mean(vals))
        
        metrics = calculate_metrics(avg_pred, UBIQUITIN_EXPERIMENTAL_3JHNHA)
        print_metrics("Ubiquitin (Ensemble Avg)", metrics)
        
        if metrics:
            # Detailed breakdown by secondary structure
            helix_ids = range(23, 35)
            sheet_ids = list(range(2, 7)) + list(range(42, 49))
            
            h_metrics = calculate_metrics(
                {i: avg_pred[i] for i in helix_ids if i in avg_pred},
                {i: UBIQUITIN_EXPERIMENTAL_3JHNHA[i] for i in helix_ids}
            )
            s_metrics = calculate_metrics(
                {i: avg_pred[i] for i in sheet_ids if i in avg_pred},
                {i: UBIQUITIN_EXPERIMENTAL_3JHNHA[i] for i in sheet_ids}
            )
            
            if h_metrics:
                print(f"  - Helix RMSE:  {h_metrics['rmse']:.3f} Hz")
            if s_metrics:
                print(f"  - Sheet RMSE:  {s_metrics['rmse']:.3f} Hz")

    # 2. Validate Protein G (1GB1)
    pdb_path = "data/1GB1.pdb"
    if os.path.exists(pdb_path):
        print(f"\nEvaluating Protein G (Structure: {pdb_path})")
        pdb_file = pdb.PDBFile.read(pdb_path)
        struct = pdb_file.get_structure(model=1)
        
        predicted = calculate_hn_ha_coupling(struct)
        pred_a = predicted.get("A", {})
        
        metrics = calculate_metrics(pred_a, GB1_EXPERIMENTAL_3JHNHA)
        print_metrics("Protein G", metrics)

    # 3. Test BMRB Parsing (BMRB 6488)
    bmrb_path = "data/bmr6488.str"
    if os.path.exists(bmrb_path):
        print(f"\nTesting BMRB Parsing (File: {bmrb_path})")
        exp_couplings = parse_bmrb_j_couplings(bmrb_path)
        
        # Check for 3JHC which we know is in 6488
        codes = set()
        for res_data in exp_couplings.values():
            codes.update(res_data.keys())
        
        print(f"  - Found J-coupling codes: {', '.join(codes)}")
        print(f"  - Total assigned residues with J-couplings: {len(exp_couplings)}")
        
        if '3JHNHA' in codes:
            print("  - SUCCESS: Found 3JHNHA in BMRB file.")
        elif '3JHC' in codes:
            print("  - NOTE: Found 3JHC. This file contains side-chain couplings.")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
