from typing import List, Dict, Set, Any, Tuple

"""
This module contains data definitions for the synth_pdb package, starting
with the 20 standard amino acids and their atomic configurations.
"""

# The 20 standard amino acids represented by their 3-letter codes.
# This list is used to randomly select amino acids for the sequence.
STANDARD_AMINO_ACIDS: List[str] = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLU",
    "GLN",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",  # Proline
    "VAL",
    "SER",
    "THR",
    "TRP",
    "TYR",
]

# EDUCATIONAL NOTE - Proline Sterics (The "Proline Effect"):
# --------------------------------------------------------
# Proline is the "structure breaker" of the protein world. Unlike any other 
# amino acid, its sidechain is cyclized back onto the backbone Nitrogen.
# 
# Consequences:
# 1. Exceptional Rigidity: The N-CA bond is locked in a ring, preventing 
#    rotation. This makes Proline the most restricted residue in the 
#    Ramachandran plot.
# 2. No H-Bonding Donor: Proline lacks the amide Hydrogen (H) that normally 
#    participates in alpha-helix and beta-sheet hydrogen bonding.
# 3. Cis-Trans Isomerization: While 99.9% of peptide bonds are "trans" (180 deg),
#    Proline is unique in that "cis" (0 deg) is energetically accessible, 
#    often forming sharp turns in protein chains.

# --- D-Amino Acids ---
# Mapping from 3-letter L-code to 3-letter D-code for PDB compatibility.
# Glycine is achiral and therefore has no D-form.
# 
# EDUCATIONAL NOTE - The "Mirror Image" World:
# -------------------------------------------
# Amino acids (except Glycine) are chiral, meaning they have a "handedness."
# In nature, almost all proteins are made of L-amino acids (Left-handed). 
# However, D-amino acids (Right-handed) exist and are structurally fascinating:
# 
# 1. Protease Resistance: Because most proteases are evolved to bind L-peptides,
#    incorporating D-amino acids can "camouflage" a peptide from degradation, 
#    extending its half-life from minutes to hours in the bloodstream.
# 2. Bacterial Signatures: D-Ala and D-Glu are key components of the bacterial 
#    cell wall (peptidoglycan). This makes them a target for the immune system 
#    and antibiotics like Vancomycin.
# 3. Structural Induction: D-amino acids favor different backbone angles. 
#    For example, a "D-Proline" can induce a different type of beta-turn 
#    that is energetically impossible for L-Proline.
#
# These 3-letter codes are standardized in the PDB (e.g., DAL for D-Alanine).
L_TO_D_MAPPING: Dict[str, str] = {
    "ALA": "DAL", "ARG": "DAR", "ASN": "DAN", "ASP": "DAS", "CYS": "DCY",
    "GLU": "DGL", "GLN": "DGN", "HIS": "DHI", "ILE": "DIL", "LEU": "DLE",
    "LYS": "DLY", "MET": "DME", "PHE": "DPH", "PRO": "DPR", "SER": "DSE",
    "THR": "DTH", "TRP": "DTR", "TYR": "DTY", "VAL": "DVA"
}
D_AMINO_ACIDS: List[str] = list(L_TO_D_MAPPING.values())

# Post-Translational Modifications
MODIFIED_AMINO_ACIDS: List[str] = [
    "SEP", # Phosphoserine
    "TPO", # Phosphothreonine
    "PTR", # Phosphotyrosine
]

ALL_VALID_AMINO_ACIDS: List[str] = STANDARD_AMINO_ACIDS + MODIFIED_AMINO_ACIDS + D_AMINO_ACIDS

# --- Amino Acid Frequencies (Approximate percentages in proteins) ---
# Source: Based on general protein composition data (e.g., from D. M. Smith, The Encyclopedia of Life Sciences, 2001)
# These are normalized to sum to 1.0
AMINO_ACID_FREQUENCIES: Dict[str, float] = {
    "ALA": 0.081,  # Alanine
    "ARG": 0.051,  # Arginine
    "ASN": 0.038,  # Asparagine
    "ASP": 0.054,  # Aspartic Acid
    "CYS": 0.019,  # Cysteine
    "GLU": 0.063,  # Glutamic Acid
    "GLN": 0.038,  # Glutamine
    "GLY": 0.073,  # Glycine
    "HIS": 0.023,  # Histidine
    "ILE": 0.055,  # Isoleucine
    "LEU": 0.091,  # Leucine
    "LYS": 0.059,  # Lysine
    "MET": 0.018,  # Methionine
    "PHE": 0.039,  # Phenylalanine
    "PRO": 0.052,  # Proline
    "SER": 0.062,  # Serine
    "THR": 0.054,  # Threonine
    "TRP": 0.014,  # Tryptophan
    "TYR": 0.032,  # Tyrosine
    "VAL": 0.069,  # Valine
}

# Mapping for 1-letter to 3-letter amino acid codes
# Standard IUPAC codes
ONE_TO_THREE_LETTER_CODE: Dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

# --- Ramachandran Angle Presets for Different Secondary Structures ---
# These presets define typical phi/psi backbone dihedral angles for common conformations
# Values are in degrees
# References:
#   - Ramachandran et al. (1963) J. Mol. Biol.
#   - Lovell et al. (2003) Proteins: Structure, Function, and Bioinformatics

RAMACHANDRAN_PRESETS: Dict[str, Dict[str, float]] = {
    'alpha': {
        'phi': -57.0,   # Alpha helix (right-handed)
        'psi': -47.0,
    },
    'beta': {
        'phi': -135.0,  # Beta sheet (antiparallel)
        'psi': 135.0,
    },
    'ppii': {
        'phi': -75.0,   # Polyproline II helix (left-handed)
        'psi': 145.0,
    },
    'extended': {
        'phi': -120.0,  # Extended/stretched conformation
        'psi': 120.0,
    },
    'curved': {
        # A specialized "tight" conformation to help the initial guess of cyclic peptides 
        # have N and C ends meet in 3D space.
        'phi': -80.0,
        'psi': -20.0,
    },
    # Residue-specific for random sampling
    'GLY': {
        'phi_mean': 0.0,
        'phi_std': 80.0,
        'psi_mean': 0.0,
        'psi_std': 80.0,
    },
    'PRO': {
        'phi_mean': -60.0,
        'phi_std': 15.0,
        'psi_mean': 120.0,
        'psi_std': 50.0,
    },
    'GENERAL': {
        'phi_mean': -90.0,
        'phi_std': 40.0,
        'psi_mean': 0.0,
        'psi_std': 80.0,
    },
}

# --- Ramachandran Probability Regions for Random Sampling ---
# Based on Lovell et al. (2003) Proteins: Structure, Function and Bioinformatics
# Used for realistic random conformation sampling

RAMACHANDRAN_REGIONS: Dict[str, Dict[str, Any]] = {
    'general': {
        'favored': [
            {'name': 'alpha', 'phi': -60.0, 'psi': -45.0, 'std': 10.0, 'weight': 0.60},
            {'name': 'beta', 'phi': -135.0, 'psi': 135.0, 'std': 10.0, 'weight': 0.30},
            {'name': 'left', 'phi': 60.0, 'psi': 45.0, 'std': 10.0, 'weight': 0.10},
        ],
    },
    'GLY': {
        # Glycine is more flexible (no side chain)
        'favored': [
            {'name': 'alpha', 'phi': -60.0, 'psi': -45.0, 'std': 15.0, 'weight': 0.40},
            {'name': 'beta', 'phi': -135.0, 'psi': 135.0, 'std': 20.0, 'weight': 0.30},
            {'name': 'left', 'phi': 60.0, 'psi': 45.0, 'std': 15.0, 'weight': 0.30},
        ],
    },
    'PRO': {
        # Proline is restricted (cyclic structure)
        'favored': [
            {'name': 'ppii', 'phi': -75.0, 'psi': 145.0, 'std': 10.0, 'weight': 0.70},
            {'name': 'alpha', 'phi': -60.0, 'psi': -30.0, 'std': 8.0, 'weight': 0.30},
        ],
    },
    'PRE_PRO': {
        # Residue immediately preceding Proline.
        # Steric clash with Pro CD limits Alpha region.
        # Beta/Extended is heavily favored.
        'favored': [
            {'name': 'beta', 'phi': -135.0, 'psi': 135.0, 'std': 10.0, 'weight': 0.75}, # Majority Beta
            {'name': 'alpha', 'phi': -60.0, 'psi': -45.0, 'std': 10.0, 'weight': 0.20}, # Minor Alpha
            {'name': 'left', 'phi': 60.0, 'psi': 45.0, 'std': 10.0, 'weight': 0.05},  # Very minor Left
        ],
    },
}

# --- MolProbity-style Ramachandran Polygons ---
# Polygonal definitions for "Favored" (98%) and "Allowed" (99.8%) regions.
# Coordinates are (phi, psi) pairs in degrees.
# These regions are approximations of the "Favored" (98% contour) and "Allowed" (99.8% contour) regions
# defined by the Top8000 dataset.
#
# Source Reference:
# Lovell et al. "Structure validation by Calpha geometry: phi,psi and Cbeta deviation."
# Proteins: Structure, Function, and Bioinformatics, 50: 437-450 (2003).
# URL: http://kinemage.biochem.duke.edu/databases/top8000.php
#
# Note: The polygons below are simplified manual tracings of the contours for computational efficiency.

RAMACHANDRAN_POLYGONS: Dict[str, Dict[str, List[List[Tuple[float, float]]]]] = {
    "General": {
        "Favored": [
            # Alpha-Helical Region (Core)
            [(-90, -70), (-70, -80), (-50, -60), (-40, -50), (-30, -30), (-50, -10), (-70, -10), (-90, -30)],
            # Beta-Sheet Region (Broad top-left)
            [(-180, 180), (-180, 90), (-110, 90), (-70, 120), (-50, 150), (-50, 180)],
            # Polyproline II / Extended (Connecting to Beta)
            [(-180, -180), (-180, -140), (-140, -140), (-140, -180)],
        ],
        "Allowed": [
            # Alpha Allowed (Generous surround)
            [(-110, -90), (-40, -90), (-20, -20), (-40, 10), (-110, 10)],
            # Beta Allowed (Generous surround)
            [(-180, 180), (-180, 60), (-50, 60), (-30, 140), (-30, 180)],
            [(-180, -180), (-180, -120), (-120, -120), (-120, -180)],
        ]
    },
    "GLY": {
        "Favored": [
            # Glycine is unique: Achiral, so symmetric.
            # Alpha-Right (Positive Phi)
            [(40, 20), (80, 20), (80, 80), (40, 80)],
            # Alpha-Left (Negative Phi)
            [(-80, -80), (-40, -80), (-40, -20), (-80, -20)],
            # Beta / Extended (Both sides)
            [(-180, 180), (-180, 100), (-60, 100), (-60, 180)],
            [(60, 180), (60, 100), (180, 100), (180, 180)],
            [(-180, -180), (-180, -100), (-60, -100), (-60, -180)],
            [(60, -180), (60, -100), (180, -100), (180, -180)],
        ],
        "Allowed": [
             # Practically everywhere except steric limit (Phi=0, Psi=0)
             [(-180, 180), (-180, -180), (180, -180), (180, 180)]
        ]
    },
    "PRO": {
        "Favored": [
            # Restricted Phi ~ -60 +/- 15
            # Alpha region
            [(-75, -20), (-50, -20), (-50, -50), (-75, -50)],
            # Polyproline II (PPII) - Very common for Proline
            [(-90, 120), (-60, 120), (-60, 180), (-90, 180)],
        ],
        "Allowed": [
            # Slightly wider strip
            [(-100, -180), (-40, -180), (-40, 180), (-100, 180)]
        ]
    },
    "Pre-Pro": {
         "Favored": [
             # Beta region is similar
             [(-180, 180), (-180, 90), (-60, 120), (-60, 180)],
             # Alpha region is strictly restricted (steric clash with Pro CD)
             [(-70, -30), (-50, -20), (-40, -50), (-60, -70)],
         ],
         "Allowed": [
             [(-180, 180), (-180, -180), (-30, -180), (-30, 180)],
         ]
    }
}

# --- Standard Bond Lengths and Angles (Approximations) ---

# Values are in Angstroms for bond lengths and degrees for angles
# 
# EDUCATIONAL NOTE - Engh & Huber Parameters (The Gold Standard):
# -------------------------------------------------------------
# These bond lengths and angles are not just "good guesses." They are the 
# "Engh & Huber Parameters," derived in 1991 from a statistical survey of 
# small-molecule crystal structures in the Cambridge Structural Database (CSD).
# 
# Significance:
# 1. Foundation of Crystallography: Almost all protein structure refinement 
#    software (like Phenix or CCP4) uses these targets as "restraints." If a 
#    structure deviates too far from 1.458Ą for an N-CA bond, it's flagged.
# 2. Physics-Informed: By assuming these bond lengths are fixed (rigid), we 
#    reduce the complexity of protein simulation from thousands of variables 
#    down to just a few torsion angles (Phi, Psi, Chi).
# 
# Source: Engh, R. A., & Huber, R. (1991). Acta Crystallographica Section A.

# Peptide bond geometry

BOND_LENGTH_N_CA: float = 1.458  # N-Calpha

BOND_LENGTH_CA_C: float = 1.525  # Calpha-C

BOND_LENGTH_C_N: float = 1.329  # C-N (peptide bond)

BOND_LENGTH_C_O: float = 1.231  # C=O (carbonyl)


ANGLE_CA_C_N: float = 116.2  # Calpha-C-N

ANGLE_C_N_CA: float = 121.7  # C-N-Calpha

ANGLE_N_CA_C: float = 110.0  # N-Calpha-C (tetrahedral approx)

ANGLE_CA_C_O: float = 120.8  # Calpha-C=O


# Side chain geometry (approximate)

BOND_LENGTH_CA_CB: float = 1.53  # Calpha-Cbeta (typical)

BOND_LENGTH_C_H: float = 1.08  # C-H (typical)

BOND_LENGTH_N_H: float = 1.01  # N-H (typical)


# Van der Waals radii in Angstroms (approximate values)
# Source: Wikipedia, various chemistry texts.
# These are simplified values for common protein atoms.

VAN_DER_WAALS_RADII: Dict[str, float] = {
    "H": 1.20,  # Hydrogen
    "C": 1.70,  # Carbon
    "N": 1.55,  # Nitrogen
    "O": 1.52,  # Oxygen
    "S": 1.80,  # Sulfur
    "ZN": 1.40, # Zinc
}

# Amino acid properties for sequence improbability checks
CHARGED_AMINO_ACIDS: Set[str] = {
    "ARG",
    "HIS",
    "LYS",
    "ASP",
    "GLU",
}  # K, R, H (positive); D, E (negative)
POSITIVE_AMINO_ACIDS: Set[str] = {"ARG", "HIS", "LYS"}
NEGATIVE_AMINO_ACIDS: Set[str] = {"ASP", "GLU"}

HYDROPHOBIC_AMINO_ACIDS: Set[str] = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR"}
HYDROPHILIC_AMINO_ACIDS: Set[str] = {
    "ARG",
    "ASN",
    "ASP",
    "GLN",
    "GLU",
    "HIS",
    "LYS",
    "SER",
    "THR",
}  # Contains charged ones too
POLAR_UNCHARGED_AMINO_ACIDS: Set[str] = {
    "ASN",
    "GLN",
    "SER",
    "THR",
    "CYS",
    "TYR",
}  # CYS, TYR can be ambivalent

# --- Atomic Definitions for Each Amino Acid ---
# NOTE: This dictionary is currently UNUSED in the codebase.
# The generator uses biotite's residue templates instead for atom placement.
# This data is RESERVED for potential future use in custom rotamer library implementation
# or manual side-chain placement algorithms.
#
# Each amino acid defines its atoms relative to the C-alpha (CA) position (0,0,0)
# for side chain atoms. Backbone atoms (N, C, O) would be placed based on previous
# residue geometry in the generator.
# Format: {'name': 'ATOM_NAME', 'element': 'ELEMENT_SYMBOL', 'coords': [x, y, z]}
# For simplicity, coords are relative to CA, assuming CA is at (0,0,0) for side chain placement.

AMINO_ACID_ATOMS: Dict[str, List[Dict[str, Any]]] = {
    "ALA": [
        # Backbone (N, CA, C, O handled by generator)
        {"name": "CB", "element": "C", "coords": [1.4, 0.0, 0.0]},  # Placeholder for CB
        {"name": "HB1", "element": "H", "coords": [2.0, 0.8, 0.0]},
        {"name": "HB2", "element": "H", "coords": [2.0, -0.8, 0.0]},
        {"name": "HB3", "element": "H", "coords": [1.0, 0.0, 0.8]},
    ],
    "ARG": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD", "element": "C", "coords": [4.3, 0.0, 0.0]},
        {"name": "NE", "element": "N", "coords": [5.5, 0.0, 0.0]},
        {"name": "CZ", "element": "C", "coords": [6.6, 0.0, 0.0]},
        {"name": "NH1", "element": "N", "coords": [7.7, 0.8, 0.0]},
        {"name": "NH2", "element": "N", "coords": [7.7, -0.8, 0.0]},
    ],
    "ASN": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "OD1", "element": "O", "coords": [3.5, 0.8, 0.0]},
        {"name": "ND2", "element": "N", "coords": [3.5, -0.8, 0.0]},
    ],
    "ASP": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "OD1", "element": "O", "coords": [3.5, 0.8, 0.0]},
        {"name": "OD2", "element": "O", "coords": [3.5, -0.8, 0.0]},
    ],
    "CYS": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "SG", "element": "S", "coords": [2.9, 0.0, 0.0]},
    ],
    "GLU": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD", "element": "C", "coords": [4.3, 0.0, 0.0]},
        {"name": "OE1", "element": "O", "coords": [4.9, 0.8, 0.0]},
        {"name": "OE2", "element": "O", "coords": [4.9, -0.8, 0.0]},
    ],
    "GLN": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD", "element": "C", "coords": [4.3, 0.0, 0.0]},
        {"name": "OE1", "element": "O", "coords": [4.9, 0.8, 0.0]},
        {"name": "NE2", "element": "N", "coords": [4.9, -0.8, 0.0]},
    ],
    "GLY": [
        # Glycine has no beta carbon, only alpha-hydrogens
        {"name": "HA1", "element": "H", "coords": [0.7, 0.7, 0.0]},
        {"name": "HA2", "element": "H", "coords": [0.7, -0.7, 0.0]},
    ],
    "HIS": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "ND1", "element": "N", "coords": [3.5, 1.0, 0.0]},
        {"name": "CD2", "element": "C", "coords": [3.5, -1.0, 0.0]},
        {"name": "CE1", "element": "C", "coords": [4.5, 1.0, 0.0]},
        {"name": "NE2", "element": "N", "coords": [4.5, -1.0, 0.0]},
    ],
    "ILE": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG1", "element": "C", "coords": [2.5, 1.0, 0.0]},
        {"name": "CG2", "element": "C", "coords": [2.5, -1.0, 0.0]},
        {"name": "CD1", "element": "C", "coords": [3.5, 1.0, 0.0]},
    ],
    "LEU": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD1", "element": "C", "coords": [3.9, 1.0, 0.0]},
        {"name": "CD2", "element": "C", "coords": [3.9, -1.0, 0.0]},
    ],
    "LYS": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD", "element": "C", "coords": [4.3, 0.0, 0.0]},
        {"name": "CE", "element": "C", "coords": [5.7, 0.0, 0.0]},
        {"name": "NZ", "element": "N", "coords": [7.0, 0.0, 0.0]},
    ],
    "MET": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "SD", "element": "S", "coords": [4.3, 0.0, 0.0]},
        {"name": "CE", "element": "C", "coords": [5.7, 0.0, 0.0]},
    ],
    "PHE": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD1", "element": "C", "coords": [3.9, 1.0, 0.0]},
        {"name": "CD2", "element": "C", "coords": [3.9, -1.0, 0.0]},
        {"name": "CE1", "element": "C", "coords": [4.9, 1.0, 0.0]},
        {"name": "CE2", "element": "C", "coords": [4.9, -1.0, 0.0]},
        {"name": "CZ", "element": "C", "coords": [5.9, 0.0, 0.0]},
    ],
    "PRO": [
        # Proline is special, its N is part of a ring. Simplified here.
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD", "element": "C", "coords": [4.0, 0.0, 0.0]},
    ],
    "SER": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "OG", "element": "O", "coords": [2.9, 0.0, 0.0]},
    ],
    "THR": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "OG1", "element": "O", "coords": [2.5, 1.0, 0.0]},
        {"name": "CG2", "element": "C", "coords": [2.5, -1.0, 0.0]},
    ],
    "TRP": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD1", "element": "C", "coords": [3.9, 1.0, 0.0]},
        {"name": "CD2", "element": "C", "coords": [3.9, -1.0, 0.0]},
        {"name": "NE1", "element": "N", "coords": [4.9, 1.0, 0.0]},
        {"name": "CE2", "element": "C", "coords": [4.9, -1.0, 0.0]},
        {"name": "CE3", "element": "C", "coords": [5.9, -1.0, 0.0]},
        {"name": "CZ2", "element": "C", "coords": [5.9, 1.0, 0.0]},
        {"name": "CZ3", "element": "C", "coords": [6.9, -1.0, 0.0]},
        {"name": "CH2", "element": "C", "coords": [6.9, 1.0, 0.0]},
    ],
    "TYR": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG", "element": "C", "coords": [2.9, 0.0, 0.0]},
        {"name": "CD1", "element": "C", "coords": [3.9, 1.0, 0.0]},
        {"name": "CD2", "element": "C", "coords": [3.9, -1.0, 0.0]},
        {"name": "CE1", "element": "C", "coords": [4.9, 1.0, 0.0]},
        {"name": "CE2", "element": "C", "coords": [4.9, -1.0, 0.0]},
        {"name": "CZ", "element": "C", "coords": [5.9, 0.0, 0.0]},
        {"name": "OH", "element": "O", "coords": [6.9, 0.0, 0.0]},
    ],
    "VAL": [
        {"name": "CB", "element": "C", "coords": [1.5, 0.0, 0.0]},
        {"name": "CG1", "element": "C", "coords": [2.5, 1.0, 0.0]},
        {"name": "CG2", "element": "C", "coords": [2.5, -1.0, 0.0]},
    ],
}

# --- Rotamer Library (Chi Angles) ---
# Simplified rotamer library with common chi1 and chi2 angles
# Based on Dunbrack rotamer library (backbone-independent)
# Values are in degrees
# Reference: Dunbrack & Cohen (1997) Protein Science
# Note: Only the most common rotamer is included for each amino acid

ROTAMER_LIBRARY: Dict[str, List[Dict[str, List[float]]]] = {
    # Aliphatic amino acids
    'ALA': [],  # No side-chain dihedrals
    'VAL': [
        {'chi1': [-60.0], 'prob': 0.70},  # Gauche- (most common)
        {'chi1': [180.0], 'prob': 0.20},  # Trans
        {'chi1': [60.0],  'prob': 0.10},  # Gauche+
    ],
    'LEU': [
        {'chi1': [-60.0], 'chi2': [65.0], 'prob': 0.60},   # g-, g+
        {'chi1': [-60.0], 'chi2': [180.0], 'prob': 0.25},  # g-, t
        {'chi1': [180.0], 'chi2': [65.0], 'prob': 0.10},   # t, g+
    ],
    'ILE': [
        {'chi1': [-60.0], 'chi2': [170.0], 'prob': 0.60},  # g-, t
        {'chi1': [-60.0], 'chi2': [-60.0], 'prob': 0.25},  # g-, g-
        {'chi1': [180.0], 'chi2': [170.0], 'prob': 0.10},  # t, t
    ],
    'MET': [
        {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [70.0], 'prob': 0.40},
        {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [180.0],'prob': 0.30},
    ],
    
    # Aromatic amino acids
    'PHE': [
        {'chi1': [-60.0], 'chi2': [90.0], 'prob': 0.50},   # g-, g+ (perp)
        {'chi1': [180.0], 'chi2': [90.0], 'prob': 0.30},   # t, g+
        {'chi1': [-60.0], 'chi2': [0.0],  'prob': 0.10},   # g-, planar (rare)
    ],
    'TYR': [
        {'chi1': [-60.0], 'chi2': [90.0], 'prob': 0.50},
        {'chi1': [180.0], 'chi2': [90.0], 'prob': 0.30},
    ],
    'TRP': [
        {'chi1': [-60.0], 'chi2': [-90.0], 'prob': 0.50},
        {'chi1': [180.0], 'chi2': [-90.0], 'prob': 0.30},
    ],
    
    # Polar uncharged
    'SER': [
        {'chi1': [60.0], 'prob': 0.45},   # g+
        {'chi1': [-60.0], 'prob': 0.35},  # g-
        {'chi1': [180.0], 'prob': 0.20},  # t
    ],
    'THR': [
        {'chi1': [60.0], 'prob': 0.45},   # g+
        {'chi1': [-60.0], 'prob': 0.45},  # g-
        {'chi1': [180.0], 'prob': 0.10},  # t
    ],
    'CYS': [
        {'chi1': [-60.0], 'prob': 0.50},
        {'chi1': [180.0], 'prob': 0.30},
        {'chi1': [60.0], 'prob': 0.20},
    ],
    'ASN': [
        {'chi1': [-60.0], 'chi2': [-20.0], 'prob': 0.40},
        {'chi1': [180.0], 'chi2': [-20.0], 'prob': 0.30},
        {'chi1': [-60.0], 'chi2': [180.0], 'prob': 0.15}, # Flip
    ],
    'GLN': [
        {'chi1': [-60.0], 'chi2': [60.0], 'chi3': [0.0], 'prob': 0.40},
        {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [0.0],'prob': 0.30},
    ],
    
    # Charged
    'ASP': [
        {'chi1': [-60.0], 'chi2': [0.0], 'prob': 0.45},
        {'chi1': [180.0], 'chi2': [0.0], 'prob': 0.35},
    ],
    'GLU': [
        {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [0.0], 'prob': 0.40},
        {'chi1': [180.0], 'chi2': [180.0], 'chi3': [0.0], 'prob': 0.30},
    ],
    'LYS': [
        {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [180.0], 'prob': 0.40},
        {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [60.0],  'prob': 0.20},
    ],
    'ARG': [
        {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [-85.0], 'prob': 0.35},
        {'chi1': [180.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [-85.0], 'prob': 0.25},
    ],
    'HIS': [
        {'chi1': [-60.0], 'chi2': [-75.0], 'prob': 0.40},
        {'chi1': [180.0], 'chi2': [-75.0], 'prob': 0.30},
    ],
    
    # Special cases
    'GLY': [],
    'PRO': [],
}

# --- Backbone-Dependent Rotamer Library (Simplified) ---
# EDUCATIONAL NOTE - Backbone Dependency:
# Rotamer probabilities depend heavily on the local backbone conformation (Phi/Psi angles).
# This is due to steric hindrance between side-chain atoms and the backbone.
#
# Key Biophysical Principles (Dunbrack, 2002):
# 1. Alpha-Helix (-60, -45): 
#    - The "trans" (t, 180 deg) rotamer for Chi1 is often strongly disfavored for 
#      branched residues (Val, Ile, Thr) due to clashes with the backbone i-3 or i-4.
#    - "g-" (-60 deg) is typically the dominant rotamer.
#
# 2. Beta-Sheet (-135, 135):
#    - The extended backbone allows more freedom. "trans" rotamers become much more 
#      favorable compared to helices.
#
# This simplified library maps secondary structure types ('alpha', 'beta', etc.)
# to specific rotamer probability distributions.
BACKBONE_DEPENDENT_ROTAMER_LIBRARY: Dict[str, Dict[str, List[Dict[str, List[float]]]]] = {
    'VAL': {
        'alpha': [
            # In Helix: g- is dominant, trans is rare
            {'chi1': [-60.0], 'prob': 0.90},  # g-
            {'chi1': [180.0], 'prob': 0.05},  # t (rare)
            {'chi1': [60.0],  'prob': 0.05},  # g+ (rare)
        ],
        'beta': [
            # In Sheet: trans is very common
            {'chi1': [-60.0], 'prob': 0.55},  # g-
            {'chi1': [180.0], 'prob': 0.40},  # t (uniquely favored in sheets)
            {'chi1': [60.0],  'prob': 0.05},  # g+
        ],
    },
    'ILE': {
        'alpha': [
            {'chi1': [-60.0], 'chi2': [170.0], 'prob': 0.85},
            {'chi1': [-60.0], 'chi2': [-60.0], 'prob': 0.10},
            {'chi1': [180.0], 'chi2': [170.0], 'prob': 0.05}, # t disallowed
        ],
        'beta': [
            {'chi1': [-60.0], 'chi2': [170.0], 'prob': 0.50},
            {'chi1': [-60.0], 'chi2': [-60.0], 'prob': 0.20},
            {'chi1': [180.0], 'chi2': [170.0], 'prob': 0.30}, # t allowed
        ]
    },
    'THR': {
        'alpha': [
            {'chi1': [60.0], 'prob': 0.40},
            {'chi1': [-60.0], 'prob': 0.55}, # g- 
            {'chi1': [180.0], 'prob': 0.05},
        ],
        'beta': [
            {'chi1': [60.0], 'prob': 0.30},
            {'chi1': [-60.0], 'prob': 0.40},
            {'chi1': [180.0], 'prob': 0.30}, # t allowed
        ]
    },
    # --- Expanded Library (LEU, LYS, Aromatics) ---
    # EDUCATIONAL NOTE - Rotamers for Non-Branched Residues:
    # Leucine (LEU) and Lysine (LYS) have flexible side chains.
    # However, secondary structure still dictates Chi1 preferences.
    # In Helices, side chains project outwards and must avoid i-3/i-4 backbone atoms.
    # 'trans' (180 deg) at Chi1 is generally disfavored in helices compared to sheets.
    'LEU': {
        'alpha': [
            # In alpha-helices, g- (mt) is strongly preferred (~90%)
            # Reference: Lovell et al., Proteins 2000.
            {'chi1': [-60.0], 'chi2': [180.0], 'prob': 0.65},  # mt
            {'chi1': [-60.0], 'chi2': [60.0],  'prob': 0.25},  # mp (trans is gauche to C-gamma)
            {'chi1': [180.0], 'chi2': [60.0],  'prob': 0.05},  # tp
            {'chi1': [180.0], 'chi2': [180.0], 'prob': 0.05},  # tt
        ],
        'beta': [
            # In beta-sheets, the backbone is extended, allowing trans rotamers more easily.
            # tp (trans, plus) becomes a major conformer.
            {'chi1': [-60.0], 'chi2': [180.0], 'prob': 0.40},  # mt
            {'chi1': [180.0], 'chi2': [60.0],  'prob': 0.40},  # tp (Distinctive for Beta)
            {'chi1': [-60.0], 'chi2': [60.0],  'prob': 0.15}, 
            {'chi1': [180.0], 'chi2': [180.0], 'prob': 0.05}, 
        ]
    },
    'LYS': {
        'alpha': [
             # Helices favor g- at Chi1 to avoid backbone clashes.
             {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [180.0], 'prob': 0.70},
             {'chi1': [180.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [180.0], 'prob': 0.20},
             {'chi1': [60.0],  'chi2': [180.0], 'chi3': [180.0], 'chi4': [180.0], 'prob': 0.10},
        ],
        'beta': [
             # Sheets allow extended (all-trans) conformations much more readily.
             {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [180.0], 'prob': 0.40},
             {'chi1': [180.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [180.0], 'prob': 0.50}, # All-trans favored
             {'chi1': [60.0],  'chi2': [180.0], 'chi3': [180.0], 'chi4': [180.0], 'prob': 0.10},
        ]
    },
    # EDUCATIONAL NOTE - Aromatic Residues (PHE, TYR, TRP):
    # These bulky side chains are highly restricted by steric clashes.
    # In alpha-helices, Chi1=180 (trans) causes the ring to clash with the i-3 residue.
    # Thus, g- is overwhelmingly dominant in helices.
    'PHE': {
        'alpha': [
            {'chi1': [-60.0], 'chi2': [90.0], 'prob': 0.90},  # g- (Dominant)
            {'chi1': [180.0], 'chi2': [90.0], 'prob': 0.05},  # t (Rare/Disfavored)
            {'chi1': [-60.0], 'chi2': [0.0],  'prob': 0.05}
        ],
        'beta': [
            {'chi1': [-60.0], 'chi2': [90.0], 'prob': 0.50},
            {'chi1': [180.0], 'chi2': [90.0], 'prob': 0.45},  # t (Common in sheets)
            {'chi1': [-60.0], 'chi2': [0.0],  'prob': 0.05}
        ]
    },
    'TYR': {
        'alpha': [
            {'chi1': [-60.0], 'chi2': [90.0], 'prob': 0.90},
            {'chi1': [180.0], 'chi2': [90.0], 'prob': 0.05},
        ],
        'beta': [
            {'chi1': [-60.0], 'chi2': [90.0], 'prob': 0.50},
            {'chi1': [180.0], 'chi2': [90.0], 'prob': 0.45},
        ]
    },
    'TRP': {
        'alpha': [
            {'chi1': [-60.0], 'chi2': [-90.0], 'prob': 0.90}, # g-
            {'chi1': [180.0], 'chi2': [-90.0], 'prob': 0.05}, # t
        ],
        'beta': [
            {'chi1': [-60.0], 'chi2': [-90.0], 'prob': 0.50},
            {'chi1': [180.0], 'chi2': [-90.0], 'prob': 0.45}, # t
        ]
    },
    
    # --- Charged Residues (ARG, ASP, GLU) & Polar (GLN, HIS, MET) ---
    # EDUCATIONAL NOTE - Electrostatics vs Sterics:
    # While charged residues form salt bridges on the surface, their Chi1 preference
    # is still dominated by backbone sterics.
    # - Long chains (ARG, GLU, GLN, MET) behave like Lysine: g- in Helix, Trans in Sheet.
    # - Short chains (ASP, ASN) have specific "g-" preferences to avoid O...C-backbone clashes.
    
    'ARG': {
        'alpha': [
            {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [-85.0], 'prob': 0.70}, # g-
            {'chi1': [180.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [-85.0], 'prob': 0.20}, # t
            {'chi1': [60.0],  'chi2': [180.0], 'chi3': [180.0], 'chi4': [-85.0], 'prob': 0.10},
        ],
        'beta': [
            {'chi1': [180.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [-85.0], 'prob': 0.50}, # t (Favored in sheet)
            {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [180.0], 'chi4': [-85.0], 'prob': 0.40},
            {'chi1': [60.0],  'chi2': [180.0], 'chi3': [180.0], 'chi4': [-85.0], 'prob': 0.10},
        ]
    },
    'GLU': {
        'alpha': [
            {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [0.0], 'prob': 0.70},
            {'chi1': [180.0], 'chi2': [180.0], 'chi3': [0.0], 'prob': 0.20},
            {'chi1': [60.0],  'chi2': [180.0], 'chi3': [0.0], 'prob': 0.10},
        ],
        'beta': [
            {'chi1': [180.0], 'chi2': [180.0], 'chi3': [0.0], 'prob': 0.50}, # t
            {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [0.0], 'prob': 0.40},
            {'chi1': [60.0],  'chi2': [180.0], 'chi3': [0.0], 'prob': 0.10},
        ]
    },
    'GLN': {
        'alpha': [
            {'chi1': [-60.0], 'chi2': [60.0], 'chi3': [0.0], 'prob': 0.70},
            {'chi1': [180.0], 'chi2': [60.0], 'chi3': [0.0], 'prob': 0.20},
        ],
        'beta': [
            {'chi1': [180.0], 'chi2': [60.0], 'chi3': [0.0], 'prob': 0.50},
            {'chi1': [-60.0], 'chi2': [60.0], 'chi3': [0.0], 'prob': 0.40},
        ]
    },
    'MET': {
        'alpha': [
            {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [70.0], 'prob': 0.75}, # g-
            {'chi1': [180.0], 'chi2': [180.0], 'chi3': [70.0], 'prob': 0.15},
        ],
        'beta': [
            {'chi1': [180.0], 'chi2': [180.0], 'chi3': [70.0], 'prob': 0.55}, # t
            {'chi1': [-60.0], 'chi2': [180.0], 'chi3': [70.0], 'prob': 0.35},
        ]
    },
    'ASP': {
        'alpha': [
            {'chi1': [-60.0], 'chi2': [0.0], 'prob': 0.80},
            {'chi1': [180.0], 'chi2': [0.0], 'prob': 0.15},
        ],
        'beta': [
            {'chi1': [-60.0], 'chi2': [0.0], 'prob': 0.50},
            {'chi1': [180.0], 'chi2': [0.0], 'prob': 0.45}, # t allowed
        ]
    },
    'ASN': {
        'alpha': [
            {'chi1': [-60.0], 'chi2': [-20.0], 'prob': 0.80},
            {'chi1': [180.0], 'chi2': [-20.0], 'prob': 0.15},
        ],
        'beta': [
            {'chi1': [-60.0], 'chi2': [-20.0], 'prob': 0.50},
            {'chi1': [180.0], 'chi2': [-20.0], 'prob': 0.45},
        ]
    },
    'HIS': {
        # Similar to Aromatics
        'alpha': [
            {'chi1': [-60.0], 'chi2': [-75.0], 'prob': 0.90}, # g-
            {'chi1': [180.0], 'chi2': [-75.0], 'prob': 0.05}, # t disallowed
        ],
        'beta': [
            {'chi1': [-60.0], 'chi2': [-75.0], 'prob': 0.50},
            {'chi1': [180.0], 'chi2': [-75.0], 'prob': 0.45}, # t allowed
        ]
    },
    'CYS': {
         'alpha': [{'chi1': [-60.0], 'prob': 0.90}, {'chi1': [180.0], 'prob': 0.10}],
         'beta':  [{'chi1': [-60.0], 'prob': 0.50}, {'chi1': [180.0], 'prob': 0.50}],
    },
    'SER': {
         'alpha': [{'chi1': [-60.0], 'prob': 0.45}, {'chi1': [60.0], 'prob': 0.45}, {'chi1': [180.0], 'prob': 0.10}],
         'beta':  [{'chi1': [-60.0], 'prob': 0.30}, {'chi1': [60.0], 'prob': 0.40}, {'chi1': [180.0], 'prob': 0.30}],
    }
}

# --- Beta-Turn Definitions ---
# Beta-turns involve 4 residues (i to i+3). The geometry is defined by the
# Phi and Psi angles of residues i+1 and i+2.
#
# References:
# - Venkatachalam CM (1968). Biopolymers.
# - Richardson JS (1981). Adv. Protein Chem.
#
# Format: "Type": [(Phi_i+1, Psi_i+1), (Phi_i+2, Psi_i+2)]
# Values in degrees.
BETA_TURN_TYPES: Dict[str, List[Tuple[float, float]]] = {
    "typeI":  [(-60.0, -30.0), (-90.0, 0.0)],
    "typeII": [(-60.0, 120.0), (80.0, 0.0)],   # Classic Type II (Res 3 often Gly)
    "typeI'": [(60.0, 30.0), (90.0, 0.0)],     # Inverse Type I (Goldenberg)
    "typeII'": [(60.0, -120.0), (-80.0, 0.0)], # Inverse Type II
    # Type VIII is also common
    "typeVIII": [(-60.0, -30.0), (-120.0, 120.0)],
}

# --- Chi Angle Definitions ---
# Defines the atoms required to calculate each Chi angle.
# N-CA-CB-G  (Chi1)
# CA-CB-G-D  (Chi2)
# CB-G-D-E   (Chi3)
# G-D-E-Z    (Chi4)
AMINO_ACID_CHI_DEFINITIONS: Dict[str, List[Dict[str, List[str]]]] = {
    'VAL': [{'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG1']}],
    'THR': [{'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'OG1']}],
    'ILE': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG1']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG1', 'CD1']}
    ],
    'LEU': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'CD1']}
    ],
    'MET': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'SD']},
        {'name': 'chi3', 'atoms': ['CB', 'CG', 'SD', 'CE']}
    ],
    'ARG': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'CD']},
        {'name': 'chi3', 'atoms': ['CB', 'CG', 'CD', 'NE']},
        {'name': 'chi4', 'atoms': ['CG', 'CD', 'NE', 'CZ']}
    ],
    'LYS': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'CD']},
        {'name': 'chi3', 'atoms': ['CB', 'CG', 'CD', 'CE']},
        {'name': 'chi4', 'atoms': ['CG', 'CD', 'CE', 'NZ']}
    ],
    'ASP': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'OD1']}
    ],
    'ASN': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'OD1']}
    ],
    'GLU': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'CD']},
        {'name': 'chi3', 'atoms': ['CB', 'CG', 'CD', 'OE1']}
    ],
    'GLN': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'CD']},
        {'name': 'chi3', 'atoms': ['CB', 'CG', 'CD', 'OE1']}
    ],
    'HIS': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'ND1']}
    ],
    'PHE': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'CD1']}
    ],
    'TYR': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'CD1']}
    ],
    'TRP': [
        {'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'CG']},
        {'name': 'chi2', 'atoms': ['CA', 'CB', 'CG', 'CD1']}
    ],
    'CYS': [{'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'SG']}],
    'SER': [{'name': 'chi1', 'atoms': ['N', 'CA', 'CB', 'OG']}],
    # PRO, GLY, ALA have no standard Chi1 or special definitions handled specially
}
