"""Residual Dipolar Couplings (RDC) calculation module."""

# EDUCATIONAL NOTE - Introduction to RDCs:
# ==========================================
# Residual Dipolar Couplings (RDCs) are a powerful NMR observable for determining the
# structure and dynamics of biomolecules.
#
# 1. What are they?
#    In solution, molecules tumble rapidly, averaging out the through-space dipolar
#    couplings between nuclear spins to zero. However, if the solution is "anisotropic"
#    (e.g., by adding rod-shaped liquid crystals), molecules will have a slight preference
#    for a particular orientation. This incomplete averaging results in a small, measurable
#    "residual" dipolar coupling.
#
# 2. Why are they useful?
#    The magnitude of the RDC for a pair of atoms (like a backbone N-H) depends on the
#    angle of the vector between them with respect to a global "alignment tensor" that
#    describes the average orientation of the molecule.
#
#    RDC ~ < 3*cos^2(theta) - 1 >
#
#    Where 'theta' is the angle between the N-H vector and the main axis of alignment.
#    This provides long-range orientational information, which is a powerful restraint
#    for structure determination, complementary to short-range NOEs.

import logging
from typing import Dict

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)


def calculate_rdcs(structure: struc.AtomArray, Da: float, R: float) -> Dict[int, float]:
    """
    Calculates Residual Dipolar Couplings (RDCs) for backbone N-H vectors.

    This function assumes that the principal axis system (PAS) of the alignment
    tensor is aligned with the Cartesian coordinate axes (x, y, z) of the
    provided structure. The z-axis is assumed to be the principal axis.

    Parameters
    ----------
    structure : struc.AtomArray
        The protein structure (or a single residue) for which to calculate RDCs.
        Must contain backbone 'N' and 'H' atoms.
    Da : float
        The axial component of the alignment tensor in Hz. It defines the
        magnitude of the alignment. Must be a non-zero float.
    R : float
        The rhombicity of the alignment tensor (dimensionless, 0 <= R <= 2/3).
        It describes the deviation from axial symmetry.

    Returns
    -------
    Dict[int, float]
        A dictionary mapping each residue ID to its calculated N-H RDC value in Hz.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If input values (Da, R, or empty structure) are invalid.
    """
    # EDUCATIONAL NOTE - The Alignment Tensor:
    # ========================================
    # The alignment tensor (A) describes the average orientation of the molecule.
    # In its own Principal Axis System (PAS), it is diagonal:
    #      | Azz 0  0 |
    #  A = | 0  Ayy 0 |
    #      | 0  0  Axx|
    #
    # We use two parameters to describe it:
    # 1. Da (Axial component) = Azz
    # 2. R (Rhombicity) = (Axx - Ayy) / Azz
    #
    # The RDC for a vector 'v' depends on its orientation (theta, phi) in this PAS.

    # 1. Input Validation
    if not isinstance(structure, struc.AtomArray):
        raise TypeError("Input 'structure' must be a biotite.structure.AtomArray.")
    if structure.array_length() == 0:
        logger.warning("Input 'structure' is empty. Returning no RDCs.")
        return {}
    if not isinstance(Da, (int, float)):
        raise ValueError("Parameter 'Da' must be a numeric value.")
    if Da == 0:
        logger.warning("Parameter 'Da' is zero. All RDCs will be zero.")
    if not isinstance(R, (int, float)) or not (0 <= R <= 2 / 3):
        raise ValueError("Parameter 'R' must be a numeric value between 0 and 2/3 (inclusive).")

    logger.info(f"Starting RDC calculation with Da={Da:.2f} Hz, R={R:.3f}.")

    try:
        rdcs = {}

        # For efficiency, create a filter for backbone amide nitrogens and hydrogens
        n_atoms = structure[structure.atom_name == "N"]
        h_atoms = structure[structure.atom_name == "H"]

        if n_atoms.array_length() == 0 or h_atoms.array_length() == 0:
            logger.warning("Structure lacks backbone 'N' or 'H' atoms. Cannot calculate RDCs.")
            return {}

        # Create a lookup map from residue ID to the coordinate of its amide hydrogen
        h_coord_map = {h.res_id: h.coord for h in h_atoms}

        logger.debug(
            f"Found {n_atoms.array_length()} N atoms and {h_atoms.array_length()} H atoms."
        )

        # Iterate over each backbone nitrogen in the structure
        for n_atom in n_atoms:
            res_id = n_atom.res_id

            # Proline residues do not have a backbone amide proton, so they have no N-H RDC.
            if n_atom.res_name == "PRO":
                continue

            # Find the amide hydrogen corresponding to this nitrogen's residue
            if res_id not in h_coord_map:
                logger.debug(
                    f"No corresponding backbone H found for residue {res_id}, skipping RDC calculation."
                )
                continue

            h_coord = h_coord_map[res_id]
            n_coord = n_atom.coord

            # --- Vector Calculation ---
            # Determine the vector between the Nitrogen and Hydrogen atoms
            nh_vector = h_coord - n_coord

            # Normalize the vector to get a unit vector, which simplifies angle calculations
            norm = np.linalg.norm(nh_vector)
            if norm == 0:
                logger.warning(f"Residue {res_id} has a zero-length N-H vector. Skipping.")
                continue
            unit_vector = nh_vector / norm

            # --- Angle Calculation in the PAS ---
            # x, y, z components of the unit vector correspond to the cosines of the
            # angles the vector makes with the x, y, and z axes.
            x, y, z = unit_vector

            # 'theta' is the polar angle with respect to the Z-axis (the principal axis).
            # cos(theta) is simply the z-component of the unit vector.
            cos_theta = z

            # sin^2(theta) is needed for the rhombic part of the equation.
            # We use sin^2 = 1 - cos^2 to avoid an expensive sqrt() operation.
            sin_theta_sq = 1 - cos_theta**2

            # 'phi' is the azimuthal angle in the X-Y plane.
            # The RDC equation requires cos(2*phi).
            # cos(phi) = x / sin(theta)
            # sin(phi) = y / sin(theta)
            # Using the double angle identity: cos(2*phi) = cos^2(phi) - sin^2(phi)
            # cos(2*phi) = (x^2 - y^2) / sin_theta_sq
            if sin_theta_sq < 1e-9:
                # When the vector is nearly aligned with the Z-axis, sin(theta) is close to 0.
                # In this case, phi is ill-defined. However, the rhombic term in the RDC
                # equation is multiplied by sin^2(theta), so the whole term becomes zero.
                # We can set cos_2phi to any value (e.g., 1.0) as it will be nullified.
                cos_2phi = 1.0
            else:
                cos_2phi = (x * x - y * y) / sin_theta_sq

            # --- The RDC Equation ---
            # The full formula for the RDC value is:
            # D(theta, phi) = Da * [ (3*cos^2(theta) - 1) + (3/2) * R * sin^2(theta) * cos(2*phi) ]
            axial_term = 3 * cos_theta**2 - 1
            rhombic_term = 1.5 * R * sin_theta_sq * cos_2phi

            rdc_val = Da * (axial_term + rhombic_term)

            rdcs[res_id] = round(rdc_val, 2)

        if not rdcs and len(n_atoms) > 0:
            logger.warning(
                "No RDCs were calculated. Ensure the structure contains backbone N and corresponding H atoms."
            )
        elif rdcs:
            logger.info(f"Successfully calculated {len(rdcs)} RDCs.")

        return rdcs

    except Exception as e:
        logger.error(f"An unexpected error occurred during RDC calculation: {e}", exc_info=True)
        raise
