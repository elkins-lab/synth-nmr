"""
synth_nmr.trajectory
~~~~~~~~~~~~~~~~~~~~
MD Trajectory loading and Ensemble NMR observable averaging.

MOTIVATION — Why single structures are not enough:
===================================================
Every other module in synth-nmr answers the question:
  "What would the NMR spectrum of THIS particular 3D structure look like?"

That is useful, but it is a simplification.  A protein in aqueous solution
is in constant thermal motion — it samples an enormous ensemble of slightly
different conformations on timescales from picoseconds (bond vibrations) to
milliseconds (domain movements).

A solution NMR experiment does NOT see a single snapshot.  The spectrometer
integrates over whatever conformational dynamics are fast compared to the
relevant NMR timescale:

  • Fast (ns and faster): chemical shifts → the peak appears at the
    TIME-AVERAGE of the instantaneous chemical shift.
  • Fast (ns): NOEs → the cross-relaxation rate is proportional to
    the TIME-AVERAGE of r⁻⁶, giving an "effective distance" that is
    shorter than the arithmetic-mean distance.
  • Fast (ps–ns): Bond-vector flexibility → the order parameter S²
    reports on HOW MUCH a bond vector has moved during the experiment.

This module bridges the gap by:
  1. Holding a set of structure "frames" (a TrajectoryEnsemble).
  2. Providing functions that compute the physically correct ensemble average
     of each NMR observable over those frames.

Dependency strategy (optional MDTraj):
=======================================
MDTraj is the most convenient way to read GROMACS (.gro/.xtc) or AMBER
(.prmtop/.nc) trajectory files in Python.  However, it is a large C-extension
and is not always available in every environment.

We follow the same pattern used for numba in synth-nmr:
  • MDTraj is imported inside load_trajectory() — NOT at module import time.
  • If MDTraj is absent and the user passes a file path, a clear ImportError
    with install instructions is raised.
  • If the user passes a plain list of biotite AtomArrays, MDTraj is never
    needed.  All the averaging functions work purely with AtomArrays.

Install MDTraj support:
    pip install synth-nmr[trajectory]   # installs mdtraj>=1.9.0

Usage:
======
>>> from synth_nmr.trajectory import load_trajectory, ensemble_average_shifts
>>> from synth_nmr import predict_chemical_shifts
>>>
>>> # Option A: from a list of biotite AtomArrays (no MDTraj required)
>>> frames = [load_structure(f"frame_{i}.pdb") for i in range(100)]
>>> ensemble = load_trajectory(frames)
>>>
>>> # Option B: from an MDTraj trajectory (requires MDTraj)
>>> # ensemble = load_trajectory("md.xtc", topology="protein.pdb")
>>>
>>> # Per-frame predictions
>>> per_frame = [predict_chemical_shifts(f) for f in ensemble]
>>>
>>> # Ensemble average
>>> avg_shifts = ensemble_average_shifts(per_frame)
>>> print(avg_shifts[1]["CA"])  # time-averaged CA shift of residue 1

References:
===========
• Lipari, G. & Szabo, A. (1982) J. Am. Chem. Soc. 104, 4546–4559.
  (Model-free approach and definition of S²)
• Clore, G.M. et al. (1990) J. Am. Chem. Soc. 112, 4989–4991.
  (S² from trajectory: C(∞) = |<μ>|²)
• Solomon, I. (1955) Phys. Rev. 99, 559–565.
  (NOE cross-relaxation rate ∝ r⁻⁶)
• Bax, A. (2003) Protein Sci. 12, 1–16.
  (Review: using NMR to validate MD simulations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import biotite.structure as struc

logger = logging.getLogger(__name__)


# ── Type aliases ──────────────────────────────────────────────────────────────

# A single frame's shift data: {res_id: {atom_name: shift_ppm}}
FrameShifts = Dict[int, Dict[str, float]]

# A single frame's NOE data: {(res_i, res_j): distance_angstrom}
FrameNoes = Dict[Tuple[int, int], float]

# A single frame's RDC data: {res_id: rdc_hz}
FrameRdcs = Dict[int, float]


# ══════════════════════════════════════════════════════════════════════════════
# TrajectoryEnsemble
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrajectoryEnsemble:
    """
    A container for an ordered sequence of protein structure frames.

    Each frame is a biotite ``AtomArray`` representing one snapshot of the
    protein conformation — either from an MD simulation, an NMR conformational
    ensemble from the PDB, or any other source of multiple structures.

    EDUCATIONAL NOTE — What is a trajectory?
    ========================================
    In molecular dynamics, a trajectory is a time-ordered sequence of
    coordinate sets (frames) obtained by numerically integrating Newton's
    equations of motion.  Each frame is separated by a fixed time step
    (typically 1–2 fs).  Frames are usually saved every ~1 ps, so a 100-ns
    simulation produces ~100,000 frames.

    For NMR observables, we typically do NOT need every frame.  Saving
    every 10th–100th frame (called "striding") is sufficient because NMR
    observables are smooth functions of conformation.

    The ``TrajectoryEnsemble`` is intentionally agnostic to the source:
      • MD trajectories (GROMACS, AMBER, NAMD, OpenMM)
      • NMR conformational ensembles (PDB with multiple MODELs)
      • Synthetic ensembles (e.g., from short MD runs of AlphaFold structures)

    Parameters
    ----------
    frames : list of struc.AtomArray
        Ordered list of structure snapshots.  Must contain at least one frame.
        All frames should have the same atom topology (same atoms in the same
        order) for meaningful ensemble averaging.
    """

    frames: List[struc.AtomArray] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Validate: must have at least one frame
        if len(self.frames) == 0:
            raise ValueError(
                "TrajectoryEnsemble requires at least one frame.  "
                "An empty ensemble has no physical meaning."
            )
        # Validate: every element must be a biotite AtomArray
        for i, frame in enumerate(self.frames):
            if not isinstance(frame, struc.AtomArray):
                raise TypeError(
                    f"Frame {i} is not a biotite.structure.AtomArray "
                    f"(got {type(frame).__name__}).  "
                    "Each frame must be an AtomArray snapshot."
                )

    def __len__(self) -> int:
        """Return the number of frames in the ensemble."""
        return len(self.frames)

    def __iter__(self) -> Iterator[struc.AtomArray]:
        """Iterate over frames in order."""
        return iter(self.frames)

    def __repr__(self) -> str:
        n_atoms = self.frames[0].array_length() if self.frames else 0
        return f"TrajectoryEnsemble(" f"n_frames={len(self)}, " f"n_atoms_per_frame={n_atoms})"


# ══════════════════════════════════════════════════════════════════════════════
# load_trajectory
# ══════════════════════════════════════════════════════════════════════════════


def load_trajectory(
    source: Union[List[struc.AtomArray], Any],
    topology: Optional[str] = None,
    stride: int = 1,
) -> TrajectoryEnsemble:
    """
    Load a trajectory into a TrajectoryEnsemble from various sources.

    This is the main entry point for creating an ensemble.  Two code paths:

    **Path A — Plain Python list of AtomArrays (always available):**
    Pass a ``list`` of biotite ``AtomArray`` objects.  No MDTraj required.
    This is the code path used internally by all tests and is the most
    portable option.

    **Path B — MDTraj Trajectory object or file path (requires MDTraj):**
    Pass an MDTraj ``Trajectory`` object, or a string path to a trajectory
    file (e.g. ``md.xtc``) together with a ``topology`` path.  MDTraj handles
    all the heavy lifting of format conversion and coordinate extraction;
    each MDTraj frame is converted to a biotite ``AtomArray`` automatically.

    Parameters
    ----------
    source : list of struc.AtomArray, or MDTraj Trajectory, or str (file path)
        The trajectory source.
    topology : str, optional
        Path to a topology file (PDB, .prmtop, etc.).  Required when
        ``source`` is a file path string.
    stride : int, optional
        Load every ``stride``-th frame.  Default is 1 (all frames).
        Useful for large trajectories where full sampling is unnecessary.

    Returns
    -------
    TrajectoryEnsemble

    Raises
    ------
    ValueError
        If ``source`` is an empty list, or if ``stride`` < 1.
    TypeError
        If ``source`` is an unrecognised type.
    ImportError
        If ``source`` requires MDTraj but MDTraj is not installed.

    Examples
    --------
    >>> # From a list of AtomArrays (no MDTraj required)
    >>> frames = [load_structure(f"frame_{i}.pdb") for i in range(100)]
    >>> ensemble = load_trajectory(frames)

    >>> # From an GROMACS .xtc file (requires MDTraj)
    >>> ensemble = load_trajectory("md.xtc", topology="protein.pdb")
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    # ── Path A: plain list of AtomArrays ─────────────────────────────────────
    if isinstance(source, list):
        if len(source) == 0:
            raise ValueError(
                "load_trajectory received an empty list.  " "Provide at least one AtomArray frame."
            )
        frames: List[struc.AtomArray] = list(source)[::stride]
        logger.info(
            f"load_trajectory: loaded {len(frames)} frames "
            f"(stride={stride}) from AtomArray list."
        )
        return TrajectoryEnsemble(frames=frames)

    # ── Path B: MDTraj Trajectory object ─────────────────────────────────────
    if isinstance(source, str):
        # File-path string: load via MDTraj
        try:
            import mdtraj  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "Loading a trajectory from a file path requires MDTraj, "
                "which is not currently installed.  Install it with:\n\n"
                "    pip install synth-nmr[trajectory]\n\n"
                "Alternatively, convert your trajectory to a list of "
                "biotite AtomArrays and pass the list directly."
            )
        if topology is None:
            raise ValueError(
                "A topology file path must be provided via the `topology` "
                "argument when loading from a file path."
            )
        mdtraj_traj = mdtraj.load(source, top=topology, stride=stride)
        frames = _mdtraj_to_atomarrays(mdtraj_traj)
        logger.info(
            f"load_trajectory: loaded {len(frames)} frames " f"(stride={stride}) from '{source}'."
        )
        return TrajectoryEnsemble(frames=frames)

    # Check if source is an MDTraj Trajectory object (without hard-importing mdtraj)
    source_type_name = type(source).__module__ + "." + type(source).__qualname__
    if source_type_name.startswith("mdtraj"):
        raw_frames = _mdtraj_to_atomarrays(source)
        mdtraj_frames: List[struc.AtomArray] = list(raw_frames)[::stride]
        logger.info(
            f"load_trajectory: loaded {len(mdtraj_frames)} frames "
            f"from MDTraj Trajectory object."
        )
        return TrajectoryEnsemble(frames=mdtraj_frames)

    raise TypeError(
        f"Unrecognised source type: {type(source).__name__}.  "
        "Pass a list of biotite AtomArrays, a file path string, "
        "or an MDTraj Trajectory object."
    )


def _mdtraj_to_atomarrays(traj: Any) -> List[struc.AtomArray]:
    """
    Convert an MDTraj ``Trajectory`` to a list of biotite ``AtomArray`` objects.

    EDUCATIONAL NOTE — Coordinate units:
    =====================================
    MDTraj stores coordinates in NANOMETRES (nm), while biotite and the rest
    of synth-nmr use ANGSTROMS (Å).  We multiply by 10 on conversion.

    MDTraj topology encodes atom names, residue names, residue IDs, and
    chain identifiers.  We map these directly onto biotite AtomArray fields.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The MDTraj trajectory to convert.

    Returns
    -------
    list of struc.AtomArray
        One AtomArray per frame.
    """
    frames: List[struc.AtomArray] = []
    topology = traj.topology
    n_atoms = topology.n_atoms

    # Build per-atom static fields once (they are constant across frames)
    atom_names = np.array([a.name for a in topology.atoms])
    res_names = np.array([a.residue.name for a in topology.atoms])
    res_ids = np.array([a.residue.resSeq for a in topology.atoms])
    chain_ids = np.array([a.residue.chain.chain_id for a in topology.atoms])
    elements = np.array([a.element.symbol for a in topology.atoms])

    for frame_idx in range(traj.n_frames):
        arr = struc.AtomArray(n_atoms)
        arr.atom_name = atom_names
        arr.res_name = res_names
        arr.res_id = res_ids
        arr.chain_id = chain_ids
        arr.element = elements
        # MDTraj xyz is shape (n_frames, n_atoms, 3) in nm → convert to Å
        arr.coord = traj.xyz[frame_idx] * 10.0
        frames.append(arr)

    return frames


# ══════════════════════════════════════════════════════════════════════════════
# ensemble_average_shifts
# ══════════════════════════════════════════════════════════════════════════════


def ensemble_average_shifts(
    per_frame_shifts: List[FrameShifts],
) -> FrameShifts:
    """
    Compute time-averaged chemical shifts from a list of per-frame shift
    dictionaries.

    PHYSICS — Fast-exchange time averaging:
    ========================================
    In a solution NMR experiment, the protein samples many conformations
    during the data acquisition time (seconds to hours).  As long as
    conformational exchange is fast compared to the difference in Larmor
    frequencies between conformations (the "fast exchange limit"), the
    NMR spectrum shows a SINGLE peak positioned at the population-weighted
    average chemical shift.

    For a Boltzmann ensemble that we approximate with equal-weight MD frames:

        δ_obs(nucleus) = (1/N) Σ_{i=1}^{N}  δ_i(nucleus)

    This is a simple arithmetic mean — appropriate for chemical shifts because
    they depend on the instantaneous electron density distribution, which
    responds instantaneously to the nuclear coordinates.

    Note on missing nuclei:
    ========================
    If a nucleus is predicted in only a subset of frames (e.g. SPARTA+ cannot
    assign a shift in a disordered frame with poorly defined backbone), it is
    EXCLUDED from the average rather than averaged over fewer frames.  This
    prevents silent bias: an average over half the frames would not represent
    the same observable as an average over all frames.

    Parameters
    ----------
    per_frame_shifts : list of dict
        Each element is a dict of the form ``{res_id: {atom_name: shift_ppm}}``,
        as returned by ``predict_chemical_shifts()`` or ``predict_empirical_shifts()``.

    Returns
    -------
    dict
        ``{res_id: {atom_name: mean_shift_ppm}}`` — the ensemble-averaged shifts.

    Raises
    ------
    ValueError
        If the input list is empty.
    TypeError
        If the input is not a list.

    Examples
    --------
    >>> per_frame = [predict_chemical_shifts(f) for f in ensemble]
    >>> avg = ensemble_average_shifts(per_frame)
    >>> print(avg[1]["CA"])  # mean CA shift for residue 1
    """
    if not isinstance(per_frame_shifts, list):
        raise TypeError(f"per_frame_shifts must be a list, got {type(per_frame_shifts).__name__}.")
    if len(per_frame_shifts) == 0:
        raise ValueError(
            "per_frame_shifts must contain at least one frame.  "
            "An empty list has no physical meaning."
        )

    n_frames = len(per_frame_shifts)

    # Step 1: Collect all (res_id, atom_name) pairs that appear in EVERY frame.
    # We build a set of keys present in each frame and intersect.
    #
    # Implementation note: we use a defaultdict to accumulate values, then
    # filter to only those keys seen in all n_frames frames.

    # {(res_id, atom_name): list of float}
    accumulator: Dict[Tuple[int, str], List[float]] = {}

    for frame_dict in per_frame_shifts:
        for res_id, nucleus_dict in frame_dict.items():
            for atom_name, shift_val in nucleus_dict.items():
                key = (res_id, atom_name)
                if key not in accumulator:
                    accumulator[key] = []
                accumulator[key].append(float(shift_val))

    # Step 2: Compute mean only for keys present in every frame
    result: FrameShifts = {}
    for (res_id, atom_name), values in accumulator.items():
        if len(values) == n_frames:
            # Present in every frame → include in average
            if res_id not in result:
                result[res_id] = {}
            result[res_id][atom_name] = float(np.mean(values))
        else:
            logger.debug(
                f"Nucleus ({res_id}, {atom_name}) present in {len(values)}/{n_frames} "
                "frames — excluded from ensemble average to avoid bias."
            )

    n_residues = len(result)
    logger.info(
        f"ensemble_average_shifts: averaged {n_frames} frames, " f"retained {n_residues} residues."
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ensemble_average_noes
# ══════════════════════════════════════════════════════════════════════════════


def ensemble_average_noes(
    per_frame_noes: List[FrameNoes],
) -> FrameNoes:
    """
    Compute ensemble-averaged effective NOE distances using the r⁻⁶ average.

    PHYSICS — The sixth-power average (Solomon 1955):
    ==================================================
    The Nuclear Overhauser Effect (NOE) arises from cross-relaxation between
    two nuclear spins mediated by their through-space dipolar coupling.  The
    cross-relaxation rate σ_ij between protons i and j is:

        σ_ij  ∝  <r_ij⁻⁶>_t

    where <·>_t denotes time averaging.  The EFFECTIVE distance extracted
    from a NOESY peak volume (which is proportional to σ_ij) is:

        r_eff  =  <r_ij⁻⁶>^(-1/6)
                =  [ (1/N) Σ_k  r_k⁻⁶ ]^(-1/6)

    This is called the "sixth-power average" or "r⁻⁶ average".

    WHY THIS MATTERS:  The r⁻⁶ average is always SHORTER than the arithmetic
    mean distance when distances vary.  The NOE is dominated by the closest
    approach — a single close-contact conformation can produce a large NOE
    even if the average distance is long.  Using a simple arithmetic mean
    would systematically over-estimate NOE distances and produce incorrect
    structural restraints.

    Example:
      Frame 1: r = 2 Å  →  r⁻⁶ = 0.015625 Å⁻⁶
      Frame 2: r = 4 Å  →  r⁻⁶ = 0.000244 Å⁻⁶
      Arithmetic mean distance:  3.0 Å
      r⁻⁶ mean:  0.007935 Å⁻⁶
      r_eff = 0.007935^(-1/6)  = 2.27 Å  ← significantly shorter than 3.0 Å

    The missing-key policy (only average pairs present in all frames) is the
    same as in ``ensemble_average_shifts``.

    Parameters
    ----------
    per_frame_noes : list of dict
        Each element is ``{(res_i, res_j): distance_angstrom}`` — one dict
        per trajectory frame, as returned by ``calculate_synthetic_noes()``.

    Returns
    -------
    dict
        ``{(res_i, res_j): r_eff_angstrom}`` — the ensemble-averaged effective
        NOE distances.

    Raises
    ------
    ValueError
        If the input list is empty.

    Examples
    --------
    >>> per_frame = [calculate_synthetic_noes(f, cutoff=5.0) for f in ensemble]
    >>> avg_noes = ensemble_average_noes(per_frame)
    >>> print(avg_noes[(1, 5)])  # effective distance between residues 1 and 5
    """
    if len(per_frame_noes) == 0:
        raise ValueError(
            "per_frame_noes must contain at least one frame.  "
            "An empty list has no physical meaning."
        )

    n_frames = len(per_frame_noes)

    # Accumulate r⁻⁶ values for each atom pair
    # {pair: list of r⁻⁶ values}
    accumulator: Dict[Tuple[int, int], List[float]] = {}

    for frame_dict in per_frame_noes:
        for pair, dist in frame_dict.items():
            if pair not in accumulator:
                accumulator[pair] = []
            # r⁻⁶ accumulation — this is the key physics step
            accumulator[pair].append(dist**-6)

    # Compute r_eff = <r⁻⁶>^(-1/6) only for pairs observed in every frame
    result: FrameNoes = {}
    for pair, r6_values in accumulator.items():
        if len(r6_values) == n_frames:
            mean_r6 = float(np.mean(r6_values))
            r_eff = mean_r6 ** (-1.0 / 6.0)
            result[pair] = r_eff
        else:
            logger.debug(
                f"NOE pair {pair} present in {len(r6_values)}/{n_frames} "
                "frames — excluded from ensemble average."
            )

    logger.info(
        f"ensemble_average_noes: averaged {n_frames} frames, " f"retained {len(result)} NOE pairs."
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ensemble_average_rdcs
# ══════════════════════════════════════════════════════════════════════════════


def ensemble_average_rdcs(
    per_frame_rdcs: List[FrameRdcs],
) -> FrameRdcs:
    """
    Compute time-averaged Residual Dipolar Couplings (RDCs) from a list of
    per-frame RDC dictionaries.

    PHYSICS — Motional averaging of RDCs:
    ======================================
    The RDC for a bond vector μ in an alignment medium is:

        D(μ) = D_a · [ (3cos²θ − 1) + (3/2)R · sin²θ · cos(2φ) ]

    where θ and φ are the polar and azimuthal angles of μ in the alignment
    tensor Principal Axis System (PAS).

    In the fast-exchange limit (dynamics faster than the magnitude of the
    RDC ≈ Hz), the observed RDC is the time-average:

        D_obs = <D(μ(t))>_t  =  (1/N) Σ_k  D(μ_k)

    This arithmetic mean is the correct averaging for RDCs, analogous to
    chemical shifts.  (It differs from NOEs, which require r⁻⁶ averaging.)

    Note on the relationship between S² and RDC dynamics:
    ======================================================
    For small-amplitude bond-vector fluctuations, the averaged RDC is
    approximately:
        D_obs ≈ S² · D_rigid
    where D_rigid is the RDC for the average bond orientation and S² is the
    Lipari-Szabo order parameter.  This is the basis of RDC-based S² estimation.
    The ``ensemble_average_rdcs`` function computes the exact frame-by-frame
    average rather than this approximation.

    Parameters
    ----------
    per_frame_rdcs : list of dict
        Each element is ``{res_id: rdc_hz}`` — one dict per trajectory frame,
        as returned by ``calculate_rdcs()``.

    Returns
    -------
    dict
        ``{res_id: mean_rdc_hz}``

    Raises
    ------
    ValueError
        If the input list is empty.

    Examples
    --------
    >>> per_frame = [calculate_rdcs(f, Da=10.0, R=0.5) for f in ensemble]
    >>> avg_rdcs = ensemble_average_rdcs(per_frame)
    >>> print(avg_rdcs[1])  # mean N-H RDC for residue 1
    """
    if len(per_frame_rdcs) == 0:
        raise ValueError(
            "per_frame_rdcs must contain at least one frame.  "
            "An empty list has no physical meaning."
        )

    n_frames = len(per_frame_rdcs)

    # Accumulate RDC values per residue
    accumulator: Dict[int, List[float]] = {}
    for frame_dict in per_frame_rdcs:
        for res_id, rdc_val in frame_dict.items():
            if res_id not in accumulator:
                accumulator[res_id] = []
            accumulator[res_id].append(float(rdc_val))

    # Arithmetic mean for residues seen in every frame
    result: FrameRdcs = {}
    for res_id, values in accumulator.items():
        if len(values) == n_frames:
            result[res_id] = float(np.mean(values))
        else:
            logger.debug(
                f"RDC residue {res_id} present in {len(values)}/{n_frames} "
                "frames — excluded from ensemble average."
            )

    logger.info(
        f"ensemble_average_rdcs: averaged {n_frames} frames, " f"retained {len(result)} residues."
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# compute_s2_from_trajectory
# ══════════════════════════════════════════════════════════════════════════════


def compute_s2_from_trajectory(
    ensemble: TrajectoryEnsemble,
) -> Dict[int, float]:
    """
    Compute the Lipari-Szabo generalized order parameter S² for backbone
    N-H bond vectors directly from a trajectory ensemble.

    PHYSICS — What is S²?
    ======================
    The order parameter S² (0 ≤ S² ≤ 1) quantifies the spatial restriction
    of a bond vector's motion on the ps-ns timescale:

        S² = 1.0 → perfectly rigid bond (no internal motion)
        S² = 0.0 → isotropically disordered bond (glass-like motion)
        S² ≈ 0.85 → typical well-ordered backbone amide

    DERIVATION FROM A TRAJECTORY:
    ==============================
    The time-correlation function of a unit bond vector μ(t) is:

        C(τ) = <P₂(μ(0) · μ(τ))>

    where P₂ is the second Legendre polynomial.  In the Lipari-Szabo
    model-free framework, assuming independence of overall tumbling and
    internal motion:

        C(τ) → S²  as τ → ∞

    For an MD trajectory with many frames, the plateau value S² is estimated
    directly from the squared magnitude of the MEAN unit vector:

        S² = |<μ>|²  =  (<μ_x>² + <μ_y>² + <μ_z>²)

    INTUITION:
      • If μ is identical in every frame (rigid), <μ> equals that unit
        vector, and |<μ>|² = 1.
      • If μ points in a completely random direction each frame (disordered),
        <μ> ≈ 0 (components cancel), and |<μ>|² ≈ 0.
      • Partial restriction gives intermediate values.

    This is exact under the Lipari-Szabo framework and requires no model
    fitting — it is a direct measurement from the trajectory.  It is
    equivalent to the order parameter obtained from the long-time plateau
    of the reorientational autocorrelation function.

    References:
    -----------
    • Lipari, G. & Szabo, A. (1982) J. Am. Chem. Soc. 104, 4546.
    • Clore, G.M., Szabo, A., Bax, A., et al. (1990) J. Am. Chem. Soc.
      112, 4989.

    PRACTICAL NOTE — Overall tumbling removal:
    ===========================================
    For this estimate to be valid, the OVERALL rotation of the protein must
    be removed from the trajectory (i.e., all frames must be aligned to a
    common reference orientation).  If MDTraj is used for loading, RMSD
    fitting removes overall translation; rotational superposition should be
    applied before calling this function.  For equal-weight ensembles from
    PDB models, overall tumbling is already absent.

    Parameters
    ----------
    ensemble : TrajectoryEnsemble
        The trajectory ensemble.  Must contain backbone N and H atoms.

    Returns
    -------
    dict
        ``{res_id: s2_value}`` — one S² value per residue with a detectable
        N-H bond vector.  Proline residues (no amide proton) are excluded.
        Returns an empty dict if no N-H pairs are found.

    Examples
    --------
    >>> s2 = compute_s2_from_trajectory(ensemble)
    >>> for res_id, s2_val in sorted(s2.items()):
    ...     print(f"Residue {res_id}: S² = {s2_val:.3f}")
    """
    # Build a dict {res_id: list of unit NH vectors (one per frame)}
    # We accumulate numpy arrays because we need the vector mean, not a scalar.

    # {res_id: list of np.ndarray shape (3,)}
    nh_vectors: Dict[int, List[np.ndarray]] = {}

    for frame in ensemble:
        # Extract backbone N and H atoms; build a fast residue→coord lookup
        n_mask = frame.atom_name == "N"
        h_mask = frame.atom_name == "H"

        n_atoms = frame[n_mask]
        h_atoms = frame[h_mask]

        if n_atoms.array_length() == 0 or h_atoms.array_length() == 0:
            # This frame has no N-H pairs — skip it silently
            continue

        # Build a lookup: res_id -> H coord (for amide proton matching)
        h_coord_map: Dict[int, np.ndarray] = {int(h.res_id): np.asarray(h.coord) for h in h_atoms}

        for n_atom in n_atoms:
            res_id = int(n_atom.res_id)

            # Skip Proline: no backbone amide proton (N is tertiary amine)
            # This mirrors the exclusion rule in rdc.py and relaxation.py
            if n_atom.res_name == "PRO":
                continue

            if res_id not in h_coord_map:
                # Amide H not found for this residue in this frame
                continue

            # Compute the N→H vector and normalise to a unit vector
            nh_vec = h_coord_map[res_id] - np.asarray(n_atom.coord)
            norm = np.linalg.norm(nh_vec)
            if norm < 1e-9:
                # Zero-length vector: degenerate geometry, skip this frame/residue
                logger.warning(
                    f"Residue {res_id}: zero-length N-H vector in one frame, "
                    "skipping this (frame, residue) pair."
                )
                continue

            unit_vec = nh_vec / norm

            if res_id not in nh_vectors:
                nh_vectors[res_id] = []
            nh_vectors[res_id].append(unit_vec)

    # Compute S² = |<μ>|² for each residue
    #
    # EDUCATIONAL NOTE — Why |<μ>|² and not <|μ|²>?
    # ================================================
    # |<μ>|² is the squared magnitude of the VECTOR MEAN.
    # <|μ|²> = 1 always (we always use unit vectors), so that is trivially useless.
    # The key insight is that the VECTOR mean captures directionality:
    #   - rigid vector: all μ_i point the same way → |<μ>| = 1
    #   - random vectors: Σμ_i ≈ 0 → |<μ>| ≈ 0

    result: Dict[int, float] = {}

    for res_id, vec_list in nh_vectors.items():
        mu_mean = np.mean(np.stack(vec_list, axis=0), axis=0)  # shape (3,)
        s2 = float(np.dot(mu_mean, mu_mean))  # |<μ>|² = dot product with itself
        # Clamp to [0, 1] to correct for tiny floating-point overshoots
        s2 = float(np.clip(s2, 0.0, 1.0))
        result[res_id] = s2

    if not result:
        logger.warning(
            "compute_s2_from_trajectory: no N-H bond vectors found in ensemble.  "
            "Ensure the structure contains backbone 'N' and 'H' atoms."
        )
    else:
        logger.info(
            f"compute_s2_from_trajectory: computed S² for {len(result)} residues "
            f"over {len(ensemble)} frames."
        )

    return result
