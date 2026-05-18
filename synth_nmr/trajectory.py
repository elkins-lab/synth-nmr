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
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)


# ── Type aliases ──────────────────────────────────────────────────────────────

# A single frame's shift data: {res_id: {atom_name: shift_ppm}}
FrameShifts = Dict[int, Dict[str, float]]

# A single frame's NOE data: {(res_i, res_j): distance_angstrom}
FrameNoes = Dict[Tuple[int, int], float]

# A single frame's RDC data: {res_id: rdc_hz}
FrameRdcs = Dict[int, float]

# A single frame's J-coupling data: {chain_id: {res_id: j_hz}}
FrameJCouplings = Dict[str, Dict[int, float]]


# ══════════════════════════════════════════════════════════════════════════════
# TrajectoryEnsemble
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrajectoryEnsemble:
    """
    A container for an ordered sequence of protein structure frames.

    Encapsulates a biotite ``AtomArrayStack``, where the first dimension
    represents time (frames) and subsequent dimensions represent atoms and
    coordinates.

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

    Using an ``AtomArrayStack`` instead of a list of ``AtomArray``s is much
    more memory-efficient for large trajectories because the topology
    (atom names, residue IDs, etc.) is stored only ONCE for the whole
    ensemble.

    Parameters
    ----------
    stack : struc.AtomArrayStack
        The stack of structure snapshots. Must have at least one frame.
    """

    stack: struc.AtomArrayStack

    # ── Educational Note: The Advantage of AtomArrayStack ────────────────────
    # In earlier versions of synth-nmr, a TrajectoryEnsemble was stored as a
    # simple Python list of AtomArray objects. While intuitive, this had
    # two major drawbacks:
    #
    # 1. Memory Overhead: Each AtomArray stores its own copy of the topology
    #    (atom names, residue IDs, chain IDs, elements). In a 100,000 frame
    #    trajectory, this is 100,000 redundant copies of the same strings and
    #    integers.
    #
    # 2. Performance Bottleneck: To calculate an ensemble average, we had to
    #    loop over the list in Python, extract coordinates, and perform
    #    arithmetic. This "for-loop" in Python is orders of magnitude slower
    #    than optimized C or Fortran code.
    #
    # By switching to biotite.structure.AtomArrayStack, we solve both:
    # - Topology is stored ONCE for all frames (Global arrays).
    # - Coordinates are stored in a single 3D NumPy array (frames, atoms, 3).
    # - We can use NumPy vectorization to perform math across all frames
    #   at once (e.g., np.mean(stack.coord, axis=0)).
    # ─────────────────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # Validate: must have at least one frame
        if self.stack.stack_depth() == 0:
            raise ValueError(
                "TrajectoryEnsemble requires at least one frame.  "
                "An empty ensemble has no physical meaning."
            )

    def __len__(self) -> int:
        """Return the number of frames in the ensemble."""
        return self.stack.stack_depth()

    def __getitem__(self, index: int | slice) -> struc.AtomArray | TrajectoryEnsemble:
        """
        Get a frame or a sub-ensemble.

        Parameters
        ----------
        index : int or slice
            The frame index or range to retrieve.

        Returns
        -------
        biotite.structure.AtomArray or TrajectoryEnsemble
            A single frame if `index` is an int, or a new TrajectoryEnsemble
            containing the specified range if `index` is a slice.
        """
        if isinstance(index, slice):
            return TrajectoryEnsemble(stack=self.stack[index])
        return self.stack[index]

    def __iter__(self) -> Iterator[struc.AtomArray]:
        """Iterate over frames in order."""
        return iter(self.stack)

    def __repr__(self) -> str:
        n_atoms = self.stack.array_length()
        return f"TrajectoryEnsemble(n_frames={len(self)}, n_atoms_per_frame={n_atoms})"


# ══════════════════════════════════════════════════════════════════════════════
# load_trajectory
# ══════════════════════════════════════════════════════════════════════════════


def load_trajectory(
    source: list[struc.AtomArray] | struc.AtomArrayStack | Any,
    topology: str | None = None,
    stride: int = 1,
) -> TrajectoryEnsemble:
    """
    Load a trajectory into a TrajectoryEnsemble from various sources.

    This is the main entry point for creating an ensemble.  Three code paths:

    **Path A — Plain Python list of AtomArrays or an AtomArrayStack:**
    Pass a ``list`` of biotite ``AtomArray`` objects or a single
    ``AtomArrayStack``.

    **Path B — MDTraj Trajectory object or file path (requires MDTraj):**
    Pass an MDTraj ``Trajectory`` object, or a string path to a trajectory
    file (e.g. ``md.xtc``) together with a ``topology`` path.

    Parameters
    ----------
    source : list of struc.AtomArray, AtomArrayStack, or MDTraj Trajectory, or str (file path)
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
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    # ── Path A1: AtomArrayStack ──────────────────────────────────────────────
    if isinstance(source, struc.AtomArrayStack):
        stack = source[::stride]
        logger.info(
            f"load_trajectory: loaded {stack.stack_depth()} frames (stride={stride}) "
            "from AtomArrayStack."
        )
        return TrajectoryEnsemble(stack=stack)

    # ── Path A2: plain list of AtomArrays ────────────────────────────────────
    if isinstance(source, list):
        if len(source) == 0:
            raise ValueError(
                "load_trajectory received an empty list.  Provide at least one AtomArray frame."
            )
        # Convert list to AtomArrayStack
        stack = struc.stack(source[::stride])
        logger.info(
            f"load_trajectory: loaded {stack.stack_depth()} frames (stride={stride}) "
            "from AtomArray list."
        )
        return TrajectoryEnsemble(stack=stack)

    # ── Path B: MDTraj ───────────────────────────────────────────────────────
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
        stack = _mdtraj_to_stack(mdtraj_traj)
        logger.info(f"load_trajectory: loaded {stack.stack_depth()} frames from '{source}'.")
        return TrajectoryEnsemble(stack=stack)

    # Check if source is an MDTraj Trajectory object (without hard-importing mdtraj)
    source_type_name = type(source).__module__ + "." + type(source).__qualname__
    if source_type_name.startswith("mdtraj"):
        raw_stack = _mdtraj_to_stack(source)
        stack = raw_stack[::stride]
        logger.info(
            f"load_trajectory: loaded {stack.stack_depth()} frames from MDTraj Trajectory object."
        )
        return TrajectoryEnsemble(stack=stack)

    raise TypeError(
        f"Unrecognised source type: {type(source).__name__}.  "
        "Pass a list of biotite AtomArrays, an AtomArrayStack, a file path string, "
        "or an MDTraj Trajectory object."
    )


def _mdtraj_to_stack(traj: Any) -> struc.AtomArrayStack:
    """
    Convert an MDTraj ``Trajectory`` to a biotite ``AtomArrayStack``.

    EDUCATIONAL NOTE — Coordinate units:
    =====================================
    MDTraj stores coordinates in NANOMETRES (nm), while biotite and the rest
    of synth-nmr use ANGSTROMS (Å).  We multiply by 10 on conversion.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The MDTraj trajectory to convert.

    Returns
    -------
    struc.AtomArrayStack
    """
    topology = traj.topology
    n_atoms = topology.n_atoms
    n_frames = traj.n_frames

    # Build per-atom static fields once
    atom_names = np.array([a.name for a in topology.atoms])
    res_names = np.array([a.residue.name for a in topology.atoms])
    res_ids = np.array([a.residue.resSeq for a in topology.atoms])
    chain_ids = np.array([a.residue.chain.chain_id for a in topology.atoms])
    elements = np.array([a.element.symbol for a in topology.atoms])

    stack = struc.AtomArrayStack(n_frames, n_atoms)
    stack.atom_name = atom_names
    stack.res_name = res_names
    stack.res_id = res_ids
    stack.chain_id = chain_ids
    stack.element = elements
    # MDTraj xyz is shape (n_frames, n_atoms, 3) in nm → convert to Å
    stack.coord = traj.xyz * 10.0

    return stack


# ══════════════════════════════════════════════════════════════════════════════
# ensemble_average_shifts
# ══════════════════════════════════════════════════════════════════════════════


def ensemble_average_shifts(
    per_frame_shifts: list[FrameShifts],
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
    # Implementation note: we use a dictionary to accumulate values for each
    # nucleus. We only average those that are consistently present across
    # all frames to avoid introducing statistical bias from missing data.

    # {(res_id, atom_name): list of float}
    accumulator: dict[tuple[int, str], list[float]] = {}

    for frame_dict in per_frame_shifts:
        for res_id, nucleus_dict in frame_dict.items():
            for atom_name, shift_val in nucleus_dict.items():
                key = (res_id, atom_name)
                if key not in accumulator:
                    accumulator[key] = []
                accumulator[key].append(float(shift_val))

    # Step 2: Compute mean only for keys present in every frame
    #
    # PHYSICS RECAP: Chemical shifts respond to the local electronic environment.
    # In the fast-exchange regime, the nucleus "sees" the average environment.
    # This is modeled by a simple arithmetic mean of the instantaneous shifts.
    result: FrameShifts = {}
    for (res_id, atom_name), values in accumulator.items():
        if len(values) == n_frames:
            # Present in every frame → include in average
            if res_id not in result:
                result[res_id] = {}
            # Use vectorized NumPy mean for efficiency. Even though we are
            # iterating in Python, NumPy handles the numerical sum/division
            # in optimized C code.
            result[res_id][atom_name] = float(np.mean(np.array(values, dtype=np.float64)))
        else:
            logger.debug(
                f"Nucleus ({res_id}, {atom_name}) present in {len(values)}/{n_frames} "
                "frames — excluded from ensemble average to avoid bias."
            )

    n_residues = len(result)
    logger.info(
        f"ensemble_average_shifts: averaged {n_frames} frames, retained {n_residues} residues."
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ensemble_average_noes
# ══════════════════════════════════════════════════════════════════════════════


def ensemble_average_noes(
    per_frame_noes: list[FrameNoes],
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
    accumulator: dict[tuple[int, int], list[float]] = {}

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
            # Use vectorized NumPy mean and power for efficiency
            mean_r6 = float(np.mean(np.array(r6_values, dtype=np.float64)))
            r_eff = mean_r6 ** (-1.0 / 6.0)
            result[pair] = r_eff
        else:
            logger.debug(
                f"NOE pair {pair} present in {len(r6_values)}/{n_frames} "
                "frames — excluded from ensemble average."
            )

    logger.info(
        f"ensemble_average_noes: averaged {n_frames} frames, retained {len(result)} NOE pairs."
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ensemble_average_rdcs
# ══════════════════════════════════════════════════════════════════════════════


def ensemble_average_rdcs(
    per_frame_rdcs: list[FrameRdcs],
) -> FrameRdcs:
    """
    Compute time-averaged Residual Dipolar Couplings (RDCs) from a list of
    per-frame RDC dictionaries.

    PHYSICS — Motional averaging of RDCs:
    ======================================
    The RDC for a bond vector μ in an alignment medium is a reporting of
    the average orientation of that bond relative to the external magnetic
    field, as filtered by the alignment tensor of the protein.

    In the fast-exchange limit (dynamics faster than the magnitude of the
    RDC ≈ Hz), the observed RDC is the time-average:

        D_obs = <D(μ(t))>_t  =  (1/N) Σ_k  D(μ_k)

    This arithmetic mean is the correct averaging for RDCs, analogous to
    chemical shifts.  (It differs from NOEs, which require r⁻⁶ averaging.)

    Why arithmetic mean?
    ====================
    RDCs depend linearly on the order parameters of the alignment tensor.
    As long as the protein structure undergoes small-amplitude fluctuations
    around a mean state, and the alignment tensor remains constant (or also
    averages), the observed coupling is the direct average of the
    instantaneous values.

    Parameters
    ----------
    per_frame_rdcs : list of dict
        Each element is ``{res_id: rdc_hz}`` — one dict per trajectory frame,
        as returned by ``calculate_rdcs()``.

    Returns
    -------
    dict
        ``{res_id: mean_rdc_hz}`` — ensemble averaged RDCs.
    """
    if len(per_frame_rdcs) == 0:
        raise ValueError(
            "per_frame_rdcs must contain at least one frame.  "
            "An empty list has no physical meaning."
        )

    n_frames = len(per_frame_rdcs)

    # Accumulate RDC values per residue
    # We use a dictionary to collect values across all frames.
    # PHYSICS NOTE: RDCs can be positive or negative. The arithmetic mean
    # correctly preserves the sign and magnitude of the averaged alignment.
    accumulator: dict[int, list[float]] = {}
    for frame_dict in per_frame_rdcs:
        for res_id, rdc_val in frame_dict.items():
            if res_id not in accumulator:
                accumulator[res_id] = []
            accumulator[res_id].append(float(rdc_val))

    # Arithmetic mean for residues seen in every frame
    # We enforce consistent presence to ensure the average is representative
    # of the entire ensemble and doesn't suffer from sampling artifacts.
    result: FrameRdcs = {}
    for res_id, values in accumulator.items():
        if len(values) == n_frames:
            # Vectorized mean calculation via NumPy
            result[res_id] = float(np.mean(np.array(values, dtype=np.float64)))
        else:
            logger.debug(
                f"RDC residue {res_id} present in {len(values)}/{n_frames} "
                "frames — excluded from ensemble average."
            )

    logger.info(
        f"ensemble_average_rdcs: averaged {n_frames} frames, retained {len(result)} residues."
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# compute_s2_from_trajectory
# ══════════════════════════════════════════════════════════════════════════════


def compute_s2_from_trajectory(
    ensemble: TrajectoryEnsemble,
) -> dict[int, float]:
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
    S² = |<μ>|²  =  (<μ_x>² + <μ_y>² + <μ_z>²)

    This is exact under the Lipari-Szabo framework and requires no model
    fitting — it is a direct measurement from the trajectory.

    Note: The trajectory must be RMSD-aligned to a reference structure
    to remove overall tumbling before calling this function.

    Parameters
    ----------
    ensemble : TrajectoryEnsemble
        The trajectory ensemble. Must contain backbone N and H atoms.

    Returns
    -------
    dict
        ``{res_id: s2_value}``
    """
    stack = ensemble.stack
    # Select backbone N and H atoms
    n_mask = stack.atom_name == "N"
    h_mask = stack.atom_name == "H"

    # We need to find matching N-H pairs in the same residue
    # Filter for residues that have BOTH N and H
    n_indices = np.where(n_mask)[0]
    h_indices = np.where(h_mask)[0]

    # Map res_id to index for N and H
    n_res_ids = stack.res_id[n_indices]
    h_res_ids = stack.res_id[h_indices]

    # Find common res_ids
    common_res_ids = np.intersect1d(n_res_ids, h_res_ids)

    # Exclude Proline (no amide H)
    # Get res_names for N atoms
    n_res_names = stack.res_name[n_indices]
    pro_mask = n_res_names == "PRO"
    pro_res_ids = n_res_ids[pro_mask]
    common_res_ids = np.setdiff1d(common_res_ids, pro_res_ids)

    if common_res_ids.size == 0:
        logger.warning(
            "compute_s2_from_trajectory: no N-H bond vectors found in ensemble. "
            "Ensure the structure contains backbone 'N' and 'H' atoms."
        )
        return {}

    # Get final indices for paired N and H atoms
    # We use vectorized lookup for matching N and H indices.
    # This is slightly more complex but still much faster than per-frame loop.
    n_id_to_idx = {rid: idx for rid, idx in zip(n_res_ids, n_indices)}
    h_id_to_idx = {rid: idx for rid, idx in zip(h_res_ids, h_indices)}

    final_n_indices = np.array([n_id_to_idx[rid] for rid in common_res_ids])
    final_h_indices = np.array([h_id_to_idx[rid] for rid in common_res_ids])
    final_res_ids = [int(rid) for rid in common_res_ids]

    # ── Vectorized Calculation ───────────────────────────────────────────────
    # PERFORMANCE NOTE — The Power of Vectorization:
    # Instead of looping over frames and residues in Python, we perform
    # high-dimensional NumPy operations. For a 10,000 frame trajectory,
    # this is typically 100x faster.

    # stack.coord has shape (frames, atoms, 3)
    # Extract coordinates for all matched N and H atoms across all frames
    # Shape: (frames, pairs, 3)
    n_coords = stack.coord[:, final_n_indices, :]
    h_coords = stack.coord[:, final_h_indices, :]

    # Compute N->H vectors for all frames and all pairs at once
    # Shape: (frames, pairs, 3)
    nh_vecs = h_coords - n_coords

    # Compute norms (bond lengths) for all frames/pairs
    # Shape: (frames, pairs)
    norms = np.linalg.norm(nh_vecs, axis=2)

    # PHYSICS VALIDATION: Skip residues where ANY frame has a zero-length vector
    # (norm < 1e-9). These are degenerate geometries where the N and H are
    # placed at the same position. In such cases, the bond vector is
    # undefined, and normalising it would produce NaNs.
    valid_pair_mask = np.all(norms > 1e-9, axis=0)

    if not np.any(valid_pair_mask):
        logger.warning("compute_s2_from_trajectory: all N-H pairs have degenerate geometry.")
        return {}

    # Filter to only valid pairs that survive the quality check
    final_res_ids = [rid for i, rid in enumerate(final_res_ids) if valid_pair_mask[i]]
    nh_vecs = nh_vecs[:, valid_pair_mask, :]
    norms = norms[:, valid_pair_mask]

    # Normalize to unit vectors: μ = v / |v|
    # Shape: (frames, pairs, 3)
    unit_vecs = nh_vecs / norms[:, :, np.newaxis]

    # Compute mean vector across the frame dimension (axis 0)
    # <μ> = (1/N) Σ μ_i
    # Shape: (pairs, 3)
    mu_mean = np.mean(unit_vecs, axis=0)

    # Compute S² = |<μ>|² (squared magnitude of mean vector)
    # S² = <μ_x>² + <μ_y>² + <μ_z>²
    # Shape: (pairs,)
    s2_values = np.sum(mu_mean**2, axis=1)

    # Clamp to [0, 1] to prevent floating point noise from exceeding 1.0
    s2_values = np.clip(s2_values, 0.0, 1.0)

    result = {rid: float(val) for rid, val in zip(final_res_ids, s2_values)}

    logger.info(
        f"compute_s2_from_trajectory: computed S² for {len(result)} residues "
        f"over {stack.stack_depth()} frames using vectorized engine."
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ensemble_average_j_couplings
# ══════════════════════════════════════════════════════════════════════════════


def ensemble_average_j_couplings(
    per_frame_j: list[FrameJCouplings],
) -> FrameJCouplings:
    """
    Compute time-averaged J-couplings from a list of per-frame dictionaries.

    PHYSICS — Fast-exchange averaging of J-couplings:
    ================================================
    Scalar couplings (J-couplings) arise from the mediated interaction
    between nuclear spins via the bonding electrons. The observed coupling
    is extremely sensitive to the local dihedral angles (Karplus relationship).

    In the fast-exchange limit (where the timescale of conformational
    transitions is faster than the reciprocal of the coupling difference),
    the observed J-coupling is the simple arithmetic mean of the
    instantaneous values:

        J_obs = <J(theta(t))>_t = (1/N) Σ J(theta_i)

    This is valid because the Fermi contact interaction (which dominates
    J-coupling) depends on the electron spin density, which averages
    instantaneously over nuclear positions.

    Importance of Rotameric Averaging:
    ==================================
    Side-chain couplings (Ha-Hb, C'-Cg) depend on the chi1 angle. In solution,
    side-chains often jump between staggered rotamers (e.g., -60, 180, +60).
    The spectrometer does not see separate peaks for each rotamer; it sees
    a single peak at the weighted average position. This averaging
    correctly accounts for the populations of different rotameric states.

    Parameters
    ----------
    per_frame_j : list of dict
        Each element is {chain_id: {res_id: j_hz}} — the per-frame predictions.

    Returns
    -------
    dict
        {chain_id: {res_id: mean_j_hz}} — the ensemble-averaged values.
    """
    if not per_frame_j:
        raise ValueError("per_frame_j must contain at least one frame.")

    n_frames = len(per_frame_j)

    # Accumulate J-coupling values across the ensemble.
    # We use a tuple (chain_id, res_id) as the key for precise tracking.
    # PHYSICS NOTE: J-couplings are typically positive for 3-bond HN-HA
    # interactions, but can be negative in other cases. The arithmetic
    # mean preserves the correct physical average.
    accumulator: dict[tuple[str, int], list[float]] = {}

    for frame_dict in per_frame_j:
        for chain_id, res_dict in frame_dict.items():
            for res_id, j_val in res_dict.items():
                key = (chain_id, res_id)
                if key not in accumulator:
                    accumulator[key] = []
                accumulator[key].append(float(j_val))

    # Compute mean only for residue couplings that appear in every frame.
    # This prevents sampling bias from incomplete frame data.
    result: FrameJCouplings = {}
    for (chain_id, res_id), values in accumulator.items():
        if len(values) == n_frames:
            if chain_id not in result:
                result[chain_id] = {}
            # Use NumPy for efficient averaging of the accumulated data.
            # We must nest under chain_id to match the FrameJCouplings type.
            result[chain_id][res_id] = float(np.mean(np.array(values, dtype=np.float64)))

    return result
