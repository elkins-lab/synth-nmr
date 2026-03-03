# Ensemble NMR & Molecular Dynamics


!!! note "Who is this page for?"
    This page is written for biologists and chemists who may be new to NMR spectroscopy.  We introduce NMR concepts from scratch using physical intuition and analogies before diving into equations.  If you are already comfortable with fast exchange, order parameters, and NOE averaging, you can skip ahead to the [Practical Guide](#full-practical-guide).

---

## The Core Problem: Proteins are not statues

Imagine you take a protein crystal structure from the Protein Data Bank.  You have a set of atomic coordinates — a single, precise snapshot of exactly where every atom is.  If you use `synth-nmr` to calculate the predicted NMR spectrum from that snapshot, you get a perfectly valid prediction for that one conformation.

But there is a fundamental problem: **proteins in solution are not statues**.

In a test tube at 25 °C, a protein is constantly moving.  Individual bonds vibrate back and forth billions of times per second.  Side chains rotate.  Loops, tails, and disordered regions swing and flex on timescales ranging from picoseconds to milliseconds.  Even the "rigid" core of a well-folded protein is gently breathing and jiggling at all times.

This is not a bug — it is a feature.  The flexibility of proteins is central to their function.  Enzymes need to open and close around their substrates.  Signaling proteins need to switch between active and inactive shapes.  Antibodies need to be flexible enough to grip almost any antigen.

An NMR spectrometer in solution does not take a snapshot.  It **time-averages over all the conformations the protein visits during the experiment** (typically seconds to hours of data acquisition).  What you observe is a smeared, averaged signal.

The MD Trajectory module in `synth-nmr` bridges this gap.  It lets you analyze a collection of structures, called a **trajectory ensemble**, and compute the NMR observables that a real spectrometer would actually measure for a dynamic protein.

---

## What is a Molecular Dynamics Trajectory?

**Molecular Dynamics (MD) simulation** is a computational technique in which a protein is modeled as a system of atoms connected by a force field, and Newton's equations of motion are numerically integrated through time.  Each "snapshot" recorded during the simulation is called a **frame**.  Typically:

- Frames are separated by ~2 femtoseconds (2 × 10⁻¹⁵ seconds) in simulation time.
- Frames are *saved* every ~1–10 picoseconds (to keep file sizes manageable).
- A typical simulation might run for 100–1,000 nanoseconds, producing 10,000–1,000,000 frames.

The result is a **trajectory**: a movie of the protein sampling its conformational space.

You can also use any other collection of related structures as an ensemble — for example, the multiple models deposited in an NMR structure entry in the PDB (which represent the family of structures consistent with the experimental data).

---

## How NMR "sees" a moving protein

### The fast-exchange trick: proteins look like their average

Here is the key physical concept.  Consider a simple case: a single backbone atom that flips between two conformations, A and B, many thousands of times per second.

- In **conformation A**, it would resonate at frequency $f_A$.
- In **conformation B**, it would resonate at frequency $f_B$.

If the switching is **slower** than the NMR timescale (roughly slower than $f_A - f_B$), the spectrometer sees *two separate peaks* — one at $f_A$ and one at $f_B$.

If the switching is **faster**, which is the typical case for proteins in solution, the spectrometer cannot resolve the two conformations.  It sees only *one peak*, located at the **population-weighted average** of $f_A$ and $f_B$.  This is called the **fast-exchange limit**.

!!! tip "Analogy: a fast-spinning fan blade"
    When a fan spins slowly, you see individual blades.  When it spins fast enough, the blades blur into a smooth disc, and the disc appears to occupy the average position of all the blades simultaneously.  NMR in the fast-exchange limit is similar: instead of seeing each conformation individually, you see only their average.

For most protein backbone motions at physiological temperatures, exchange is fast.  This means:

| Observable | What you measure | How to compute it from a trajectory |
|---|---|---|
| Chemical shifts (δ) | The time-average of the shift | **Arithmetic mean** over all frames |
| Residual Dipolar Couplings (RDCs) | The time-average of the coupling | **Arithmetic mean** over all frames |
| NOE distances | *Not* a simple average — see below | **Sixth-power mean** over all frames |

---

### NOEs are asymmetric: close contacts dominate

The **Nuclear Overhauser Effect (NOE)** is an NMR measurement that reports on the *distance* between two hydrogen atoms.  NOE intensity is related to distance by an inverse sixth-power law (see the [NOEs page](noes.md) for the full derivation):

$$\text{NOE intensity} \propto \frac{1}{r^6}$$

Because the exponent is so large (6!), small distances contribute enormously more than large ones:
- At **2 Å** separation, the NOE contribution is $2^{-6} = 0.0156$
- At **4 Å** separation, the NOE contribution is $4^{-6} = 0.00024$  — **64× weaker**

This has a profound consequence for how you must average an NOE over a trajectory.  Suppose a proton pair spends half its time at 2 Å (close) and half at 4 Å (far away).

- **Arithmetic mean distance**: (2 + 4) / 2 = **3.0 Å**
- **Correct effective distance** (from averaging the NOE signal):

$$r_\text{eff} = \left\langle r^{-6} \right\rangle^{-1/6} = \left(\frac{2^{-6} + 4^{-6}}{2}\right)^{-1/6} = \textbf{2.27 Å}$$

The correct answer is **2.27 Å** — significantly shorter than 3.0 Å.  The transient close approach completely dominates the NOE signal.  Using a simple average would lead you to conclude the atoms are further apart than they really behave, producing incorrect structural restraints.

!!! important "Why r⁻⁶ averaging matters"
    Using arithmetic mean distances for NOEs is a well-known mistake in structural biology.  The r⁻⁶ average (also called the "sixth-power average") correctly weights the brief, close-contact conformations that contribute most to the experimental signal.  `synth-nmr` uses this correct averaging automatically in `ensemble_average_noes()`.

---

### Order parameters: measuring how much a bond wiggles

The **Lipari-Szabo order parameter S²** is one of the most informative quantities in protein dynamics.  It answers a simple question:

> *How much does this specific bond vector (like an N–H backbone bond) move around during the experiment?*

The scale runs from 0 to 1:

| S² value | What it means |
|---|---|
| **1.0** | The bond vector is completely rigid — it points in exactly the same direction in every conformation |
| **≈ 0.85** | Typical well-ordered backbone amide in a helix or β-sheet |
| **0.4–0.7** | Partially flexible — seen in surface loops, interdomain linkers |
| **≈ 0.0** | Isotropically disordered — the bond samples every direction equally (completely floppy) |

!!! tip "Analogy: a weathervane vs. a spinning top"
    A weathervane that always points north has S² ≈ 1 (rigid).  A spinning top wobbling in all directions has S² ≈ 0 (disordered).  Most backbone amide groups are like a weathervane with a small range of wobble: they will mostly point one way, with small fluctuations.

### How synth-nmr computes S² from a trajectory

For each backbone N–H bond in each frame of the trajectory, `synth-nmr` computes the **unit vector** pointing along the bond (a stick of length 1 pointing from N to H, regardless of actual bond length).

If the bond barely moves across frames, all these unit vectors point nearly the same direction.  Their **vector average** will also point strongly in that direction, giving a large magnitude.

If the bond samples all directions, the unit vectors cancel out when averaged.  Their vector average will be close to zero.

The order parameter is simply:

$$S^2 = |\langle \hat{\mu} \rangle|^2$$

the **squared magnitude of the mean unit vector**.  No model fitting required — it is a direct geometric measurement from the trajectory.

!!! note "Why not just measure the average angle?"
    Because angles do not average correctly for vectors.  If a bond points left (180°) half the time and right (0°) half the time, the average angle would be 90° — but the bond is never at 90°!  Averaging the *vectors* correctly captures the true distribution.

---

## What can you do with this?

### 1. Validate an MD force field against experimental NMR data

A common and powerful use of this module is to run an MD simulation of a protein whose NMR chemical shifts or order parameters are known from BMRB (the Biological Magnetic Resonance Bank), then compare:

```python
# Compute ensemble-averaged shifts from your MD trajectory
per_frame_shifts = [predict_chemical_shifts(f) for f in ensemble]
avg_shifts = ensemble_average_shifts(per_frame_shifts)

# Download experimental shifts from BMRB (e.g., ubiquitin entry 17769)
from synth_nmr.data_pipeline import download_bmrb_file, parse_bmrb_shifts
exp_shifts = parse_bmrb_shifts(download_bmrb_file(17769))

# Compare: compute CA shift RMSD
import numpy as np
errors = []
for res_id in avg_shifts:
    if res_id in exp_shifts and "CA" in avg_shifts[res_id] and "CA" in exp_shifts[res_id]:
        errors.append(avg_shifts[res_id]["CA"] - exp_shifts[res_id]["CA"])
rmsd = np.sqrt(np.mean(np.array(errors)**2))
print(f"CA shift RMSD vs. experiment: {rmsd:.2f} ppm")
```

A lower RMSD means the MD simulation samples conformations closer to what the protein does in the real experiment.  This is one of the most direct ways to benchmark a force field.

### 2. Identify flexible regions of a protein

S² values below ~0.7 indicate regions of significant flexibility.  These regions are often biologically important: they may be binding interfaces that become structured only upon ligand binding, or linkers that allow domain re-orientation.

```python
s2_map = compute_s2_from_trajectory(ensemble)

print("Flexible residues (S² < 0.7):")
for res_id, s2_val in sorted(s2_map.items()):
    if s2_val < 0.7:
        print(f"  Residue {res_id}: S² = {s2_val:.3f}")
```

### 3. Compute back-calculated NOE restraints for structure refinement

If you have an MD ensemble and want to generate a set of NOE distance restraints compatible with that ensemble (for use in further structure refinement), the r⁻⁶-averaged distances are the physically correct restraints to use:

```python
from synth_nmr import calculate_synthetic_noes
from synth_nmr.trajectory import ensemble_average_noes

per_frame_noes = []
for frame in ensemble:
    noe_raw = calculate_synthetic_noes(frame, cutoff=5.0)
    # Flatten to {(res_i, res_j): distance} format
    flat = {(ri, rj): d for ri, peers in noe_raw.items() for rj, d in peers.items()}
    per_frame_noes.append(flat)

effective_noes = ensemble_average_noes(per_frame_noes)
# effective_noes is now a dict of (res_i, res_j) → r_eff in Å
```

---

## Full Practical Guide

### Step 1: Assemble your frames

**Option A — From PDB files (no extra software needed):**
```python
import biotite.structure.io as strucio
from synth_nmr.trajectory import load_trajectory

frames = [strucio.load_structure(f"frame_{i:04d}.pdb") for i in range(100)]
ensemble = load_trajectory(frames)
print(f"Loaded {len(ensemble)} frames")
```

**Option B — From an GROMACS or AMBER trajectory (requires MDTraj):**
```bash
pip install synth-nmr[trajectory]   # install MDTraj support
```
```python
# Works with .xtc, .trr, .nc, .dcd, and many other formats
ensemble = load_trajectory("md_production.xtc", topology="protein.pdb", stride=10)
```

> **stride=10** means "use every 10th frame" — sufficient for NMR observables and much faster.

### Step 2: Choose your observable

```python
from synth_nmr import predict_chemical_shifts, calculate_rdcs
from synth_nmr.trajectory import (
    ensemble_average_shifts,
    ensemble_average_noes,
    ensemble_average_rdcs,
    compute_s2_from_trajectory,
)

# --- Chemical shifts (arithmetic mean) ---
per_frame_shifts = [predict_chemical_shifts(f) for f in ensemble]
avg_shifts = ensemble_average_shifts(per_frame_shifts)
# avg_shifts[1]["CA"]  →  CA chemical shift of residue 1 in ppm

# --- RDCs (arithmetic mean) ---
per_frame_rdcs = [calculate_rdcs(f, Da=10.0, R=0.5) for f in ensemble]
avg_rdcs = ensemble_average_rdcs(per_frame_rdcs)
# avg_rdcs[1]  →  N-H RDC for residue 1 in Hz

# --- Order parameters (no per-frame list needed — works directly on ensemble) ---
s2_map = compute_s2_from_trajectory(ensemble)
# s2_map[1]  →  S² for residue 1  (0 = flexible, 1 = rigid)
```

### Step 3 (optional): Use the CLI

If you prefer not to write Python code, the command-line interface handles the full workflow:

```bash
# Load three PDB frames and compute order parameters
python -m synth_nmr.synth_nmr_cli \
    load trajectory frame1.pdb frame2.pdb frame3.pdb \
    ensemble s2

# Ensemble-averaged chemical shifts
python -m synth_nmr.synth_nmr_cli \
    load trajectory frame*.pdb \
    ensemble shifts

# NOE effective distances (r⁻⁶ averaged), cutoff 5.0 Å
python -m synth_nmr.synth_nmr_cli \
    load trajectory frame*.pdb \
    ensemble noes 5.0
```

---

## Summary of Averaging Rules

| Observable | Physical reason | Averaging method | synth-nmr function |
|---|---|---|---|
| Chemical shifts | Fast exchange: peak at time-average shift | Arithmetic mean | `ensemble_average_shifts()` |
| RDCs | Fast exchange: coupling at time-average orientation | Arithmetic mean | `ensemble_average_rdcs()` |
| NOE distances | NOE ∝ r⁻⁶: short contacts dominate | r⁻⁶ mean: `⟨r⁻⁶⟩^(−1/6)` | `ensemble_average_noes()` |
| S² order parameter | Bond vector autocorrelation plateau | `\|⟨μ̂⟩\|²` | `compute_s2_from_trajectory()` |

---

## Key References

| Topic | Reference |
|---|---|
| NOE cross-relaxation and r⁻⁶ dependence | Solomon, I. (1955) *Phys. Rev.* 99, 559 |
| Lipari-Szabo model-free formalism and S² | Lipari, G. & Szabo, A. (1982) *J. Am. Chem. Soc.* 104, 4546 |
| S² from trajectory: C(∞) = \|⟨μ⟩\|² | Clore, G.M. et al. (1990) *J. Am. Chem. Soc.* 112, 4989 |
| NMR as a validator of MD force fields | Bax, A. (2003) *Protein Sci.* 12, 1 |
| MDTraj Python library | Eastman et al. (2017) *J. Chem. Theory Comput.* 13, 461 |
