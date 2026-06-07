# Chemical Shifts

The **Chemical Shift** ($\delta$) is the defining parameter of NMR spectroscopy. The exact resonance frequency (Larmor frequency) of a nucleus is exquisitely sensitive to its local electronic environment.

Because electrons are charged particles in motion, they generate tiny induced magnetic fields that oppose the giant external spectrometer field ($B_0$). We say the nucleus is **shielded** by its electron cloud.

$$
B_{\text{effective}} = B_0 \cdot (1 - \sigma)
$$

Where $\sigma$ is the chemical shielding tensor. Changes to the backbone conformation, hydrogen bonding, or nearby aromatic rings will change the electron distribution, thereby changing $\sigma$, and resulting in a "shift" in the measurable frequency.

## Secondary Structure Dependence

The most profound application of chemical shifts in proteins is the determination of secondary structure. Backbone atoms ($^{13}\text{C}\alpha$, $^{13}\text{C}\beta$, $^{13}\text{C}'$, $^{15}\text{N}$, $^1\text{H}\alpha$) show massive, consistent deviations from their random coil values depending on the local $\phi / \psi$ dihedral angles.

The **Chemical Shift Index** (CSI) formalized this:
-   **$\alpha$-helices**: $^{13}\text{C}\alpha$ heavily shifts downfield (positive $\Delta\delta$), while $^{13}\text{C}\beta$ shifts upfield (negative $\Delta\delta$).
-   **$\beta$-sheets**: The exact opposite.$^{13}\text{C}\alpha$ shifts upfield, $^{13}\text{C}\beta$ shifts downfield.

By simply analyzing the assigned backbone chemical shifts without any structures or distance restraints, a researcher can perfectly map the secondary structural elements of the protein.

## The Physics of Ring Currents

A dramatic source of chemical shifting arises from **Aromatic Ring Currents**.

Residues like Phenylalanine, Tyrosine, and Tryptophan possess delocalized $\pi$-electron clouds above and below the plane of their aromatic rings. When placed in the $B_0$ spectrometer field, these $\pi$-electrons freely circulate in a loop, acting like a tiny electromagnet.

This ring current generates a powerful, highly localized magnetic field. If a proton from another residue is spatially folded such that it sits *directly above* the face of a Tryptophan ring, it will experience a massive opposing field from the ring current. It will be strongly **shielded**, and its chemical shift will move drastically upfield (even into negative ppm territories!).

Because this effect is entirely dependent on 3D spatial geometry (distance and angle from the ring plane), predicting ring current shifts is a powerful way to validate tertiary folds.

---

## `synth-nmr` Implementation

`synth-nmr` utilizes empirical algorithms based on the **SPARTA+** methodology to predict chemical shifts directly from 3D coordinates.

### Function: `predict_chemical_shifts`

```python
from synth_nmr import predict_chemical_shifts

# Predict H, N, CA, CB, and C' shifts for every residue
shifts = predict_chemical_shifts(structure)

print(shifts[10]['CA']) # Output: 56.4 (ppm)
```

**Key Steps in the Algorithm**:
1.  **Backbone Geometry**: The algorithm identifies primary secondary structures using Ramachandran $\phi / \psi$ binning and applies base conformation shifts.
2.  **Ring Current Corrections**: The algorithm identifies all aromatic rings (PHE, TYR, TRP, HIS) and establishes their geometric normal vectors. It then calculates the Haigh-Mallion spatial integral to apply shielding/deshielding corrections to all neighboring atoms based on their exact Cartesian distance and elevation angle from the ring center.

### SHIFTX2: The Primary Predictor

By default, `predict_chemical_shifts()` first attempts to use **SHIFTX2**, a
state-of-the-art hybrid machine-learning predictor (Han et al. 2011) that achieves
roughly half the error of SPARTA+ fragment-mining methods (~0.44 ppm Cα RMSD vs.
~0.9–1.1 ppm).  If SHIFTX2 is not installed or encounters an error, the function
automatically falls back to the SPARTA+ empirical model described above.

See the **[SHIFTX2 Integration](shiftx2.md)** page for:

- Why SHIFTX2 is significantly more accurate
- Step-by-step installation instructions
- How the automatic detection and fallback works
- Direct API usage (`ShiftX2Predictor` class)
- A table of all automated test cases
