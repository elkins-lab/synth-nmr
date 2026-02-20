# Scalar Couplings (J-Couplings)

In NMR spectroscopy, magnetization can be transferred not just through space (NOE) but also *through chemical bonds*. This interaction is known as **Scalar Coupling** or **J-Coupling**.

While NOEs give us distance restraints between protons, J-Couplings are sensitive to the **torsion angles** (dihedrals) of the chemical bonds separating the coupled nuclei.

The most critical scalar coupling for protein backbone structure determination is the three-bond coupling between the Amide proton (HN) and the Alpha proton (H$\alpha$): the $^{3}J_{\text{HN-H}\alpha}$ coupling. Because this coupling traverses the N-C$\alpha$ bond, its magnitude is directly related to the protein backbone dihedral angle **$\phi$** (phi).

## The Karplus Equation

In 1959, Martin Karplus derived an empirical relationship relating the magnitude of the three-bond J-coupling to the intervening dihedral angle:

$$
^3J(\theta) = A \cos^2(\theta) + B \cos(\theta) + C
$$

Where:
- $\theta$ is the relevant dihedral angle (for the peptide backbone, $\theta = \phi - 60^\circ$).
- $A, B$, and $C$ are empirical parameters that depend on the specific nuclei and their electronegative substituents.

### Parameterization by Vuister and Bax

The accuracy of the Karplus equation in predicting secondary structure depends entirely on having properly calibrated $A, B$, and $C$ parameters.

In 1993, **Geerten Vuister** and **Ad Bax** published a seminal parameterization of the Karplus curve specifically for the $^{3}J_{\text{HN-H}\alpha}$ coupling in proteins. By analyzing a high-resolution database of proteins with known crystal structures and meticulously measured NMR scalar couplings, Vuister and Bax derived the standard parameters that remain the bedrock of the field today:

$$
^3J_{\text{HN-H}\alpha}(\phi) = 6.51 \cos^2(\phi - 60^\circ) - 1.76 \cos(\phi - 60^\circ) + 1.60
$$

These parameters reveal clear biophysical signatures for secondary structure:
- **$\alpha$-helices** (where $\phi \approx -60^\circ$): Predict a small J-coupling $\sim 4 - 5 \text{ Hz}$.
- **$\beta$-sheets** (where $\phi \approx -120^\circ$ to $-140^\circ$): Predict a large J-coupling $\sim 8 - 10 \text{ Hz}$.

By measuring the J-coupling, spectroscopists can firmly establish the local backbone geometry of the protein chain.

---

## `synth-nmr` Implementation

`synth-nmr` mathematically applies the Vuister-Bax parameterized Karplus equation to any input 3D coordinates.

### Function: `calculate_hn_ha_coupling`

```python
from synth_nmr import calculate_hn_ha_coupling

# Calculate the backbone phi angles from coordinates and apply Karplus
j_couplings = calculate_hn_ha_coupling(structure)

print(j_couplings[12]) # Output: 4.8 Hz (Likely an alpha-helix!)
```

**Key Steps in the Algorithm**:
1.  **Iterative Dihedral Extraction**: The algorithm walks down the peptide backbone. For each residue, it isolates the `C'` (previous carbonyl), `N`, `CA`, and `C` coordinates.
2.  **Angle Calculation**: It calculates the precise atomic torsion angle, $\phi$.
3.  **Karplus Application**: The angle is shifted by $-60^\circ$ to adjust for the specific geometry of the H-N-C$\alpha$-H$\alpha$ system, and evaluated against the Vuister-Bax $A, B, C$ coefficients.
