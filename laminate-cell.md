# Laminate Cell Model
In this work, we conceive a laminate cell where the active material (AM) and the solid electrolyte (SE) are connected along the transport axis and thus are not tortuous. At this, stage, we are exploring whether the laminate model might be less computationally intensive and is thus tractable with direct numerical simulations. A less computationally-intensive direct numerical simulation might paint a clearer picture than pseudo-two-dimensional models paired with effective properties.

## Background and Motivation
There are various fabrication routes for experimental solid-state batteries including dry-pressing and slurry coating. Let us highlight dry-pressing. Active material particles are milled and mixed with milled solid electrolyte. Processing binder and conductive additives are then added to improve electrical and mechanical properties. Ideally, we hope that the active material particles get connected along the transport axis from the current collector to the electrode/separator interface.

This provides some motivation to consider porous electrodes that do not assume spherical active material particles. There is nothing special about spherical particles considering that processing routes merge the particles in complex ways. The laminate cell can be set such that we really only need to solve the equations in 2D. This can happen if you assume the simple laminate with layers of active material and solid electrolyte in contact with one another.
## Model Development
In developing the laminate cell model, we borrow the set up already used in the Doyle-Fuller-Newman model. We simplify the DFN model based on the material properties we are working with. We will attempt to use information learnt in phase-field model to further take advantage of variational calculus in the FEA solution to the laminate cell model.
### Model Assumptions
- For kinetics, we use the Butler-Volmer equation: $$i = i_0 \left[ e^{\frac{\alpha_aF\eta_s}{RT}} - e^{\frac{-\alpha_cF\eta_s}{RT}}\right]$$ where $$\eta_s = \phi_1 - \phi_2$$ is the surface overpotential, and $$i_0$$ is the exchange current density.
- Local ionic current density: $$i_1 = -\kappa_0 \nabla \phi_1$$
- Local electronic current density: $$i_2 = -\sigma_0 \nabla \phi_2$$
- At the AM/SE interface, we have the pore-wall flux across the interface, $$j_n$$
- Pore-wall flux: $$j_n = -D_{eff}\nabla c_2$$ for concentration of lithium $c_2$ in the AM phase. $$aj_n$$ is the volumetric charge generation
- Insertion process at the cathode proceeds as:
  $$\chemfig{Li^+  - \Theta_1} + \Theta_2 + e^- \rightleftharpoons \chemfig{Li - \Theta_2} + \Theta_1$$
- Since there is no penetration of materials into the current collectors, then the fluxes are set to zero there.
- Current in the two phases is conserved, thus: $$\nabla \cdot (i_1 + i_2) = 0$$ and thus $i = i_1 + i_2$
- $$i = i_1 + i_2 = \frac{i_0}{A}\left[e^{\frac{\alpha_a F (\phi_1 - \phi_2)}{RT}} - e^{-\frac{\alpha_c F (\phi_1 - \phi_2)}{RT}}\right]$$
  $$\implies$$ $$- \kappa_0 \nabla \phi_1 - \sigma_0 \nabla \phi_2 = \frac{i_0}{A}\left[e^{\frac{\alpha_a F (\phi_1 - \phi_2)}{RT}} - e^{-\frac{\alpha_c F (\phi_1 - \phi_2)}{RT}}\right]$$
### Model Validation
## Conclusions and Recommendations