# Laminate Cell Model
In this work, we conceive a laminate cell where the active material and the solid electrolyte are connected along the transport axis and thus are not tortuous. At this, stage, we are exploring whether the laminate model might be less computationally intensive and thus is tractable with direct numerical simulations. A less computationally-intensive direct numerical simulation might paint a clearer picture than pseudo-two-dimensional models paired with effective properties.

## Background and Motivation
There are various fabrication routes for experimental solid-state batteries including dry-pressing and slurry coating. Let us highlight dry-pressing. Active material particles are milled and mixed with milled solid electrolyte. Processing binder and conductive additives are then added to improve electrical and mechanical properties. Ideally, we hope that the active material particles get connected along the transport axis from the current collector to the electrode/separator interface.

This provides some motivation to consider porous electrodes that do not assume spherical active material particles. There is nothing special about spherical particles considering that processing routes merge the particles in complex ways. The laminate cell can be set such that we really only need to solve the equations in 2D. This can happen if you assume the simple laminate with layers of active material and solid electrolyte in contact with one another.
## Model Development
In developing the laminate cell model, we borrow the set up already used in the Doyle-Fuller-Newman model. We simplify the DFN model based on the material properties we are working with. We will attempt to use information learnt in phase-field model to further take advantage of variational calculus in the FEA solution to the laminate cell model.
### Model Assumptions
### Model Validation
## Conclusions and Recommendations