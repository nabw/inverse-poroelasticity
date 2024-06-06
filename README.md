# Fully nonlinear inverse poroelasticity: Stress-free configurationrecovery

The code in this repo was the one we used to develop this work: https://doi.org/10.1016/j.cma.2024.116960. In particular:

- The script 'pm-one-phase-2d.py' runs the 2d inverse and forward problems in the square. 
- The script 'pm-one-phase-3d.py' runs the 3d inverse and forward problems in the slab. 
- The script 'lv-one-phase.py' runs the 3d inverse and forward problems in the idealized left ventricle. 

All script take as input the parameters FORMULATION and ACCELERATION:

    python pm-one-phase-2d.py FORM ACC

where:
 - FORM == 0: Primal formulation
 - FORM == 1: Mixed-p formulation
 - FORM == 2: Mixed-u formulation

and ACC denotes the acceleration parameter. Use 0 to skip the acceleration step. Within the script you will find other relevant parameters such as the tolerances and FEM degrees, among others. 

All paraview files and exported into an 'output' folder, created on the fly during the simulations.

\* The fiber generation script is non-operative until the relevant publication gets reviewed (half a year waiting as of sending it, after an 8 month long desk-rejection).
