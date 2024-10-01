# How to Use this Repository
This repository contains a selection of submodels of current distribution with a focus on the negative electrode, separator and positive electrode of solid-state batteries.
The Python program files are auxilliary, FEniCSx models or geometry preparation scripts. Some of the scripts are tied to specific published works and can be used to reproduce
the relevant aspects of the published works. In any case, the usage of scripts can be found with `python3 script-name.py -h`.

You will need to have Python 3 and FEniCSx installed to be able to run these scripts. Installation instructions can be found at https://github.com/FEniCS/dolfinx. Additional packages required can be found at [requirements.txt](requirements.text). If you notice any errors and omissions, feel free to create a pull request and address the issue. No promise is made that the codes will be kept up to date across changing versions of the input software packages.

### Publications and Scripts
#### E. L. Molel and T. F. Fuller, Application of Open-Source Python-Based Tools for the Simulation of Electrochemical Systems. Journal of the Electrochemical Society
##### Hull Cell Demo
  - hull-cell.py
##### Relative Feature Size
  - study_3.py 
  - study_3_geo.py
##### Contact Loss Distribution
  - study_2.py
  - study_2_geo.py
  - study_4.py
  - study_4_geo.py
  - study_5.py
  - study_5_geo.py
#### E. L. Molel and T. F. Fuller, Effects of Contact Loss at Electrolyte/Negative Electrode Interface on Current Density Distribution in Solid-State Batteries. Journal of the Electrochemical Society
##### Contact loss model script
  - contact_loss_separator.py
##### Geometry preparation
  - contact_loss_separator_lma_geo.py
  - contact_loss_separator_ref_geo.py
