# Determining heterogeneous mechanical properties of biological tissues via PINNs

The data and code for the paper [W. Wu, M. Daneker, K. T. Turner, M. A. Jolley, & L. Lu. Identifying heterogeneous micromechanical properties of biological
tissues via physics-informed neural networks. *Small Methods*, 9 (1), 2400620, 2025](https://doi.org/10.1002/smtd.202400620).

## Data
All data are in the folder [data](data). The first word in the file name indicates the example name, and the last word before ".npy" indicates the constitutive model name. For example, "GRF_equi_disp0.4_neo.npy" contains data for the Gaussian random field example generated using the Neo-Hookean material model. 

## Code

All code are in the folder [src](src). The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde) v1.10.2. 

- [Neo-Hookean material model](src/NeoHookean_elasticity_map.py)
- [Mooney Rivlin material model](src/MooneyRivlin_elasticity_map.py)
- [Gent material model](src/Gent_elasticity_map.py)

To run the code, specify name of reference data file and network architecture type (defaults shown):
```bash
python NeoHookean_elasticity_map.py --data ../data/GRF_equi_disp0.4_neo.npy --network 2B
```
## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{wu2025identifying,
  author  = {Wu, Wensi and Daneker, Mitchell and Turner, Kevin T. and Jolley, Matthew A. and Lu, Lu},
  title   = {Identifying heterogeneous micromechanical properties of biological tissues via physics-informed neural networks}, 
  journal = {Small Methods},
  volume  = {9},
  number  = {1},
  pages   = {2400620},
  year    = {2025},
  doi     = {https://doi.org/10.1002/smtd.202400620}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
