> Under Construction

# Physics-informed neural networks for determining heterogeneous mechanical properties of biological tissues

The data and code for the paper [W. Wu, M. Daneker, K. T. Turner, M. A. Jolley, & L. Lu. Identifying heterogeneous micromechanical properties of biological
tissues via physics-informed neural networks. *arXiv preprint arXiv:2402.10741*, 2024](https://arxiv.org/abs/2402.10741).

## Code

All data and code are in the folder [src](src). The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde) v1.10.2. 

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
@article{wu2024heterogeneousmaterial,
  title={Identifying heterogeneous micromechanical properties of biological tissues via physics-informed neural networks}, 
  author={Wensi Wu and Mitchell Daneker and Kevin T. Turner and Matthew A. Jolley and Lu Lu},
  journal = {Small Methods},
  volume  = {},
  pages   = {2400620},
  year    = {2004},
  doi     = {https://doi.org/10.1002/smtd.202400620}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
