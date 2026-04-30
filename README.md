# pdProj: Projected-Search Interior-Point Method

This repository contains MATLAB code for the projected-search interior-point method described in

> Philip E. Gill and Minxin Zhang, **A projected-search interior-point method for nonlinearly constrained optimization**, *Computational Optimization and Applications*, 88(1), 37--70, 2024.

The main solver is `pdProj.m`. It solves nonlinear optimization problems of the form

```text
minimize        f(x)
subject to      bl <= x    <= bu,
                cl <= c(x) <= cu,
```

where `f` is a twice differentiable objective, `c` is a vector of twice differentiable nonlinear constraints, and the variable and constraint bounds may include infinite values. Equality constraints are represented by equal lower and upper bounds.

## Repository contents

```text
pdProj.m             Main projected-search primal-dual all-shifted penalty-barrier solver
pdbTest.m            Driver that calls pdProj and reports run statistics
runProb.m            Convenience wrapper for running a CUTEst problem or the custom template
my_pdproj_parms.m    Default solver parameters
getCUTEstProblem.m   CUTEst-to-MATLAB problem interface
getCUSTomProblem.m   Template for defining a small user-specified problem
ktSolverSparse.m     Sparse KKT solver using LDL factorization and inertia correction
scaleSymSparse.m     Symmetric sparse matrix scaling utility
```

A run writes a text log named `<problem-name>.out` in the working directory.

## Requirements

- **MATLAB:** The code is written in standard MATLAB.
- **CUTEst:** Testing against standard nonlinear optimization problems requires the CUTEst MATLAB interface to be installed and configured.

A custom problem template is included. Depending on the current driver setup, the custom example may still call CUTEst-related routines, so please check `runProb.m` and `getCUSTomProblem.m` before running the code without CUTEst.

## Usage

To execute the solver, run `runProb.m` from the MATLAB command window.

**Run the default custom problem:**

```matlab
>> runProb()
```

This calls the default problem specified in `runProb.m`, runs `pdProj.m`, and writes a log file named `<problem-name>.out`.

## Citation

If you use this code in your research, please cite the published paper:

```bibtex
@article{gill2024projected,
  title     = {A projected-search interior-point method for nonlinearly constrained optimization},
  author    = {Gill, Philip E. and Zhang, Minxin},
  journal   = {Computational Optimization and Applications},
  volume    = {88},
  number    = {1},
  pages     = {37--70},
  year      = {2024},
  publisher = {Springer}
}
```

## License

The mathematical formulation, text, and figures of the associated paper, *A projected-search interior-point method for nonlinearly constrained optimization*, are licensed under a Creative Commons Attribution 4.0 International License.

The MATLAB source code provided in this repository is licensed separately under the PolyForm Noncommercial License 1.0.0. This code may be used, modified, and distributed for noncommercial purposes, subject to the terms of the license.

Commercial use, including integration into commercial solvers or proprietary software pipelines, is not permitted under this license without separate permission from the copyright holders. For commercial licensing inquiries, please contact the authors directly.

A copy of the PolyForm Noncommercial License 1.0.0 is provided in the `LICENSE` file.