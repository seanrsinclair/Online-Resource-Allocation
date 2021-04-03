# Sequential Fair Allocation of Limited Resources
This repository contains a reference implementation of the heuristic algorithms appearing in the paper [1].  We also include implementation for the sequential resource allocation problem in [2].

### Dependencies
The code has been tested in `Python 3` and depends on a number of Python
packages.

* numpy 1.19.1
* matplotlib 3.3.0
* plotly 4.9.0
* pandas 1.1.0
* seaborn 0.10.1
* scipy 1.5.2
* jupyter 1.0.0

### Quick Tour

We offer implementations for online resource allocation algorithms that aim to maximize fairness with a limited budget of resources.

The following files in `functions/` implement the different algorithms for a single resource:
* `food_bank_functions.py`: implements all the different online heuristic algorithms as well as the offline solution.  Also provides implementation calculating the various fairness metrics.
* `food_bank_bayesian.py`: implements a dynamic programming version of the Nash Social Welfare by maximizing the expectation of the Eisenberg-Gale program.  This isn't directly used in the paper, but was considered during the preliminary phases of the project.

The following files in `functions_multi_resource/` implement the different algorithms for a multiple resources:
* `food_bank_functions.py`: implements the different heuristic algorithms as well as the offline optimal solution.  Also provides implementation calculating the various fairness metrics.

The files found in `simulations/` run experiments with different type distributions. To run the experiments used in the paper, you should run `waterfilling_levels.ipynb`, where all information is subsequently printed throughout the jupyter notebook.  Each file has parameters at the top which can be changed in order to replicate the parameters considered for each experiment in the paper.

### Bibliography
[1]: Sean R. Sinclair, Gauri Jain, Siddhartha Banerjee, Christina Yu.  *Sequential Fair Allocation of Limited Resources.*

[2]. Robert Lien, Seyed Iravani, Karen Smilowitz. [https://pubsonline.informs.org/doi/abs/10.1287/opre.2013.1244](*Sequential Resource Allocation for Nonprofit Operations.*)
