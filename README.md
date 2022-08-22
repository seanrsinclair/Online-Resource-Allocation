# Sequential Fair Allocation of Limited Resources
This repository contains a reference implementation of the heuristic algorithms appearing in our paper.

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

The files found in `simulations/` run experiments with different type distributions. To run the experiments used in the paper, you should run `adaptive_threshold.ipynb`, where all information is subsequently printed throughout the jupyter notebook.  Each file has parameters at the top which can be changed in order to replicate the parameters considered for each experiment in the paper.


