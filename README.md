# Frank-Wolfe Optimization Algorithms for Movie Recommendations

This work provides an implementation of the classical Frank-Wolfe algorithm and its variant, i.e., the Away-Steps Frank-Wolfe algorithm, with minimization of the Root Mean Squared Error (RMSE) under the nuclear norm-ball constraints for amovie recommendation task.

## Files

### `FW_recc.py`

This Python script contains all the core functions and the two Frank-Wolfe algorithms functions for the optimization task, that are:

### `FW_comp.ipynb`

This Jupyter notebook performs the actual comparison between the Frank-Wolfe and Away-Steps Frank-Wolfe algorithms. This notebook tests the algorithms on the movie ratings dataset and plots the gaps.

### `work_report.pdf`

A comprehensive report describing the work, methodologies, and discussions on the results obtained from the algorithms.

### `ratings.csv`

Dataset file (from at https://grouplens.org/datasets/movielens/) used for the project.