# Moving Targets

Moving Targets is novel hybrid technique which can be used to inject external constraints and domain knowledge into any
machine learning model. The algorithm is based on an iterative schemes that alternates between two steps: a learner
step, which is in charge of training the machine learning model, and a master step, which refines the model predictions
by projecting them into the feasible space, so that the model can be learnt again on these adjusted targets. Since the
model is general-purpose, it can address any type of learner and constraints, given that these constraints lead to a
solvable constrained optimization problem (see the original [paper](https://arxiv.org/abs/2002.10766) for details).

## How To Use

You can install the Moving Targets' library via ```pip``` by pasting this command in your terminal:

```
pip install moving-targets
```

The library contains various classes and utility functions, but the core of the algorithm can be found in the ```MACS```
class, which needs both a ```Learner``` and a ```Master``` instance in order to be instantiated — additional parameters
such as the initial step or a list of ```Metric``` objects may be optionally passed as well. Once the ```MACS``` object
has been built, it can be used as a classical machine learning model along with its methods ```fit()```,
```predict()```, and ```evaluate()```.

A predefined set of learners, metrics, and callbacks can be found in the respective modules. Instead, as regards the
master, you will need to extend one of the template ones in order to formulate your constrained optimization problem,
even though the master formulation is facilitated by a set of predefined loss functions and backend objects that wrap
some of the most famous constraint optimization solvers — e.g., Cplex, Gurobi, CvxPy, etc.

More specific tutorials to learn how to use Moving Targets can be found in the ```examples``` module.

## How To Cite

If you use Moving Targets in your works, please cite the following article:

```
@article{Detassis2020,
  title={Teaching the Old Dog New Tricks: Supervised Learning with Constraints},
  author={Detassis, Fabrizio and Lombardi, Michele and Milano, Michela},
  journal={arXiv preprint arXiv:2002.10766},
  year={2020}
}
```

## Contacts

* Maintainer: [luca.giuliani13@unibo.it](mailto:luca.giuliani13@unibo.it)
