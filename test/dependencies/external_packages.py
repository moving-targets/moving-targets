import os
from typing import Dict, Callable

from moving_targets.callbacks import WandBLogger
from moving_targets.learners import MultiLayerPerceptron
from moving_targets.masters.backends import CplexBackend, CvxpyBackend, GurobiBackend
from moving_targets.util.errors import MissingDependencyError
from test.abstract import AbstractTest


def _dependency_path(dependency: str) -> str:
    """This serves debugging purposes only and it is used to retrieve the dependency filepath if unexpectedly loaded."""
    if dependency == 'docplex':
        import cvxpy
        return cvxpy.__file__
    elif dependency == 'docplex':
        import docplex
        return docplex.__file__
    elif dependency == 'gurobipy':
        import gurobipy
        return gurobipy.__file__
    elif dependency == 'tensorflow':
        import tensorflow
        return tensorflow.__file__
    elif dependency == 'wandb':
        import wandb
        return wandb.__file__
    else:
        raise AssertionError(f"Unexpected dependency '{dependency}'")


def _test_dependency(dependency: str, constructor: Callable) -> bool:
    """This is the actual core test, which returns a boolean value meaning passed/not passed."""
    # creates the temporary folder
    temporary_folder = AbstractTest.get_relative_path('site-packages')
    os.makedirs(temporary_folder, exist_ok=True)
    package_path = AbstractTest.get_relative_path('venv', 'Lib', 'site-packages', dependency)
    temporary_path = AbstractTest.get_relative_path('site-packages', dependency)
    os.rename(package_path, temporary_path)
    try:
        constructor()
        raise AssertionError(f"Found unexpected dependency at '{_dependency_path(dependency)}'")
    except MissingDependencyError as exception:
        status = str(exception).startswith('This class requires')
    finally:
        # once the tests are done, source folder back to its path and the temporary folder is removed
        os.rename(temporary_path, package_path)
        os.rmdir(temporary_folder)
    return status


class TestExternalPackages(AbstractTest):
    # Since Moving Targets uses different backends both for the learners and for the masters, when delivering its code
    # as a python library these packages are not included in the requirements. Therefore, it may happen that a user
    # calls a method requiring a particular package and the expected behaviour should be to raise a custom exception
    # saying that which package (and version) should be installed.
    #
    # This test is used to check that this behaviour is correct. Indeed, it instantiates some classes requiring the
    # external packages and checks whether or not the correct exceptions are thrown.
    #
    # Since, as far as I understood by looking up on the web, there is no way to change virtual environment a part from
    # manually editing the run configurations (something that cannot be exploited since the tests are run automatically
    # using the setup.py script), the only solution I found was to call each test method with the respective python
    # package name, then moving the source folders into a temporary folder then switch them back at the end of the
    # test. Far from being elegant, this seems to work, but it may be worth a little more investigation.
    #
    # Also, the test must be run during class loading and, eventually, the results may be checked. This is due to the
    # fact that, if another test in the suite uses one of the dependencies, then python will be able to retrieve it as
    # well even if the source folder of the dependency is removed, leading to test failure.

    _results: Dict[str, bool] = {d: _test_dependency(dependency=d, constructor=c) for d, c in [
        ('cvxpy', CvxpyBackend),
        ('docplex', CplexBackend),
        ('gurobipy', GurobiBackend),
        ('wandb', lambda: WandBLogger(project='', entity='', run_name='')),
        ('tensorflow', lambda: MultiLayerPerceptron(loss='', output_units=1, output_activation=None))
    ]}

    def test(self):
        for k, v in self._results.items():
            self.assertTrue(v, msg=f"Error in testing lazy dependency '{k}'")
