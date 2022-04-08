import importlib.resources
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import Metric
from moving_targets.util.errors import not_implemented_message
from test.abstract import AbstractTest


class TestExamples(AbstractTest):
    _PRINT_GROUND_TRUTHS: bool = False
    """Turn this flag on whenever some requirement or algorithm change is expected to modify the examples results.
    When this flag is on, run the tests in order to print the expected results on the console and then copy them.
    Be careful to change the least amount of requirements or lines of code as possible since an error in this step
    will eventually lead to wrong ground truths and, subsequently, wrong regression tests."""

    RESULTS: Dict[str, Dict[str, float]]
    """Contains the expected results indexed via '<dataset>-<init_step>-<loss>' in the form of a dictionary where each
    key is a string '<split>-<metric>' and the value is the metric result."""

    def _stratify(self) -> bool:
        """Whether to stratify or not the data when splitting between train and validation."""
        raise NotImplementedError(not_implemented_message('_stratify'))

    def _learner(self) -> Learner:
        """The `Learner` instance."""
        raise NotImplementedError(not_implemented_message('_learner'))

    def _master(self, loss: str) -> Master:
        """The `Master` instance."""
        raise NotImplementedError(not_implemented_message('_master'))

    def _metrics(self) -> List[Metric]:
        """The list of `Metric` instances on which to evaluate the model."""
        raise NotImplementedError(not_implemented_message('_metrics'))

    def _test(self, dataset: str, class_column: str, init_step: str, loss: str):
        """Implements the testing strategy for a given dataset using a certain loss, a certain initial step, and
        having the given target label."""
        np.random.seed(self.SEED)
        # load data
        with importlib.resources.path('res', f'{dataset}.csv') as filepath:
            df = pd.read_csv(filepath)
            x, y = df.drop(class_column, axis=1), df[class_column].astype('category').cat.codes.values
            xtr, xts, ytr, yts = train_test_split(x, y, stratify=y if self._stratify() else None, shuffle=True)
        # define model pieces (and assert backend is gurobi)
        master = self._master(loss=loss)
        self.assertIsInstance(master.backend, GurobiBackend, msg='Gurobi backend required for tests')
        model = MACS(init_step=init_step, learner=self._learner(), master=master, metrics=self._metrics())
        model.fit(xtr, ytr, iterations=3, verbose=False)
        # test results
        exp_res = self.RESULTS[f'{dataset}-{init_step}-{loss}']
        act_res = dict(train=model.evaluate(xtr, ytr), test=model.evaluate(xts, yts))
        if self._PRINT_GROUND_TRUTHS:
            print(f"'{dataset}-{init_step}-{loss}': " + '{')
            for split, act in act_res.items():
                for metric, val in act.items():
                    print(f"    '{split}_{metric}': {val},")
            print('},')
        else:
            for split, act in act_res.items():
                for metric, val in act.items():
                    msg = f'{metric} does not match in {split}'
                    self.assertAlmostEqual(exp_res[f'{split}_{metric}'], val, places=self.PLACES, msg=msg)
