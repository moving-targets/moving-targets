import os
from typing import Union, Optional

import numpy as np
import pytest

from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import ClassificationMaster, RegressionMaster
from moving_targets.masters.backends import GurobiBackend
from moving_targets.util import probabilities
from test.test_abstract import TestAbstract


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='No solver in Github Actions')
class TestMasters(TestAbstract):
    class DummyLearner(Learner):
        def __init__(self, regression: bool):
            super(TestMasters.DummyLearner, self).__init__(mask=None, x_scaler=None, y_scaler=None, stats=False)
            self.regression: bool = regression
            self.p: Optional[np.ndarray] = None

        def _fit(self, x, y, sample_weight=None):
            self.p = y

        def _predict(self, x):
            return self.p if self.regression or self.p.ndim == 2 else probabilities.get_onehot(self.p.astype(int))

    def _test(self, task: Union[int, str], loss: str, types: Optional[str] = None):
        """Checks that the masters implementations behave correctly with respect to the given parameters."""
        rng = np.random.default_rng(self.SEED)
        x = rng.random((self.NUM_SAMPLES, self.NUM_FEATURES))
        # handle types
        if types == 'discrete':
            binary, classes = True, True
        elif types == 'discretized':
            binary, classes = False, True
        elif types == 'continuous':
            binary, classes = False, False
        else:
            binary, classes = None, None
        # handle outputs
        if task == 'regression':
            p = y = rng.random(self.NUM_SAMPLES)
            master = RegressionMaster(backend=GurobiBackend(), loss=loss)
        else:
            if task == 'binary':
                y = rng.random(self.NUM_SAMPLES).round().astype(int)
                p = y if classes else y.astype(float)
                master = ClassificationMaster(backend=GurobiBackend(), loss=loss, types=types)
            elif task == 'multiclass':
                y = rng.random((self.NUM_SAMPLES, self.NUM_CLASSES)).argmax(axis=1)
                p = y if classes else probabilities.get_onehot(y).astype(float)
                master = ClassificationMaster(backend=GurobiBackend(), loss=loss, types=types)
            elif task == 'multilabel':
                y = rng.random((self.NUM_SAMPLES, self.NUM_CLASSES)).round().astype(int)
                p = y if classes else y.astype(float)
                master = ClassificationMaster(backend=GurobiBackend(), loss=loss, types=types, labelling=True)
            else:
                raise AssertionError(f"Unsupported task '{task}'")
            self.assertEqual(master.binary, binary)
            self.assertEqual(master.classes, classes)
        # check correctness of targets
        learner = self.DummyLearner(regression=task == 'regression')
        MACS(learner=learner, master=master).fit(x, y, iterations=1)
        self.assertEqual(learner.p.shape, p.shape)
        self.assertEqual(learner.p.dtype, p.dtype)
        msg = f'Loss: {loss}\n\nRef:\n{learner.p}\n\nAct:\n{p}'
        self.assertTrue(np.allclose(learner.p, p, atol=10 ** -self.PLACES), msg=msg)

    def test_regression(self):
        self._test(task='regression', loss='mae')
        self._test(task='regression', loss='mse')

    def test_binary_discrete(self):
        self._test(task='binary', loss='hd', types='discrete')
        self._test(task='binary', loss='ce', types='discrete')
        self._test(task='binary', loss='mae', types='discrete')
        self._test(task='binary', loss='mse', types='discrete')

    def test_binary_discretized(self):
        self._test(task='binary', loss='ce', types='discretized')
        self._test(task='binary', loss='mae', types='discretized')
        self._test(task='binary', loss='mse', types='discretized')

    def test_binary_continuous(self):
        self._test(task='binary', loss='ce', types='continuous')
        self._test(task='binary', loss='mae', types='continuous')
        self._test(task='binary', loss='mse', types='continuous')

    def test_multiclass_discrete(self):
        self._test(task='multiclass', loss='hd', types='discrete')
        self._test(task='multiclass', loss='ce', types='discrete')
        self._test(task='multiclass', loss='mae', types='discrete')
        self._test(task='multiclass', loss='mse', types='discrete')

    def test_multiclass_discretized(self):
        self._test(task='multiclass', loss='ce', types='discretized')
        self._test(task='multiclass', loss='mae', types='discretized')
        self._test(task='multiclass', loss='mse', types='discretized')

    def test_multiclass_continuous(self):
        self._test(task='multiclass', loss='ce', types='continuous')
        self._test(task='multiclass', loss='mae', types='continuous')
        self._test(task='multiclass', loss='mse', types='continuous')

    def test_multilabel_discrete(self):
        self._test(task='multilabel', loss='hd', types='discrete')
        self._test(task='multilabel', loss='ce', types='discrete')
        self._test(task='multilabel', loss='mae', types='discrete')
        self._test(task='multilabel', loss='mse', types='discrete')

    def test_multilabel_discretized(self):
        self._test(task='multilabel', loss='ce', types='discretized')
        self._test(task='multilabel', loss='mae', types='discretized')
        self._test(task='multilabel', loss='mse', types='discretized')

    def test_multilabel_continuous(self):
        self._test(task='multilabel', loss='ce', types='continuous')
        self._test(task='multilabel', loss='mae', types='continuous')
        self._test(task='multilabel', loss='mse', types='continuous')
