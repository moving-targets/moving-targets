from typing import Callable

import numpy as np
import pandas as pd

from moving_targets.util.errors import not_implemented_message
from test.test_abstract import TestAbstract


class TestLearners(TestAbstract):
    @classmethod
    def _random_state(cls):
        """Defines the random seeds to be fixed before calling the moving targets learner and the reference learner."""
        raise NotImplementedError(not_implemented_message(name='_random_state', static=True))

    def _reference(self, learner, x, y, sample_weight: np.ndarray) -> np.ndarray:
        """Implements the strategy to fit the reference learner and retrieve its predictions."""
        raise NotImplementedError(not_implemented_message(name='_reference'))

    def _test(self, mt_learner: Callable, ref_learner: Callable, classification: bool):
        """Performs the tests on the given data and checks the correctness of the learner wrt to a reference learner."""
        rng = np.random.default_rng(self.SEED)
        for weights in [True, False]:
            for i in range(self.NUM_TESTS):
                # generate data
                x = rng.random((self.NUM_SAMPLES, self.NUM_FEATURES))
                y = rng.integers(0, 2, self.NUM_SAMPLES) if classification else rng.random(self.NUM_SAMPLES)
                sample_weight = rng.random(self.NUM_SAMPLES) if weights else None
                # fit and predict using the moving targets learner
                self._random_state()
                mt_pred = mt_learner().fit(x, y, sample_weight=sample_weight).predict(x)
                # fit and predict using the reference learner
                self._random_state()
                ref_pred = self._reference(ref_learner(), x, y, sample_weight)
                # check correctness
                df = pd.DataFrame.from_dict({'y': y, 'mt': mt_pred, 'ref': ref_pred})
                diff = np.abs(mt_pred - ref_pred)
                msg = f'iteration: {i}, weights: {weights}, diff: {diff.max()} (sample {diff.argmax()})\n\n\n{df}'
                self.assertTrue(np.allclose(mt_pred, ref_pred, atol=10 ** -self.PLACES), msg=msg)
