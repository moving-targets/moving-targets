from typing import Callable, Any

import numpy as np
import pandas as pd

from moving_targets.learners import Learner
from moving_targets.util.errors import not_implemented_message
from test.abstract import AbstractTest


class TestLearners(AbstractTest):
    @staticmethod
    def _random_state():
        """Defines the random seeds to be fixed before calling the moving targets learner and the reference learner."""
        raise NotImplementedError(not_implemented_message(name='_random_state', static=True))

    def _reference(self, learner, x, y, sample_weight: np.ndarray) -> np.ndarray:
        """Implements the strategy to fit the reference learner and retrieve its predictions.

        :param learner:
            The reference learner.

        :param x:
            The input data.

        :param y:
            The output data.

        :param sample_weight:
            The (optional) sample weights.

        :return:
            The reference predictions.
        """
        raise NotImplementedError(not_implemented_message(name='_reference'))

    def _test(self, mt_learner: Learner, ref_learner: Any, classification: bool, random_state: Callable):
        """Performs the tests on the given data and checks the correctness of the learner wrt to a reference learner.

        :param mt_learner:
            The `Learner` instance to test.

        :param ref_learner:
            The reference learner to be used for ground truth.

        :param classification:
            Whether the model is a regressor or a classifier.

        :param random_state:
            A Callable function without any input parameter to be called before the training to set the random seeds.
        """
        np.random.seed(self.SEED)
        for weights in [True, False]:
            for i in range(self.NUM_TESTS):
                # generate data
                x = np.random.random((self.NUM_SAMPLES, self.NUM_FEATURES))
                y = np.random.choice([0, 1], self.NUM_SAMPLES) if classification else np.random.random(self.NUM_SAMPLES)
                sample_weight = np.random.random(self.NUM_SAMPLES) if weights else None
                # fit and predict using the moving targets learner
                random_state()
                mt_learner.fit(x, y, sample_weight=sample_weight)
                mt_pred = mt_learner.predict(x)
                # fit and predict using the reference learner
                random_state()
                ref_pred = self._reference(ref_learner, x, y, sample_weight)
                # check correctness
                df = pd.DataFrame(mt_pred).join(pd.DataFrame(ref_pred), lsuffix='_mt', rsuffix='_ref')
                self.assertTrue(np.all(mt_pred == ref_pred), msg=f'iteration: {i}, weights: {weights}\n\n\n{df}')
