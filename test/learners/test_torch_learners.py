import random

import numpy as np
import torch

from moving_targets.learners import TorchMLP
from test.learners.test_learners import TestLearners


# TORCH IS IMPORTED LAZILY TO AVOID CONFLICTS WITH DEPENDENCIES TESTS

class TestTorchLearners(TestLearners):
    _EPOCHS: int = 2
    _UNITS: int = 2
    _BATCH_SIZE: int = 4
    _OPTIMIZER: str = 'RMSprop'
    _SHUFFLE: bool = True

    @classmethod
    def _random_state(cls):
        import torch
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

    def _reference(self, learner, x, y, sample_weight) -> np.ndarray:
        from moving_targets.learners import TorchLearner
        from torch.nn import Sequential, Linear, ReLU, Sigmoid, MSELoss, BCELoss
        from torch.optim import RMSprop
        from torch.utils.data import DataLoader

        if learner == 'regression_mlp':
            learner = Sequential(Linear(self.NUM_FEATURES, self._UNITS), ReLU(), Linear(self._UNITS, 1))
            loss = MSELoss()
        elif learner == 'classification_mlp':
            learner = Sequential(Linear(self.NUM_FEATURES, self._UNITS), ReLU(), Linear(self._UNITS, 1), Sigmoid())
            loss = BCELoss()
        else:
            raise AssertionError(f"Unknown reference learner '{learner}'")

        if self._OPTIMIZER == 'RMSprop':
            optimizer = RMSprop(learner.parameters())
        else:
            raise AssertionError(f"Unknown optimizer '{self._OPTIMIZER}'")

        # noinspection PyTypeChecker
        loader = DataLoader(dataset=TorchLearner._Dataset(x=x, y=y), batch_size=self._BATCH_SIZE, shuffle=self._SHUFFLE)
        learner.train()
        for i in range(self._EPOCHS):
            for inp, out in loader:
                optimizer.zero_grad()
                pred = learner(inp)
                loss(pred, out).backward()
                optimizer.step()
        learner.eval()
        return learner(torch.tensor(np.array(x), dtype=torch.float32)).detach().numpy().squeeze()

    def test_regression_mlp(self):
        self._test(
            mt_learner=lambda: TorchMLP(
                loss='MSELoss',
                optimizer=self._OPTIMIZER,
                input_units=self.NUM_FEATURES,
                hidden_units=[self._UNITS],
                iterations=self._EPOCHS,
                batch_size=self._BATCH_SIZE,
                shuffle=self._SHUFFLE,
                verbose=False
            ),
            ref_learner=lambda: 'regression_mlp',
            classification=False)

    def test_classification_mlp(self):
        self._test(
            mt_learner=lambda: TorchMLP(
                loss='BCELoss',
                output_activation='Sigmoid',
                optimizer=self._OPTIMIZER,
                input_units=self.NUM_FEATURES,
                hidden_units=[self._UNITS],
                iterations=self._EPOCHS,
                batch_size=self._BATCH_SIZE,
                shuffle=self._SHUFFLE,
                verbose=False
            ),
            ref_learner=lambda: 'classification_mlp',
            classification=True
        )
