from typing import Dict, List

from examples.fair_regression import FairRegression
from moving_targets.learners import Learner, LinearRegression
from moving_targets.masters import Master
from moving_targets.metrics import R2, MSE, DIDI, Metric
from test.examples.abstract import TestExamples


class TestFairRegression(TestExamples):
    RESULTS: Dict[str, Dict[str, float]] = {
        'pretraining-mae': {
            'train_r2': 0.6274578993449247,
            'train_mse': 120455.33220289696,
            'train_didi': 0.20000000001566742,
            'test_r2': 0.6211597739109092,
            'test_mse': 122160.08635677413,
            'test_didi': 0.07344816029747665
        },
        'pretraining-mse': {
            'train_r2': 0.6672933957651843,
            'train_mse': 107575.18269407043,
            'train_didi': 0.19998596826032417,
            'test_r2': 0.6798037421355614,
            'test_mse': 103249.86582242488,
            'test_didi': 0.10502457781834651
        },
        'projection-mae': {
            'train_r2': 0.6506123172166276,
            'train_mse': 112968.7337975184,
            'train_didi': 0.20000000003458698,
            'test_r2': 0.6420186625256126,
            'test_mse': 115433.96949008411,
            'test_didi': 0.1484669648551802
        },
        'projection-mse': {
            'train_r2': 0.667290703804177,
            'train_mse': 107576.05309517903,
            'train_didi': 0.19992813648324625,
            'test_r2': 0.6797794266819954,
            'test_mse': 103257.7065365383,
            'test_didi': 0.10483991326455229
        }
    }

    def _stratify(self) -> bool:
        return False

    def _learner(self) -> Learner:
        return LinearRegression()

    def _master(self, loss: str) -> Master:
        return FairRegression(backend='gurobi', protected='race', loss=loss)

    def _metrics(self) -> List[Metric]:
        return [R2(name='r2'),
                MSE(name='mse'),
                DIDI(protected='race', classification=False, percentage=True, name='didi')]

    def _results(self, dataset: str, class_column: str, init_step: str, loss: str) -> Dict[str, float]:
        return self.RESULTS[f'{init_step}-{loss}']

    def test_pretraining_mae(self):
        self._test(dataset='communities', class_column='violentPerPop', init_step='pretraining', loss='mae')

    def test_pretraining_mse(self):
        self._test(dataset='communities', class_column='violentPerPop', init_step='pretraining', loss='mse')

    def test_projection_mae(self):
        self._test(dataset='communities', class_column='violentPerPop', init_step='projection', loss='mae')

    def test_projection_mse(self):
        self._test(dataset='communities', class_column='violentPerPop', init_step='projection', loss='mse')
