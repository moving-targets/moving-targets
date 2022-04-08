from typing import Dict, List

from examples.fair_regression import FairRegression
from moving_targets.learners import Learner, LinearRegression
from moving_targets.masters import Master
from moving_targets.metrics import R2, MSE, DIDI, Metric
from test.examples.abstract import TestExamples


class TestFairRegression(TestExamples):
    RESULTS: Dict[str, Dict[str, float]] = {
        'communities-pretraining-mae': {
            'train_r2': 0.6672353190284235,
            'train_mse': 107593.96084721736,
            'train_didi': 0.20000363559036277,
            'test_r2': 0.6789768009383581,
            'test_mse': 103516.5196809794,
            'test_didi': 0.10413275407545175
        },
        'communities-pretraining-mse': {
            'train_r2': 0.6673377143367666,
            'train_mse': 107560.85301628818,
            'train_didi': 0.20042226004852792,
            'test_r2': 0.6789615681097666,
            'test_mse': 103521.43162941586,
            'test_didi': 0.10499622069848828
        },
        'communities-projection-mae': {
            'train_r2': 0.6672508880503402,
            'train_mse': 107588.92686124974,
            'train_didi': 0.20000957104672837,
            'test_r2': 0.678949000809302,
            'test_mse': 103525.48405680954,
            'test_didi': 0.1042080601192551
        },
        'communities-projection-mse': {
            'train_r2': 0.6673360946440641,
            'train_mse': 107561.37671715903,
            'train_didi': 0.20040137356685922,
            'test_r2': 0.6789624120210789,
            'test_mse': 103521.1595034068,
            'test_didi': 0.10494881034643971
        }
    }

    def _stratify(self) -> bool:
        return False

    def _learner(self) -> Learner:
        return LinearRegression(x_scaler='std', y_scaler='norm')

    def _master(self, loss: str) -> Master:
        return FairRegression(backend='gurobi', protected='race', loss=loss)

    def _metrics(self) -> List[Metric]:
        return [R2(name='r2'), MSE(name='mse'), DIDI(protected='race', classification=False, name='didi')]

    def test_pretraining_mae(self):
        self._test(dataset='communities', class_column='violentPerPop', init_step='pretraining', loss='mae')

    def test_pretraining_mse(self):
        self._test(dataset='communities', class_column='violentPerPop', init_step='pretraining', loss='mse')

    def test_projection_mae(self):
        self._test(dataset='communities', class_column='violentPerPop', init_step='projection', loss='mae')

    def test_projection_mse(self):
        self._test(dataset='communities', class_column='violentPerPop', init_step='projection', loss='mse')
