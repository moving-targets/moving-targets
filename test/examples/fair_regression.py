from typing import Dict, List

from examples.fair_regression import FairRegression
from moving_targets.learners import Learner, LinearRegression
from moving_targets.masters import Master
from moving_targets.metrics import R2, MSE, DIDI, Metric
from test.examples.abstract import TestExamples


class TestFairRegression(TestExamples):
    RESULTS: Dict[str, Dict[str, float]] = {
        'pretraining-mae': {
            'train_r2': 0.6274325395771669,
            'train_mse': 120463.53186474589,
            'train_didi': 0.2000082316422503,
            'test_r2': 0.633682905697627,
            'test_mse': 118121.90150952153,
            'test_didi': 0.11487491288332247
        },
        'pretraining-mse': {
            'train_r2': 0.6673610779801694,
            'train_mse': 107553.29876826465,
            'train_didi': 0.20064353323988612,
            'test_r2': 0.6789660497246872,
            'test_mse': 103519.98649653922,
            'test_didi': 0.10530482332299039
        },
        'projection-mae': {
            'train_r2': 0.6472605828867416,
            'train_mse': 114052.46170760521,
            'train_didi': 0.1972686651703124,
            'test_r2': 0.6217917958567216,
            'test_mse': 121956.28578290988,
            'test_didi': 0.10045722754403157
        },
        'projection-mse': {
            'train_r2': 0.6673626183288317,
            'train_mse': 107552.80072197808,
            'train_didi': 0.20064497977228993,
            'test_r2': 0.6789609209850722,
            'test_mse': 103521.64029999284,
            'test_didi': 0.1053222474384824
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
