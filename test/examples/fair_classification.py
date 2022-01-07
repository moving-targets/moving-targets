from typing import Dict, List

from examples.fair_classification import FairClassification
from moving_targets.learners import Learner, LogisticRegression
from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import DIDI, Metric, Accuracy, CrossEntropy
from test.examples.abstract import TestExamples


class TestFairClassification(TestExamples):
    RESULTS: Dict[str, Dict[str, float]] = {
        'pretraining-hd': {
            'train_acc': 0.7510720127315327,
            'train_ce': 2.908681588239539,
            'train_didi': 0.0,
            'test_acc': 0.7510940193608274,
            'test_ce': 2.934056866668003,
            'test_didi': 0.0
        },
        'pretraining-ce': {
            'train_acc': 0.7908580522523319,
            'train_ce': 0.5227493416366334,
            'train_didi': 0.22297872296231847,
            'test_acc': 0.7902134995358706,
            'test_ce': 0.5286598802408132,
            'test_didi': 0.25406196759119154
        },
        'pretraining-mae': {
            'train_acc': 0.7895318509349719,
            'train_ce': 0.5201796683134032,
            'train_didi': 0.23614024254872445,
            'test_acc': 0.7896830659063785,
            'test_ce': 0.5267724445479272,
            'test_didi': 0.25531116634220874
        },
        'pretraining-mse': {
            'train_acc': 0.7895318509349719,
            'train_ce': 0.5201796683134032,
            'train_didi': 0.23614024254872445,
            'test_acc': 0.7896830659063785,
            'test_ce': 0.5267724445479272,
            'test_didi': 0.25531116634220874
        },
        'projection-hd': {
            'train_acc': 0.7510720127315327,
            'train_ce': 2.908681588239539,
            'train_didi': 0.0,
            'test_acc': 0.7510940193608274,
            'test_ce': 2.934056866668003,
            'test_didi': 0.0
        },
        'projection-ce': {
            'train_acc': 0.7874541355377747,
            'train_ce': 0.5259544268314297,
            'train_didi': 0.12930132833831512,
            'test_acc': 0.7854395968704416,
            'test_ce': 0.5326381355157793,
            'test_didi': 0.19976697093050635
        },
        'projection-mae': {
            'train_acc': 0.7510720127315327,
            'train_ce': 1.4307889947446413,
            'train_didi': 0.0,
            'test_acc': 0.7510940193608274,
            'test_ce': 1.4351622822879013,
            'test_didi': 0.0
        },
        'projection-mse': {
            'train_acc': 0.7510720127315327,
            'train_ce': 1.4307889947446413,
            'train_didi': 0.0,
            'test_acc': 0.7510940193608274,
            'test_ce': 1.4351622822879013,
            'test_didi': 0.0
        }
    }

    def _stratify(self) -> bool:
        return True

    def _learner(self) -> Learner:
        return LogisticRegression()

    def _master(self, loss: str) -> Master:
        # we use 'solution_limit' instead of 'time_limit' to avoid time-based stopping criteria which may lead to
        # different results, and we use a value of '2' since the first solution simply returns a vector of zeros
        return FairClassification(backend=GurobiBackend(solution_limit=2), protected='race', loss=loss)

    def _metrics(self) -> List[Metric]:
        return [Accuracy(name='acc'),
                CrossEntropy(name='ce'),
                DIDI(protected='race', classification=True, percentage=True, name='didi')]

    def _results(self, dataset: str, class_column: str, init_step: str, loss: str) -> Dict[str, float]:
        return self.RESULTS[f'{init_step}-{loss}']

    def test_pretraining_hd(self):
        self._test(dataset='adult', class_column='income', init_step='pretraining', loss='hd')

    def test_pretraining_ce(self):
        self._test(dataset='adult', class_column='income', init_step='pretraining', loss='ce')

    def test_pretraining_mae(self):
        self._test(dataset='adult', class_column='income', init_step='pretraining', loss='mae')

    def test_pretraining_mse(self):
        self._test(dataset='adult', class_column='income', init_step='pretraining', loss='mse')

    def test_projection_hd(self):
        self._test(dataset='adult', class_column='income', init_step='projection', loss='hd')

    def test_projection_ce(self):
        self._test(dataset='adult', class_column='income', init_step='projection', loss='ce')

    def test_projection_mae(self):
        self._test(dataset='adult', class_column='income', init_step='projection', loss='mae')

    def test_projection_mse(self):
        self._test(dataset='adult', class_column='income', init_step='projection', loss='mse')
