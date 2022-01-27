from typing import Dict, List

from examples.fair_classification import FairClassification
from moving_targets.learners import Learner, LogisticRegression
from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import DIDI, Metric, Accuracy, CrossEntropy
from test.examples.abstract import TestExamples


class TestFairClassification(TestExamples):
    RESULTS: Dict[str, Dict[str, float]] = {
        'pretraining-ce': {
            'train_acc': 0.8484593961363335,
            'train_ce': 0.8054257266659179,
            'train_didi': 0.21900563218058464,
            'test_acc': 0.8443177297440658,
            'test_ce': 0.858181155909902,
            'test_didi': 0.40581167184915606
        },
        'pretraining-hd': {
            'train_acc': 0.7514698731267406,
            'train_ce': 3.6614882632520014,
            'train_didi': 0.11239426937050906,
            'test_acc': 0.7516244529903195,
            'test_ce': 3.663267204949637,
            'test_didi': 0.18918639898902026
        },
        'pretraining-mae': {
            'train_acc': 0.8462490606074002,
            'train_ce': 0.3305013647046738,
            'train_didi': 0.259439667062541,
            'test_acc': 0.8436546877072006,
            'test_ce': 0.337123136363474,
            'test_didi': 0.3482011586210676
        },
        'pretraining-mse': {
            'train_acc': 0.8462932673179788,
            'train_ce': 0.3310296783739907,
            'train_didi': 0.2629533503358335,
            'test_acc': 0.8441851213366928,
            'test_ce': 0.3374851792593458,
            'test_didi': 0.34737495135656643
        },
        'projection-ce': {
            'train_acc': 0.8477520887670749,
            'train_ce': 0.32796634699589094,
            'train_didi': 0.31988394860656894,
            'test_acc': 0.8463068558546611,
            'test_ce': 0.3344659093437666,
            'test_didi': 0.26151594590002947
        },
        'projection-hd': {
            'train_acc': 0.7514698731267406,
            'train_ce': 3.6614882632520014,
            'train_didi': 0.11239426937050906,
            'test_acc': 0.7516244529903195,
            'test_ce': 3.663267204949637,
            'test_didi': 0.18918639898902026
        },
        'projection-mae': {
            'train_acc': 0.7514698731267406,
            'train_ce': 3.6614882632520014,
            'train_didi': 0.11239426937050906,
            'test_acc': 0.7516244529903195,
            'test_ce': 3.663267204949637,
            'test_didi': 0.18918639898902026
        },
        'projection-mse': {
            'train_acc': 0.7514698731267406,
            'train_ce': 3.6614882632520014,
            'train_didi': 0.11239426937050906,
            'test_acc': 0.7516244529903195,
            'test_ce': 3.663267204949637,
            'test_didi': 0.18918639898902026
        }
    }

    def _stratify(self) -> bool:
        return True

    def _learner(self) -> Learner:
        return LogisticRegression(max_iter=10000, x_scaler='std')

    def _master(self, loss: str) -> Master:
        # we use 'solution_limit' instead of 'time_limit' to avoid time-based stopping criteria which may lead to
        # different results, and we use a value of '2' since the first solution simply returns a vector of zeros
        return FairClassification(backend=GurobiBackend(solution_limit=2), protected='race', loss=loss)

    def _metrics(self) -> List[Metric]:
        return [Accuracy(name='acc'), CrossEntropy(name='ce'), DIDI(protected='race', classification=True, name='didi')]

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
