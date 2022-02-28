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
            'train_acc': 0.8464258874497149,
            'train_ce': 0.32723644103481114,
            'train_didi': 0.2686298053945948,
            'test_acc': 0.8448481633735578,
            'test_ce': 0.3338098009458882,
            'test_didi': 0.38514044533373576
        },
        'pretraining-hd': {
            'train_acc': 0.8406348083639097,
            'train_ce': 1.5722980342687716,
            'train_didi': 0.22082223974210263,
            'test_acc': 0.8415329531892322,
            'test_ce': 1.6632173225965998,
            'test_didi': 0.2880360610111668
        },
        'pretraining-mae': {
            'train_acc': 0.7512488395738474,
            'train_ce': 4.057117828731541,
            'train_didi': 0.014369104265741079,
            'test_acc': 0.7510940193608274,
            'test_ce': 4.058965047539217,
            'test_didi': 0.0
        },
        'pretraining-mse': {
            'train_acc': 0.7510720127315327,
            'train_ce': 1.300984522733852,
            'train_didi': 0.0,
            'test_acc': 0.7510940193608274,
            'test_ce': 1.303033491486402,
            'test_didi': 0.0
        },
        'projection-ce': {
            'train_acc': 0.8221564033420273,
            'train_ce': 0.36713668739106503,
            'train_didi': 0.14988098539710373,
            'test_acc': 0.8245590770454847,
            'test_ce': 0.369416393502257,
            'test_didi': 0.1451249778538799
        },
        'projection-hd': {
            'train_acc': 0.8465143008708722,
            'train_ce': 1.3273700399501647,
            'train_didi': 0.21434295815856858,
            'test_acc': 0.845245988595677,
            'test_ce': 1.4050198331575598,
            'test_didi': 0.32160864501229514
        },
        'projection-mae': {
            'train_acc': 0.8464700941602935,
            'train_ce': 0.5276367968094331,
            'train_didi': 0.19692639632575293,
            'test_acc': 0.8423286036334704,
            'test_ce': 0.5554519401155185,
            'test_didi': 0.3406549905425884
        },
        'projection-mse': {
            'train_acc': 0.8473542283718668,
            'train_ce': 0.39180277939327857,
            'train_didi': 0.24245036155113317,
            'test_acc': 0.8445829465588118,
            'test_ce': 0.40707912342906016,
            'test_didi': 0.3311465584675247
        }
    }

    def _stratify(self) -> bool:
        return True

    def _learner(self) -> Learner:
        return LogisticRegression(max_iter=10000, x_scaler='std')

    def _master(self, loss: str) -> Master:
        # we use 'solution_limit' instead of 'time_limit' to avoid time-based stopping criteria
        return FairClassification(backend=GurobiBackend(solution_limit=3), protected='race', loss=loss)

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
