from typing import Dict, List

from examples.fair_classification import FairClassification
from moving_targets.learners import Learner, LogisticRegression
from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.metrics import DIDI, Metric, Accuracy, CrossEntropy
from test.examples.abstract import TestExamples


class TestFairClassification(TestExamples):
    RESULTS: Dict[str, Dict[str, float]] = {
        'adult-pretraining-ce': {
            'train_acc': 0.8464258874497149,
            'train_ce': 0.3273146658852211,
            'train_didi': 0.26263046393957795,
            'test_acc': 0.84537859700305,
            'test_ce': 0.3336856663273787,
            'test_didi': 0.32541518651841117
        },
        'adult-pretraining-hd': {
            'train_acc': 0.840590601653331,
            'train_ce': 1.6114427450352018,
            'train_didi': 0.2352616299019666,
            'test_acc': 0.8414003447818592,
            'test_ce': 1.7056061163756506,
            'test_didi': 0.2923796901198495
        },
        'adult-pretraining-mae': {
            'train_acc': 0.8481499491622828,
            'train_ce': 0.8208600391856758,
            'train_didi': 0.21561414128655274,
            'test_acc': 0.845245988595677,
            'test_ce': 0.872181988548318,
            'test_didi': 0.35198008105013934
        },
        'adult-pretraining-mse': {
            'train_acc': 0.8473984350824455,
            'train_ce': 0.3975507272845652,
            'train_didi': 0.22452896719584453,
            'test_acc': 0.8445829465588118,
            'test_ce': 0.41351429520578087,
            'test_didi': 0.33413079657672795
        },
        'adult-projection-ce': {
            'train_acc': 0.8467353344237655,
            'train_ce': 0.3272828293160022,
            'train_didi': 0.26176691087685144,
            'test_acc': 0.845113380188304,
            'test_ce': 0.33383612851494265,
            'test_didi': 0.34240540535006486
        },
        'adult-projection-hd': {
            'train_acc': 0.8476194686353389,
            'train_ce': 1.3298530667700381,
            'train_didi': 0.23145880707214603,
            'test_acc': 0.8444503381514388,
            'test_ce': 1.411019984964159,
            'test_didi': 0.43141606612379135
        },
        'adult-projection-mae': {
            'train_acc': 0.8489456699526988,
            'train_ce': 0.733863555987578,
            'train_didi': 0.23234632753361797,
            'test_acc': 0.845113380188304,
            'test_ce': 0.7799597506168637,
            'test_didi': 0.38688779560590514
        },
        'adult-projection-mse': {
            'train_acc': 0.8473542283718668,
            'train_ce': 0.40470002445051095,
            'train_didi': 0.2325922068995007,
            'test_acc': 0.8448481633735578,
            'test_ce': 0.42138531705150434,
            'test_didi': 0.3484499271239641
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
