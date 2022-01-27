from typing import Dict, List

from examples.balanced_counts import BalancedCounts
from moving_targets.learners import LogisticRegression, Learner
from moving_targets.masters import Master
from moving_targets.metrics import Accuracy, CrossEntropy, ClassFrequenciesStd, Metric
from test.examples.abstract import TestExamples


class TestBalancedCounts(TestExamples):
    RESULTS: Dict[str, Dict[str, float]] = {
        'iris-pretraining-ce': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1547145561320399,
            'train_std': 0.0042089689356342224,
            'test_acc': 0.9736842105263158,
            'test_ce': 0.15156964302010467,
            'test_std': 0.03282155602433281
        },
        'iris-pretraining-hd': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1547145561320399,
            'train_std': 0.0042089689356342224,
            'test_acc': 0.9736842105263158,
            'test_ce': 0.15156964302010467,
            'test_std': 0.03282155602433281
        },
        'iris-pretraining-mae': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1547145561320399,
            'train_std': 0.0042089689356342224,
            'test_acc': 0.9736842105263158,
            'test_ce': 0.15156964302010467,
            'test_std': 0.03282155602433281
        },
        'iris-pretraining-mse': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1547145561320399,
            'train_std': 0.0042089689356342224,
            'test_acc': 0.9736842105263158,
            'test_ce': 0.15156964302010467,
            'test_std': 0.03282155602433281
        },
        'iris-projection-ce': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1547145561320399,
            'train_std': 0.0042089689356342224,
            'test_acc': 0.9736842105263158,
            'test_ce': 0.15156964302010467,
            'test_std': 0.03282155602433281
        },
        'iris-projection-hd': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1547145561320399,
            'train_std': 0.0042089689356342224,
            'test_acc': 0.9736842105263158,
            'test_ce': 0.15156964302010467,
            'test_std': 0.03282155602433281
        },
        'iris-projection-mae': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1547145561320399,
            'train_std': 0.0042089689356342224,
            'test_acc': 0.9736842105263158,
            'test_ce': 0.15156964302010467,
            'test_std': 0.03282155602433281
        },
        'iris-projection-mse': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1547145561320399,
            'train_std': 0.0042089689356342224,
            'test_acc': 0.9736842105263158,
            'test_ce': 0.15156964302010467,
            'test_std': 0.03282155602433281
        },
        'redwine-pretraining-ce': {
            'train_acc': 0.31693077564637195,
            'train_ce': 4.161992535040387,
            'train_std': 0.0041817094680265034,
            'test_acc': 0.275,
            'test_ce': 4.3357722699363626,
            'test_std': 0.02616719744684597
        },
        'redwine-pretraining-hd': {
            'train_acc': 0.3419516263552961,
            'train_ce': 2.473778301579593,
            'train_std': 0.012752878735967211,
            'test_acc': 0.335,
            'test_ce': 2.7854796051843778,
            'test_std': 0.03962497809322914
        },
        'redwine-pretraining-mae': {
            'train_acc': 0.34528773978315264,
            'train_ce': 3.8832158073552527,
            'train_std': 0.0033157799393596205,
            'test_acc': 0.3175,
            'test_ce': 4.252806309359823,
            'test_std': 0.018689717910004844
        },
        'redwine-pretraining-mse': {
            'train_acc': 0.34528773978315264,
            'train_ce': 3.8832158073552527,
            'train_std': 0.0033157799393596205,
            'test_acc': 0.3175,
            'test_ce': 4.252806309359823,
            'test_std': 0.018689717910004844
        },
        'redwine-projection-ce': {
            'train_acc': 0.3244370308590492,
            'train_ce': 3.927653197928465,
            'train_std': 0.003711712516769167,
            'test_acc': 0.305,
            'test_ce': 3.9304102312405327,
            'test_std': 0.03236081306491267
        },
        'redwine-projection-hd': {
            'train_acc': 0.3035863219349458,
            'train_ce': 2.655716345841689,
            'train_std': 0.012252144981607071,
            'test_acc': 0.295,
            'test_ce': 2.712843292355665,
            'test_std': 0.03572658518371003
        },
        'redwine-projection-mae': {
            'train_acc': 0.32360300250208507,
            'train_ce': 3.4802111797112554,
            'train_std': 0.003711712516769165,
            'test_acc': 0.305,
            'test_ce': 3.47630616742298,
            'test_std': 0.03448026810929533
        },
        'redwine-projection-mse': {
            'train_acc': 0.32360300250208507,
            'train_ce': 3.4802111797112554,
            'train_std': 0.003711712516769165,
            'test_acc': 0.305,
            'test_ce': 3.47630616742298,
            'test_std': 0.03448026810929533
        },
        'whitewine-pretraining-ce': {
            'train_acc': 0.24938742172610945,
            'train_ce': 5.770802759957259,
            'train_std': 0.0029814115198532622,
            'test_acc': 0.25551020408163266,
            'test_ce': 5.8012439618349045,
            'test_std': 0.010353426066205498
        },
        'whitewine-pretraining-hd': {
            'train_acc': 0.2531990198747618,
            'train_ce': 2.2207467657139723,
            'train_std': 0.02018362684505328,
            'test_acc': 0.26448979591836735,
            'test_ce': 2.241166141986775,
            'test_std': 0.024365085771505573
        },
        'whitewine-pretraining-mae': {
            'train_acc': 0.2654505853525728,
            'train_ce': 4.134924524371716,
            'train_std': 0.003539982187540821,
            'test_acc': 0.26285714285714284,
            'test_ce': 4.352087420508572,
            'test_std': 0.012084468976026944
        },
        'whitewine-pretraining-mse': {
            'train_acc': 0.2654505853525728,
            'train_ce': 4.134924524371716,
            'train_std': 0.003539982187540821,
            'test_acc': 0.26285714285714284,
            'test_ce': 4.352087420508572,
            'test_std': 0.012084468976026944
        },
        'whitewine-projection-ce': {
            'train_acc': 0.26953444051184317,
            'train_ce': 5.3254359147753005,
            'train_std': 0.003452144927154826,
            'test_acc': 0.28816326530612246,
            'test_ce': 5.360897783974734,
            'test_std': 0.012472138283195454
        },
        'whitewine-projection-hd': {
            'train_acc': 0.2640893002994827,
            'train_ce': 2.3050433493921583,
            'train_std': 0.017544795036764767,
            'test_acc': 0.27183673469387754,
            'test_ce': 2.3118629639025694,
            'test_std': 0.01939159084351708
        },
        'whitewine-projection-mae': {
            'train_acc': 0.2632725292676286,
            'train_ce': 4.080182872362976,
            'train_std': 0.006540184772544018,
            'test_acc': 0.28244897959183674,
            'test_ce': 4.1319337603030455,
            'test_std': 0.012661522610509587
        },
        'whitewine-projection-mse': {
            'train_acc': 0.2632725292676286,
            'train_ce': 4.080182872362976,
            'train_std': 0.006540184772544018,
            'test_acc': 0.28244897959183674,
            'test_ce': 4.1319337603030455,
            'test_std': 0.012661522610509587
        }
    }

    def _stratify(self) -> bool:
        return True

    def _learner(self) -> Learner:
        return LogisticRegression(max_iter=10000, x_scaler='std')

    def _master(self, loss: str) -> Master:
        return BalancedCounts(backend='gurobi', loss=loss)

    def _metrics(self) -> List[Metric]:
        return [Accuracy(name='acc'), CrossEntropy(name='ce'), ClassFrequenciesStd(name='std')]

    def _results(self, dataset: str, class_column: str, init_step: str, loss: str) -> Dict[str, float]:
        return self.RESULTS[f'{dataset}-{init_step}-{loss}']

    def test_iris_pretraining_hd(self):
        self._test(dataset='iris', class_column='class', init_step='pretraining', loss='hd')

    def test_iris_pretraining_ce(self):
        self._test(dataset='iris', class_column='class', init_step='pretraining', loss='ce')

    def test_iris_pretraining_mae(self):
        self._test(dataset='iris', class_column='class', init_step='pretraining', loss='mae')

    def test_iris_pretraining_mse(self):
        self._test(dataset='iris', class_column='class', init_step='pretraining', loss='mse')

    def test_iris_projection_hd(self):
        self._test(dataset='iris', class_column='class', init_step='projection', loss='hd')

    def test_iris_projection_ce(self):
        self._test(dataset='iris', class_column='class', init_step='projection', loss='ce')

    def test_iris_projection_mae(self):
        self._test(dataset='iris', class_column='class', init_step='projection', loss='mae')

    def test_iris_projection_mse(self):
        self._test(dataset='iris', class_column='class', init_step='projection', loss='mse')

    def test_redwine_pretraining_hd(self):
        self._test(dataset='redwine', class_column='quality', init_step='pretraining', loss='hd')

    def test_redwine_pretraining_ce(self):
        self._test(dataset='redwine', class_column='quality', init_step='pretraining', loss='ce')

    def test_redwine_pretraining_mae(self):
        self._test(dataset='redwine', class_column='quality', init_step='pretraining', loss='mae')

    def test_redwine_pretraining_mse(self):
        self._test(dataset='redwine', class_column='quality', init_step='pretraining', loss='mse')

    def test_redwine_projection_hd(self):
        self._test(dataset='redwine', class_column='quality', init_step='projection', loss='hd')

    def test_redwine_projection_ce(self):
        self._test(dataset='redwine', class_column='quality', init_step='projection', loss='ce')

    def test_redwine_projection_mae(self):
        self._test(dataset='redwine', class_column='quality', init_step='projection', loss='mae')

    def test_redwine_projection_mse(self):
        self._test(dataset='redwine', class_column='quality', init_step='projection', loss='mse')

    def test_whitewine_pretraining_hd(self):
        self._test(dataset='whitewine', class_column='quality', init_step='pretraining', loss='hd')

    def test_whitewine_pretraining_ce(self):
        self._test(dataset='whitewine', class_column='quality', init_step='pretraining', loss='ce')

    def test_whitewine_pretraining_mae(self):
        self._test(dataset='whitewine', class_column='quality', init_step='pretraining', loss='mae')

    def test_whitewine_pretraining_mse(self):
        self._test(dataset='whitewine', class_column='quality', init_step='pretraining', loss='mse')

    def test_whitewine_projection_hd(self):
        self._test(dataset='whitewine', class_column='quality', init_step='projection', loss='hd')

    def test_whitewine_projection_ce(self):
        self._test(dataset='whitewine', class_column='quality', init_step='projection', loss='ce')

    def test_whitewine_projection_mae(self):
        self._test(dataset='whitewine', class_column='quality', init_step='projection', loss='mae')

    def test_whitewine_projection_mse(self):
        self._test(dataset='whitewine', class_column='quality', init_step='projection', loss='mse')
