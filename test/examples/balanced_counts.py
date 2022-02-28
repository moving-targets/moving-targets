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
            'train_acc': 0.9375,
            'train_ce': 0.20345882632970388,
            'train_std': 0.011135885079684334,
            'test_acc': 0.8421052631578947,
            'test_ce': 0.26922524852558893,
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
            'train_acc': 0.9375,
            'train_ce': 0.20345882632970388,
            'train_std': 0.011135885079684334,
            'test_acc': 0.8421052631578947,
            'test_ce': 0.26922524852558893,
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
            'train_acc': 0.19849874895746455,
            'train_ce': 2.3545488218311874,
            'train_std': 0.019644963105717457,
            'test_acc': 0.26,
            'test_ce': 2.2849908030558295,
            'test_std': 0.03463099125863358
        },
        'redwine-pretraining-hd': {
            'train_acc': 0.3469557964970809,
            'train_ce': 3.2981379745466124,
            'train_std': 0.014735156641115424,
            'test_acc': 0.3425,
            'test_ce': 3.5058195292908145,
            'test_std': 0.024438130497691963
        },
        'redwine-pretraining-mae': {
            'train_acc': 0.3502919099249375,
            'train_ce': 4.1157856576353895,
            'train_std': 0.004236794732868549,
            'test_acc': 0.3175,
            'test_ce': 4.513083125564008,
            'test_std': 0.019561157657175844
        },
        'redwine-pretraining-mse': {
            'train_acc': 0.33611342785654713,
            'train_ce': 3.6188422377132903,
            'train_std': 0.004236794732868551,
            'test_acc': 0.33,
            'test_ce': 3.8851092142505324,
            'test_std': 0.025481474752367755
        },
        'redwine-projection-ce': {
            'train_acc': 0.3085904920767306,
            'train_ce': 2.159520632453983,
            'train_std': 0.008792529664091449,
            'test_acc': 0.3,
            'test_ce': 2.262814512316213,
            'test_std': 0.027600825269465324
        },
        'redwine-projection-hd': {
            'train_acc': 0.33527939949958296,
            'train_ce': 3.281921168030774,
            'train_std': 0.020363599350330296,
            'test_acc': 0.3375,
            'test_ce': 3.537296087855616,
            'test_std': 0.03975620147292188
        },
        'redwine-projection-mae': {
            'train_acc': 0.33611342785654713,
            'train_ce': 3.993802275532073,
            'train_std': 0.0038042624919110385,
            'test_acc': 0.315,
            'test_ce': 4.42828570095567,
            'test_std': 0.02702879123371142
        },
        'redwine-projection-mse': {
            'train_acc': 0.33527939949958296,
            'train_ce': 3.5560477235702623,
            'train_std': 0.0037428167967295677,
            'test_acc': 0.305,
            'test_ce': 3.92342123429749,
            'test_std': 0.026483747636784502
        },
        'whitewine-pretraining-ce': {
            'train_acc': 0.1478355567655867,
            'train_ce': 2.3447947135652956,
            'train_std': 0.034437080184119756,
            'test_acc': 0.1493877551020408,
            'test_ce': 2.321415703511258,
            'test_std': 0.03626128382267203
        },
        'whitewine-pretraining-hd': {
            'train_acc': 0.27525183773482165,
            'train_ce': 4.111642068175112,
            'train_std': 0.017640498146354475,
            'test_acc': 0.2857142857142857,
            'test_ce': 4.0854369779642,
            'test_std': 0.021969434134164745
        },
        'whitewine-pretraining-mae': {
            'train_acc': 0.267900898448135,
            'train_ce': 4.873340702895356,
            'train_std': 0.0043965546261984555,
            'test_acc': 0.2693877551020408,
            'test_ce': 5.068181626002113,
            'test_std': 0.011047370106393044
        },
        'whitewine-pretraining-mse': {
            'train_acc': 0.25673836101279607,
            'train_ce': 3.696626074171323,
            'train_std': 0.00618057558491891,
            'test_acc': 0.25795918367346937,
            'test_ce': 3.7832073700057216,
            'test_std': 0.013652784493441147
        },
        'whitewine-projection-ce': {
            'train_acc': 0.21372175333514837,
            'train_ce': 2.3140429396642217,
            'train_std': 0.0376628736060425,
            'test_acc': 0.2220408163265306,
            'test_ce': 2.3150875907388735,
            'test_std': 0.03215361474847683
        },
        'whitewine-projection-hd': {
            'train_acc': 0.2632725292676286,
            'train_ce': 2.9219773151650075,
            'train_std': 0.01574437618567817,
            'test_acc': 0.2702040816326531,
            'test_ce': 2.8524658540815455,
            'test_std': 0.01700065972631316
        },
        'whitewine-projection-mae': {
            'train_acc': 0.2640893002994827,
            'train_ce': 4.798740235324526,
            'train_std': 0.0026384832338138004,
            'test_acc': 0.26285714285714284,
            'test_ce': 4.741296865464816,
            'test_std': 0.008594997444985783
        },
        'whitewine-projection-mse': {
            'train_acc': 0.25537707595970593,
            'train_ce': 3.5806835953387157,
            'train_std': 0.007821631371293398,
            'test_acc': 0.2530612244897959,
            'test_ce': 3.588362268141559,
            'test_std': 0.010873658496503827
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
