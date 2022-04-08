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
            'train_ce': 0.19236475717832052,
            'train_std': 0.011135885079684334,
            'test_acc': 0.8947368421052632,
            'test_ce': 0.24274953809928973,
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
            'train_ce': 0.19236475717832052,
            'train_std': 0.011135885079684334,
            'test_acc': 0.8947368421052632,
            'test_ce': 0.24274953809928973,
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
            'train_acc': 0.20100083402835697,
            'train_ce': 2.3503051283799477,
            'train_std': 0.01871408398364784,
            'test_acc': 0.2525,
            'test_ce': 2.273321243823182,
            'test_std': 0.030023139224419706
        },
        'redwine-pretraining-hd': {
            'train_acc': 0.3427856547122602,
            'train_ce': 3.6993290755927895,
            'train_std': 0.0162254900332621,
            'test_acc': 0.33,
            'test_ce': 3.944571788044401,
            'test_std': 0.02612735901098481
        },
        'redwine-pretraining-mae': {
            'train_acc': 0.34528773978315264,
            'train_ce': 4.130049656675954,
            'train_std': 0.005195505029477557,
            'test_acc': 0.315,
            'test_ce': 4.544605338663489,
            'test_std': 0.021730674684008827
        },
        'redwine-pretraining-mse': {
            'train_acc': 0.33611342785654713,
            'train_ce': 3.6111965110778663,
            'train_std': 0.005013813955204731,
            'test_acc': 0.3125,
            'test_ce': 3.842640971474335,
            'test_std': 0.017179606773406922
        },
        'redwine-projection-ce': {
            'train_acc': 0.3094245204336947,
            'train_ce': 2.2397569751055455,
            'train_std': 0.02256719360777349,
            'test_acc': 0.3025,
            'test_ce': 2.279483294413772,
            'test_std': 0.0268741924943285
        },
        'redwine-projection-hd': {
            'train_acc': 0.3286071726438699,
            'train_ce': 3.58657269573169,
            'train_std': 0.019021312459555198,
            'test_acc': 0.3125,
            'test_ce': 3.6357981239189443,
            'test_std': 0.03197221015541813
        },
        'redwine-projection-mae': {
            'train_acc': 0.3336113427856547,
            'train_ce': 4.184030243971293,
            'train_std': 0.005520080525900695,
            'test_acc': 0.3075,
            'test_ce': 4.347890388298079,
            'test_std': 0.019075871903766
        },
        'redwine-projection-mse': {
            'train_acc': 0.3311092577147623,
            'train_ce': 3.716897066266321,
            'train_std': 0.007558841325728395,
            'test_acc': 0.2975,
            'test_ce': 3.876185990303711,
            'test_std': 0.018856180831641273
        },
        'whitewine-pretraining-ce': {
            'train_acc': 0.15137489790362102,
            'train_ce': 2.3531113849719727,
            'train_std': 0.03246754397117152,
            'test_acc': 0.15183673469387754,
            'test_ce': 2.334037067457527,
            'test_std': 0.03431346926013386
        },
        'whitewine-pretraining-hd': {
            'train_acc': 0.2638170432888647,
            'train_ce': 4.884817982860021,
            'train_std': 0.008204852845033599,
            'test_acc': 0.26285714285714284,
            'test_ce': 4.864481968993685,
            'test_std': 0.013770812286064617
        },
        'whitewine-pretraining-mae': {
            'train_acc': 0.27007895453307923,
            'train_ce': 4.9960941462946415,
            'train_std': 0.0029742996062860536,
            'test_acc': 0.26857142857142857,
            'test_ce': 5.200245222727124,
            'test_std': 0.010045413629088293
        },
        'whitewine-pretraining-mse': {
            'train_acc': 0.25864416008712227,
            'train_ce': 3.6553436900951355,
            'train_std': 0.006603027234749522,
            'test_acc': 0.2726530612244898,
            'test_ce': 3.708985200097328,
            'test_std': 0.011428571428571429
        },
        'whitewine-projection-ce': {
            'train_acc': 0.2188946365368908,
            'train_ce': 2.2139058656115442,
            'train_std': 0.035464161681708135,
            'test_acc': 0.22612244897959183,
            'test_ce': 2.2084853315746784,
            'test_std': 0.033863849138209404
        },
        'whitewine-projection-hd': {
            'train_acc': 0.26463381432071875,
            'train_ce': 3.3763283963593835,
            'train_std': 0.015311341934975188,
            'test_acc': 0.2612244897959184,
            'test_ce': 3.332430672866433,
            'test_std': 0.018378357567899156
        },
        'whitewine-projection-mae': {
            'train_acc': 0.2657228423631908,
            'train_ce': 4.582663693997462,
            'train_std': 0.004649402513249154,
            'test_acc': 0.2702040816326531,
            'test_ce': 4.618747995649043,
            'test_std': 0.00633827556464196
        },
        'whitewine-projection-mse': {
            'train_acc': 0.2616389872039205,
            'train_ce': 3.6552094005836353,
            'train_std': 0.0055923363954231,
            'test_acc': 0.26448979591836735,
            'test_ce': 3.707547845789974,
            'test_std': 0.008438508461866138
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
