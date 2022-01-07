from typing import Dict, List

from examples.balanced_counts import BalancedCounts
from moving_targets.learners import LogisticRegression, Learner
from moving_targets.masters import Master
from moving_targets.metrics import Accuracy, CrossEntropy, ClassFrequenciesStd, Metric
from test.examples.abstract import TestExamples


class TestBalancedCounts(TestExamples):
    RESULTS: Dict[str, Dict[str, float]] = {
        'iris-pretraining-hd': {
            'train_acc': 0.9375,
            'train_ce': 0.1548364030984502,
            'train_std': 0.01834647024693148,
            'test_acc': 1.0,
            'test_ce': 0.13842724918813115,
            'test_std': 0.012405382126079794
        },
        'iris-pretraining-ce': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1470686623348843,
            'train_std': 0.011135885079684334,
            'test_acc': 1.0,
            'test_ce': 0.12068688450671879,
            'test_std': 0.012405382126079794
        },
        'iris-pretraining-mae': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1470686623348843,
            'train_std': 0.011135885079684334,
            'test_acc': 1.0,
            'test_ce': 0.12068688450671879,
            'test_std': 0.012405382126079794
        },
        'iris-pretraining-mse': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1470686623348843,
            'train_std': 0.011135885079684334,
            'test_acc': 1.0,
            'test_ce': 0.12068688450671879,
            'test_std': 0.012405382126079794
        },
        'iris-projection-hd': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.15012559937507836,
            'train_std': 0.011135885079684334,
            'test_acc': 1.0,
            'test_ce': 0.12612979029079022,
            'test_std': 0.012405382126079794
        },
        'iris-projection-ce': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1470686623348843,
            'train_std': 0.011135885079684334,
            'test_acc': 1.0,
            'test_ce': 0.12068688450671879,
            'test_std': 0.012405382126079794
        },
        'iris-projection-mae': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1470686623348843,
            'train_std': 0.011135885079684334,
            'test_acc': 1.0,
            'test_ce': 0.12068688450671879,
            'test_std': 0.012405382126079794
        },
        'iris-projection-mse': {
            'train_acc': 0.9553571428571429,
            'train_ce': 0.1470686623348843,
            'train_std': 0.011135885079684334,
            'test_acc': 1.0,
            'test_ce': 0.12068688450671879,
            'test_std': 0.012405382126079794
        },
        'redwine-pretraining-hd': {
            'train_acc': 0.28940783986655544,
            'train_ce': 2.1778089412901007,
            'train_std': 0.018608470848311076,
            'test_acc': 0.2875,
            'test_ce': 2.1981635605783274,
            'test_std': 0.033993463423951896
        },
        'redwine-pretraining-ce': {
            'train_acc': 0.2610508757297748,
            'train_ce': 7.86548342238271,
            'train_std': 0.008149281738726635,
            'test_acc': 0.2175,
            'test_ce': 9.145958931434874,
            'test_std': 0.029391986814247328
        },
        'redwine-pretraining-mae': {
            'train_acc': 0.2827356130108424,
            'train_ce': 6.5635568185894755,
            'train_std': 0.010466049754840744,
            'test_acc': 0.2475,
            'test_ce': 6.855826251536489,
            'test_std': 0.019825628755617202
        },
        'redwine-pretraining-mse': {
            'train_acc': 0.2827356130108424,
            'train_ce': 6.5635568185894755,
            'train_std': 0.010466049754840744,
            'test_acc': 0.2475,
            'test_ce': 6.855826251536489,
            'test_std': 0.019825628755617202
        },
        'redwine-projection-hd': {
            'train_acc': 0.2785654712260217,
            'train_ce': 2.640957929355796,
            'train_std': 0.022597996237082026,
            'test_acc': 0.2825,
            'test_ce': 2.7834480881413834,
            'test_std': 0.024309920243024703
        },
        'redwine-projection-ce': {
            'train_acc': 0.2793994995829858,
            'train_ce': 6.137537796690196,
            'train_std': 0.010024737132111645,
            'test_acc': 0.2625,
            'test_ce': 6.3881577228523705,
            'test_std': 0.02044640691064217
        },
        'redwine-projection-mae': {
            'train_acc': 0.2827356130108424,
            'train_ce': 3.2132576243858546,
            'train_std': 0.02526255540594078,
            'test_acc': 0.2475,
            'test_ce': 3.3371670744748565,
            'test_std': 0.01945436254988125
        },
        'redwine-projection-mse': {
            'train_acc': 0.2827356130108424,
            'train_ce': 3.2132576243858546,
            'train_std': 0.02526255540594078,
            'test_acc': 0.2475,
            'test_ce': 3.3371670744748565,
            'test_std': 0.01945436254988125
        },
        'whitewine-pretraining-hd': {
            'train_acc': 0.19847536074053906,
            'train_ce': 2.1867912295052623,
            'train_std': 0.05402260362354483,
            'test_acc': 0.21224489795918366,
            'test_ce': 2.184982047152967,
            'test_std': 0.05955428346677334
        },
        'whitewine-pretraining-ce': {
            'train_acc': 0.20528178600598965,
            'train_ce': 11.314790860149609,
            'train_std': 0.004432534980330093,
            'test_acc': 0.21387755102040817,
            'test_ce': 11.590548457151248,
            'test_std': 0.020270086840935363
        },
        'whitewine-pretraining-mae': {
            'train_acc': 0.20691532806969778,
            'train_ce': 4.588869965212945,
            'train_std': 0.008229337578741051,
            'test_acc': 0.1820408163265306,
            'test_ce': 4.853415948303999,
            'test_std': 0.017123409766043294
        },
        'whitewine-pretraining-mse': {
            'train_acc': 0.20691532806969778,
            'train_ce': 4.588869965212945,
            'train_std': 0.008229337578741051,
            'test_acc': 0.1820408163265306,
            'test_ce': 4.853415948303999,
            'test_std': 0.017123409766043294
        },
        'whitewine-projection-hd': {
            'train_acc': 0.24911516471549142,
            'train_ce': 2.3867429325709533,
            'train_std': 0.03822520405172239,
            'test_acc': 0.2546938775510204,
            'test_ce': 2.364599981962492,
            'test_std': 0.038371068611755446
        },
        'whitewine-projection-ce': {
            'train_acc': 0.21589980942009257,
            'train_ce': 4.901146493948016,
            'train_std': 0.004422968838498539,
            'test_acc': 0.21714285714285714,
            'test_ce': 4.887949455597371,
            'test_std': 0.013017419253528466
        },
        'whitewine-projection-mae': {
            'train_acc': 0.1987476177511571,
            'train_ce': 3.3426059633790235,
            'train_std': 0.015402360891496951,
            'test_acc': 0.20244897959183675,
            'test_ce': 3.392948385569663,
            'test_std': 0.018676376647619918
        },
        'whitewine-projection-mse': {
            'train_acc': 0.1987476177511571,
            'train_ce': 3.3426059633790235,
            'train_std': 0.015402360891496951,
            'test_acc': 0.20244897959183675,
            'test_ce': 3.392948385569663,
            'test_std': 0.018676376647619918
        }
    }

    def _stratify(self) -> bool:
        return True

    def _learner(self) -> Learner:
        return LogisticRegression()

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
