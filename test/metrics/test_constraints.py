from typing import Dict

import numpy as np
import pandas as pd

from moving_targets.metrics import ClassFrequenciesStd, MonotonicViolation, DIDI
from moving_targets.util import probabilities
from test.metrics.test_metrics import TestMetrics


class TestConstraintsMetrics(TestMetrics):
    DIDI_DATA: pd.DataFrame = pd.DataFrame.from_dict({
        'bin_protected': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        'multi_protected': [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
        'onehot_protected_0': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        'onehot_protected_1': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        'onehot_protected_2': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        'reg_target': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'bin_target': [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        'multi_target': [1, 2, 0, 0, 0, 0, 1, 1, 1, 2]
    })
    """Dataframe for DIDI tests: it con considers different protected features to test all the possible inputs and
    different target features to test all the possible tasks."""

    DIDI_RESULTS: Dict[str, Dict[str, float]] = {
        'bin_protected': {
            'reg_target': 0.5,
            'bin_target': 0.8,
            'multi_target': 0.8
        },
        'multi_protected': {
            'reg_target': 0.7,
            'bin_target': 1.6333,
            'multi_target': 1.7666
        },
        'onehot_protected': {
            'reg_target': 0.7,
            'bin_target': 1.6333,
            'multi_target': 1.7666
        }
    }
    """Dictionary of results indexed by protected class type and target class type."""

    def test_class_frequencies_std(self):
        def data_generator(rng):
            # create a dictionary of data counts per class
            counts = {c: 1 + rng.integers(self.NUM_SAMPLES) for c in range(self.NUM_CLASSES)}
            values = np.array([v for v in counts.values()]) / np.sum([v for v in counts.values()])
            # create vector of classes and shuffle it, then obtain fake probabilities due to metric compatibility
            classes = np.concatenate([c * np.ones(n, dtype=int) for c, n in counts.items()])
            rng.shuffle(classes)
            # store the class value counts as y term to be used as ground truth
            return [], values, probabilities.get_onehot(vector=classes)

        self._test(data_generator=data_generator,
                   mt_metric=ClassFrequenciesStd(),
                   ref_metric=lambda x, y, p: y.std())

    def test_monotonic_violation(self):
        def diff(v):
            # computes pairwise differences between data points
            return np.array([[vi - vj for vj in v] for vi in v])

        def mono(v):
            # we consider ascending order, thus the expected monotonicities are computed as the sign of the differences
            return np.sign(diff(v))

        def data_generator(rng):
            # no need for y data
            x = rng.random(self.NUM_SAMPLES)
            p = rng.random(self.NUM_SAMPLES)
            # violations are computed by getting only positive differences between pairwise predictions and then
            # masking the values having expected decreasing monotonicity (so that the increasing dual is not counted)
            violations = diff(p)[mono(x) == -1]
            violations[violations < 1e-3] = 0.0
            # store the violations as y term to be used as ground truth
            return x, violations, p

        # test average
        self._test(data_generator=data_generator,
                   mt_metric=MonotonicViolation(monotonicities_fn=mono, aggregation='average'),
                   ref_metric=lambda x, y, p: y.mean())

        # test percentage
        self._test(data_generator=data_generator,
                   mt_metric=MonotonicViolation(monotonicities_fn=mono, aggregation='percentage'),
                   ref_metric=lambda x, y, p: np.sign(y).mean())

        # test feasibility
        self._test(data_generator=data_generator,
                   mt_metric=MonotonicViolation(monotonicities_fn=mono, aggregation='feasible'),
                   ref_metric=lambda x, y, p: float(np.all(y == 0)))

    def test_didi(self):
        for protected, target_dict in self.DIDI_RESULTS.items():
            for target, didi in target_dict.items():
                x = self.DIDI_DATA.drop(columns=[target])
                y = self.DIDI_DATA[target]
                if 'reg' in target:
                    p = y
                    didi_abs = DIDI(classification=False, protected=protected, percentage=False)
                    didi_per = DIDI(classification=False, protected=protected, percentage=True)
                else:
                    p = probabilities.get_onehot(vector=y.values)
                    didi_abs = DIDI(classification=True, protected=protected, percentage=False)
                    didi_per = DIDI(classification=True, protected=protected, percentage=True)
                msg = f"p: '{protected}', t: '{target}'"
                metric_didi_abs, metric_didi_per = didi_abs(x=x, y=y, p=p), didi_per(x=x, y=y, p=p)
                self.assertAlmostEqual(didi, metric_didi_abs, places=self.PLACES, msg=msg)
                self.assertAlmostEqual(1.0, metric_didi_per, places=self.PLACES, msg=msg)
