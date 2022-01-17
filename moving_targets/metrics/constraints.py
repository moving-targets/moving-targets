"""Constraints Metrics."""

from typing import Optional, Callable, List, Union, Dict

import numpy as np
import pandas as pd

from moving_targets.metrics.metric import Metric
from moving_targets.util import probabilities


class ClassFrequenciesStd(Metric):
    """Standard Deviation of the Class Frequencies, usually constrained to be null."""

    def __init__(self, classes: Optional[int] = None, name: str = 'std'):
        """
        :param classes:
            The number of classes or None for automatic class inference.

        :param name:
            The name of the metric.
        """
        super(ClassFrequenciesStd, self).__init__(name=name)

        self.classes: Optional[int] = classes
        """The number of classes or None for automatic class inference."""

    def __call__(self, x, y, p) -> float:
        # bincount is similar to np.unique(..., return_counts=True) but allows to fix a minimum number of classes
        # in this way, if the predictions are all the same, the counts will be [n, 0, ..., 0] instead of [n]
        num_classes = (1 + y.astype(int).max()) if self.classes is None else self.classes
        classes_counts = np.bincount(probabilities.get_classes(p), minlength=num_classes) / len(p)
        return classes_counts.std()


class DIDI(Metric):
    """Disparate Impact Discrimination Index."""

    @staticmethod
    def get_indicator_matrix(x: pd.DataFrame, protected: str) -> np.array:
        """Computes the indicator matrix given the input data and a protected feature.

        :param x:
            The input data (it must be a pandas dataframe because features are searched by column name).

        :param protected:
            The name of the protected feature.

            During the solving process, the algorithm checks inside the data for column names starting with the given
            feature name: if a single column is found this is assumed to be a categorical column (thus, the number of
            classes is inferred from the unique values in the column), otherwise, if multiple columns are found this is
            assumed to be a one-hot encoded version of the categorical column (thus, the number of classes is inferred
            from the number of columns).

        :return:
            The indicator matrix, i.e., a matrix in which the i-th row represents a boolean vector stating whether or
            not the j-th sample (represented by the j-th column) is part of the i-th protected group.

        :raise `AssertionError`:
            If there is no column name starting with the `protected` string.
        """
        features = [c for c in x.columns if c.startswith(protected)]
        assert len(features) > 0, AssertionError(f"No column starting with the given protected feature '{protected}'")
        matrix = x[features].values.astype(int).squeeze()
        # if a single protected column is found out, this is interpreted as a categorical column thus is onehot encoded,
        # otherwise, if multiple protected columns are found out, these are interpreted as already encoded columns
        matrix = probabilities.get_onehot(matrix, onehot_binary=True) if matrix.ndim == 1 else matrix
        return matrix.transpose().astype(bool)

    @staticmethod
    def classification_didi(indicator_matrix: np.ndarray, targets: np.ndarray):
        """Computes the Disparate Impact Discrimination Index for Classification Tasks.

        :param indicator_matrix:
            A matrix which associates to each row a masking vector of the samples belonging to the respective group.

        :param targets:
            The output classes.

        :return:
            The (absolute) value of the DIDI.
        """
        didi, classes = 0.0, len(np.unique(targets))
        # compute averages per class by onehot encoding the class targets then aggregating by column (i.e., class)
        total_averages_per_class = probabilities.get_onehot(targets, classes, True).mean(axis=0)
        for protected_group in indicator_matrix:
            # subset of the targets having <label> as protected feature (i.e., the current protected group)
            protected_targets = targets[protected_group]
            if len(protected_targets) > 0:
                # list of deviations from the total percentage of samples respectively to each target class
                protected_averages_per_class = probabilities.get_onehot(protected_targets, classes, True).mean(axis=0)
                # total deviation (partial didi) respectively to each protected group
                didi += np.abs(total_averages_per_class - protected_averages_per_class).sum()
        return didi

    @staticmethod
    def regression_didi(indicator_matrix: np.ndarray, targets: np.ndarray):
        """Computes the Disparate Impact Discrimination Index for Regression Tasks.

        :param indicator_matrix:
            A matrix which associates to each row a masking vector of the samples belonging to the respective group.

        :param targets:
            The output values.

        :return:
            The (absolute) value of the DIDI.
        """
        didi = 0.0
        total_average = np.mean(targets)
        for protected_group in indicator_matrix:
            # subset of the targets having <label> as protected feature (i.e., the current protected group)
            protected_targets = targets[protected_group]
            if len(protected_targets) > 0:
                # total deviation (partial didi) respectively to each protected group
                protected_average = np.mean(protected_targets)
                didi += abs(protected_average - total_average)
        return didi

    def __init__(self, classification: bool, protected: str, percentage: bool = True, name: str = 'didi'):
        """
        :param classification:
            Whether the task is a classification (True) or a regression (False) task.

        :param protected:
            The name of the protected feature.

            During the solving process, the algorithm checks inside the data for column names starting with the given
            feature name: if a single column is found this is assumed to be a categorical column (thus, the number of
            classes is inferred from the unique values in the column), otherwise, if multiple columns are found this is
            assumed to be a one-hot encoded version of the categorical column (thus, the number of classes is inferred
            from the number of columns).

        :param percentage:
            Whether or not to normalize the DIDI index of the predictions over the DIDI index of the ground truths.

        :param name:
            The name of the metric.
        """
        super(DIDI, self).__init__(name=name)

        self.classification: bool = classification
        """Whether the task is a classification (True) or a regression (False) task."""

        self.protected: str = protected
        """The name of the protected feature."""

        self.percentage: bool = percentage
        """Whether or not to normalize the DIDI index of the predictions over the DIDI index of the ground truths."""

    def __call__(self, x: pd.DataFrame, y, p) -> float:
        m = self.get_indicator_matrix(x=x, protected=self.protected)
        if self.classification:
            didi_p = DIDI.classification_didi(indicator_matrix=m, targets=probabilities.get_classes(p))
            didi_y = DIDI.classification_didi(indicator_matrix=m, targets=y) if self.percentage else 1.0
        else:
            didi_p = DIDI.regression_didi(indicator_matrix=m, targets=p)
            didi_y = DIDI.regression_didi(indicator_matrix=m, targets=y) if self.percentage else 1.0
        # handle division by zero
        if didi_y == 0.0:
            return 0.0 if didi_p == 0.0 else float('inf')
        else:
            return didi_p / didi_y


class MonotonicViolation(Metric):
    """Violation of the Monotonicity Shape Constraint."""

    def __init__(self,
                 monotonicities_fn: Callable,
                 aggregation: str = 'average',
                 eps: float = 1e-3,
                 name: str = 'monotonicity'):
        """
        :param monotonicities_fn:
            Function having signature f(x) -> M where x is the input data and M is the monotonicities matrix, i.e., a
            NxN matrix (where |x| = N) which associates to each entry (i, j) a value of -1, 0, o 1 depending on the
            expected monotonicity between x[i] and x[j].

            E.g., if x = [0, 1, 2, 3], with 0 < 1 < 2 < 3, then M =
                |  0, -1, -1, -1 |
                |  1,  0, -1, -1 |
                |  1,  1,  0, -1 |
                |  1,  1,  1,  0 |

            If the computation of the monotonicities is heavy and you call the metric multiple times with the same
            input, it would be better to precompute the matrix on the input x and pass a fake function that ignores
            the x parameter and just returns the precomputed matrix, e.g:

            .. code-block:: python

                M = monotonicities_fn(x)
                metric = MonotonicViolation(monotonicities_fn=lambda x: M)

        :param aggregation:
            The aggregation type:

            - 'average', which computes the average constraint violation in terms of pure output.
            - 'percentage', which computes the constraint violation in terms of average number of violations.
            - 'feasible', which returns a binary value depending on whether there is at least on violation.

        :param eps:
            The slack value under which a violation is considered to be acceptable.

        :param name:
            The name of the metric.
        """
        super(MonotonicViolation, self).__init__(name=name)

        self.monotonicities_fn: Callable = monotonicities_fn
        """Function having signature f(x) -> M where x is the input data and M is the monotonicities matrix."""

        self.eps: float = eps
        """The slack value under which a violation is considered to be acceptable."""

        self.aggregate: Callable
        """The aggregation function."""

        if aggregation == 'average':
            self.aggregate = lambda violations: np.mean(violations)
        elif aggregation == 'percentage':
            self.aggregate = lambda violations: np.mean(violations > 0)
        elif aggregation == 'feasible':
            self.aggregate = lambda violations: int(np.all(violations <= 0))
        else:
            raise AssertionError(f"'{aggregation}' is not a valid aggregation kind")

    def __call__(self, x, y, p) -> float:
        monotonicities = self.monotonicities_fn(x)
        if np.all(monotonicities == 0):
            return 0.0
        # get pair of higher indices and lower indices by filtering pairs with increasing monotonicities
        monotonicities = [(hi, li) for hi, row in enumerate(monotonicities) for li, m in enumerate(row) if m == 1]
        # compute violations as the difference between p[lower_indices] and p[higher_indices]
        violations = p[[li for _, li in monotonicities]] - p[[hi for hi, _ in monotonicities]]
        # then filter out values under the threshold
        violations[violations < self.eps] = 0.0
        return self.aggregate(violations)


class CausalIndependence(Metric):
    """Measure of Causal Independence obtained as the weight(s) of a linear regressor trained on a sub-dataset with the
    given feature(s) only.

    Let A be the (N, K) matrix of inputs related to the given features, where <N> is the number of data samples and <K>
    is the number of given features, then A[i, j] represents the i-th value of the j-th given feature in the dataset.
    We measure the casual relationship between each given feature and the output by training a linear regressor on the
    pair (A, y), thus solving the linear system:
        A.T @ w = (A.T @ A) @ y,
    with w being the learned weights respective to each one of the given features. The level of independence between
    A[i] and y is thus measured as abs(w[i]), since lower w[i] means that the i-th feature is not informative enough to
    let the regressor capture a trend to predict y.
    """

    def __init__(self, features: List, aggregation: Union[None, str, Callable] = 'sum', name: str = 'independence'):
        """
        :param features:
            The list of features to inspect.

        :param aggregation:
            The aggregation policy in case of multiple features. It can be either a string in ['sum', 'mean', 'max'], a
            custom callable function taking the vector 'w' as parameter, or None to get in output the weight for each
            feature without any aggregation.

        :param name:
            The name of the metric.
        """
        super(CausalIndependence, self).__init__(name=name)

        self.features: List = features
        """The list of features to inspect."""

        self.aggregation: Optional[Callable] = None
        """The aggregation policy in case of multiple features."""

        if aggregation is None:
            # if the given aggregation is None, return the weights as a dictionary indexed by feature
            self.aggregation = lambda w: {f: v for f, v in zip(self.features, w)}
        elif isinstance(aggregation, str):
            # if the given aggregation is a string, use np.sum(), np.mean(), or np.max(), respectively
            assert aggregation in ['sum', 'mean', 'max'], f"'{aggregation}' is not a supported aggregation policy"
            self.aggregation = getattr(np, aggregation)
        else:
            self.aggregation = aggregation

    def __call__(self, x, y, p) -> Union[float, Dict[str, float]]:
        w, _, _, _ = np.linalg.lstsq(x[self.features], p, rcond=None)
        return self.aggregation(np.abs(w))
