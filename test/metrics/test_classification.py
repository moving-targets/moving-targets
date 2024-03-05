from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from moving_targets.metrics import CrossEntropy, Precision, Recall, F1, Accuracy, AUC
from test.metrics.test_metrics import TestMetrics


class TestClassificationMetrics(TestMetrics):
    @staticmethod
    def _binary_generator(rng):
        y = rng.integers(0, 2, size=TestClassificationMetrics.NUM_SAMPLES)
        p = rng.random(size=TestClassificationMetrics.NUM_SAMPLES)
        return [], y, p

    @staticmethod
    def _multi_generator(rng):
        y = rng.integers(0, TestClassificationMetrics.NUM_CLASSES, size=TestClassificationMetrics.NUM_SAMPLES)
        p = rng.random(size=(TestClassificationMetrics.NUM_SAMPLES, TestClassificationMetrics.NUM_CLASSES))
        p = p / p.sum(axis=1, keepdims=1)
        return [], y, p

    def test_binary_crossentropy(self):
        self._test(data_generator=self._binary_generator,
                   mt_metric=CrossEntropy(),
                   ref_metric=lambda x, y, p: log_loss(y, p))

    def test_multi_crossentropy(self):
        self._test(data_generator=self._multi_generator,
                   mt_metric=CrossEntropy(),
                   ref_metric=lambda x, y, p: log_loss(y, p))

    def test_binary_precision(self):
        self._test(data_generator=self._binary_generator,
                   mt_metric=Precision(),
                   ref_metric=lambda x, y, p: precision_score(y, p.round().astype(int)))

    def test_multi_precision(self):
        self._test(data_generator=self._multi_generator,
                   mt_metric=Precision(average='weighted'),
                   ref_metric=lambda x, y, p: precision_score(y, p.argmax(axis=1), average='weighted'))

    def test_binary_recall(self):
        self._test(data_generator=self._binary_generator,
                   mt_metric=Recall(),
                   ref_metric=lambda x, y, p: recall_score(y, p.round().astype(int)))

    def test_multi_recall(self):
        self._test(data_generator=self._multi_generator,
                   mt_metric=Recall(average='weighted'),
                   ref_metric=lambda x, y, p: recall_score(y, p.argmax(axis=1), average='weighted'))

    def test_binary_f1(self):
        self._test(data_generator=self._binary_generator,
                   mt_metric=F1(),
                   ref_metric=lambda x, y, p: f1_score(y, p.round().astype(int)))

    def test_multi_f1(self):
        self._test(data_generator=self._multi_generator,
                   mt_metric=F1(average='weighted'),
                   ref_metric=lambda x, y, p: f1_score(y, p.argmax(axis=1), average='weighted'))

    def test_binary_accuracy(self):
        self._test(data_generator=self._binary_generator,
                   mt_metric=Accuracy(),
                   ref_metric=lambda x, y, p: accuracy_score(y, p.round().astype(int)))

    def test_multi_accuracy(self):
        self._test(data_generator=self._multi_generator,
                   mt_metric=Accuracy(),
                   ref_metric=lambda x, y, p: accuracy_score(y, p.argmax(axis=1)))

    def test_binary_auc(self):
        self._test(data_generator=self._binary_generator,
                   mt_metric=AUC(),
                   ref_metric=lambda x, y, p: roc_auc_score(y, p))

    def test_multi_auc(self):
        self._test(data_generator=self._multi_generator,
                   mt_metric=AUC(),
                   ref_metric=lambda x, y, p: roc_auc_score(y, p, multi_class='ovo'))
