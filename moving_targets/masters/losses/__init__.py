"""Moving Targets Losses for Masters."""
from moving_targets.masters.losses.classification import ClassificationLoss, BinaryRegressionLoss, BinarySAE, \
    BinarySSE, BinaryMAE, BinaryMSE, HammingDistance, CrossEntropy, ReversedCrossEntropy, SymmetricCrossEntropy
from moving_targets.masters.losses.loss import Loss
from moving_targets.masters.losses.regression import RegressionLoss, SAE, SSE, MAE, MSE

_REGRESSION_ALIASES: dict = {
    # SAE
    'sae': SAE,
    'sum_of_absolute_errors': SAE,
    'sum of absolute errors': SAE,
    # SSE
    'sse': SSE,
    'sum_of_squared_errors': SSE,
    'sum of squared errors': SSE,
    # MAE
    'mae': MAE,
    'mean_absolute_error': MAE,
    'mean absolute error': MAE,
    # MSE
    'mse': MSE,
    'mean_squared_error': MSE,
    'mean squared error': MSE
}

_CLASSIFICATION_LOSSES: dict = {
    # SAE
    'sae': BinarySAE,
    'sum_of_absolute_errors': BinarySAE,
    'sum of absolute errors': BinarySAE,
    # SSE
    'sse': BinarySSE,
    'sum_of_squared_errors': BinarySSE,
    'sum of squared errors': BinarySSE,
    # MAE
    'mae': BinaryMAE,
    'mean_absolute_error': BinaryMAE,
    'mean absolute error': BinaryMAE,
    # MSE
    'mse': BinaryMSE,
    'mean_squared_error': BinaryMSE,
    'mean squared error': BinaryMSE,
    # HD
    'hd': HammingDistance,
    'hamming_distance': HammingDistance,
    'hamming distance': HammingDistance,
    'bh': HammingDistance,
    'binary_hamming': HammingDistance,
    'binary hamming': HammingDistance,
    'ch': HammingDistance,
    'categorical_hamming': HammingDistance,
    'categorical hamming': HammingDistance,
    # CE
    'ce': CrossEntropy,
    'crossentropy': CrossEntropy,
    'bce': CrossEntropy,
    'binary_crossentropy': CrossEntropy,
    'binary crossentropy': CrossEntropy,
    'cce': CrossEntropy,
    'categorical_crossentropy': CrossEntropy,
    'categorical crossentropy': CrossEntropy,
    'll': CrossEntropy,
    'log_likelihood': CrossEntropy,
    'log likelihood': CrossEntropy,
    'nll': CrossEntropy,
    'negative_log_likelihood': CrossEntropy,
    'negative log likelihood': CrossEntropy,
    # RCE
    'rce': ReversedCrossEntropy,
    'reversed_crossentropy': ReversedCrossEntropy,
    'reversed crossentropy': ReversedCrossEntropy,
    'rbce': ReversedCrossEntropy,
    'reversed_binary_crossentropy': ReversedCrossEntropy,
    'reversed binary crossentropy': ReversedCrossEntropy,
    'rcce': ReversedCrossEntropy,
    'reversed_categorical_crossentropy': ReversedCrossEntropy,
    'reversed categorical crossentropy': ReversedCrossEntropy,
    'rll': ReversedCrossEntropy,
    'reversed_log_likelihood': ReversedCrossEntropy,
    'reversed log likelihood': ReversedCrossEntropy,
    'rnll': ReversedCrossEntropy,
    'reversed_negative_log_likelihood': ReversedCrossEntropy,
    'reversed negative log likelihood': ReversedCrossEntropy,
    # SCE
    'sce': SymmetricCrossEntropy,
    'symmetric_crossentropy': SymmetricCrossEntropy,
    'symmetric crossentropy': SymmetricCrossEntropy,
    'sbce': SymmetricCrossEntropy,
    'symmetric_binary_crossentropy': SymmetricCrossEntropy,
    'symmetric binary crossentropy': SymmetricCrossEntropy,
    'scce': SymmetricCrossEntropy,
    'symmetric_categorical_crossentropy': SymmetricCrossEntropy,
    'symmetric categorical crossentropy': SymmetricCrossEntropy,
    'sll': SymmetricCrossEntropy,
    'symmetric_log_likelihood': SymmetricCrossEntropy,
    'symmetric log likelihood': SymmetricCrossEntropy,
    'snll': SymmetricCrossEntropy,
    'symmetric_negative_log_likelihood': SymmetricCrossEntropy,
    'symmetric negative log likelihood': SymmetricCrossEntropy
}


def regression_loss(alias: str, **loss_kwargs) -> RegressionLoss:
    class_type = _REGRESSION_ALIASES.get(alias)
    assert class_type is not None, f"{alias} is not a valid alias for a regression loss"
    return class_type(**loss_kwargs)


def classification_loss(alias: str, **loss_kwargs) -> ClassificationLoss:
    class_type = _CLASSIFICATION_LOSSES.get(alias)
    assert class_type is not None, f"{alias} is not a valid alias for a classification loss"
    return class_type(**loss_kwargs)
