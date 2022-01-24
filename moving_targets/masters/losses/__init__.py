"""Moving Targets Losses for Masters."""
from moving_targets.masters.losses.classification import ClassificationLoss, HammingDistance, CrossEntropy, \
    ReversedCrossEntropy, SymmetricCrossEntropy
from moving_targets.masters.losses.loss import Loss, WeightedLoss
from moving_targets.masters.losses.regression import RegressionLoss, SAE, SSE, MAE, MSE

aliases: dict = {
    # Sum of Absolute Errors
    'sae': SAE,
    'sum_of_absolute_errors': SAE,
    'sum of absolute errors': SAE,
    # Sum of Squared Errors
    'sse': SSE,
    'sum_of_squared_errors': SSE,
    'sum of squared errors': SSE,
    # Mean Absolute Error
    'mae': MAE,
    'mean_absolute_error': MAE,
    'mean absolute error': MAE,
    # Mean Squared Error
    'mse': MSE,
    'mean_squared_error': MSE,
    'mean squared error': MSE,
    # Hamming Distance
    'hd': HammingDistance,
    'hamming_distance': HammingDistance,
    'hamming distance': HammingDistance,
    'bh': HammingDistance,
    '_hamming': HammingDistance,
    ' hamming': HammingDistance,
    'ch': HammingDistance,
    'categorical_hamming': HammingDistance,
    'categorical hamming': HammingDistance,
    # CrossEntropy
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
    # Reversed CrossEntropy
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
    # Symmetric CrossEntropy
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
"""Dictionary which associates to each loss alias the respective class type."""
