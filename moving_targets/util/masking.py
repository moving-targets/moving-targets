from typing import Optional

import numpy as np
import pandas as pd


def get_mask(reference: np.ndarray, mask: Optional[float], all_columns: bool = True) -> np.ndarray:
    """Returns a masking vector

    :param reference:
        The reference vector on which to check the given mask.

    :param mask:
        Either the masking value used to compute the mask from the reference vector, or None for no masking (in this
        case, a vector of True is returned).

    :param all_columns:
        In case of a reference array with more than one dimension, the reference is reshaped into (len(reference), -1).
        If this parameter is set to true, and index gets masked if all the columns match the mask value, otherwise an
        index gets masked if at least one of the columns match the mask value.

    :return:
        The masking vector.
    """
    if mask is None:
        return np.ones(len(reference), dtype=bool)
    reference = np.atleast_2d(reference).reshape((len(reference), -1))
    mask = np.isnan(reference) if np.isnan(mask) else np.isclose(reference, mask)
    mask = np.all(mask, axis=1) if all_columns else np.any(mask, axis=1)
    return ~mask


def mask_data(*data, mask: np.ndarray) -> list:
    """Masks a set of data using the given mask vector.

    :param data:
        The data to mask.

    :param mask:
        The mask vector.

    :return:
        The masked data.
    """
    return [d.iloc[mask] if isinstance(d, (pd.DataFrame, pd.Series)) else d[mask] for d in data]
