from typing import Optional

import numpy as np
import pandas as pd


def get_mask(reference: np.ndarray, mask: Optional[float]) -> np.ndarray:
    """Returns a masking vector

    :param reference:
        The reference vector on which to check the given mask.

    :param mask:
        Either the masking value used to compute the mask from the reference vector, or None for no masking (in this
        case, a vector of True is returned).

    :return:
        The masking vector.
    """
    if mask is None:
        return np.ones(len(reference), dtype=bool)
    elif np.isnan(mask):
        return ~np.isnan(reference)
    else:
        return ~np.isclose(reference, mask)


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
