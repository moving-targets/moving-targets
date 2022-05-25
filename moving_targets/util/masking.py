from typing import Optional

import numpy as np
import pandas as pd

from moving_targets.util.typing import Mask, is_numeric


def get_mask(reference: np.ndarray, mask: Optional[Mask]) -> np.ndarray:
    """Returns a masking vector

    :param reference:
        The reference vector on which to check the given mask.

    :param mask:
        Either an explicit mask vector (which is returned without any change), a masking value to be excluded, or None
        for no masking (in which case, a vector of True is returned).

    :return:
        The masking vector.
    """
    if mask is None:
        return np.ones(len(reference), dtype=bool)
    elif is_numeric(mask):
        mask = np.isnan(reference) if np.isnan(mask) else np.isclose(reference, mask)
        return ~mask
    else:
        lr, lm = len(reference), len(mask)
        assert lr == lm, f"The length of the given array ({lr}) does not coincide with the length of the mask ({lm})"
        return mask


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
