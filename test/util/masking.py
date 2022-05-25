from typing import Optional

import numpy as np
import pandas as pd

from moving_targets.util.masking import get_mask, mask_data
from moving_targets.util.typing import Mask
from test.abstract import AbstractTest


class TestMasking(AbstractTest):
    REF = np.array([0, 1, 2, 3, 4, 5])
    MASK = np.array([True, True, False, True, False, False])

    def _test_get(self, ref: np.ndarray, mask: Optional[Mask]):
        ref_mask = [True] * len(self.MASK) if mask is None else list(self.MASK)
        act_mask = list(get_mask(reference=ref, mask=mask))
        self.assertListEqual(act_mask, ref_mask)

    def test_get_none(self):
        self._test_get(ref=self.REF, mask=None)

    def test_get_nan(self):
        self._test_get(ref=np.where(self.MASK, self.REF, np.nan), mask=np.nan)

    def test_get_val(self):
        self._test_get(ref=np.where(self.MASK, self.REF, -1), mask=-1)

    def test_get_mask(self):
        self._test_get(ref=self.REF, mask=self.MASK)

    def test_masking(self):
        masked = self.REF[self.MASK]
        vectors = [self.REF, pd.Series(self.REF), pd.DataFrame(self.REF), np.array([self.REF] * self.NUM_FEATURES).T]
        ref_vec = [masked, pd.Series(masked), pd.DataFrame(masked), np.array([masked] * self.NUM_FEATURES).T]
        act_vec = mask_data(*vectors, mask=self.MASK)
        for ref, act in zip(ref_vec, act_vec):
            self.assertIsInstance(ref, type(act))
            # deal with matrices by adapting all the other arrays to bi-dimensional shape
            ref, act = np.atleast_2d(ref), np.atleast_2d(act)
            for r, a in zip(ref, act):
                self.assertEqual(list(r), list(a))
