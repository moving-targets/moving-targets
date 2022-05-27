from typing import Optional

import numpy as np
import pandas as pd

from moving_targets.util.masking import get_mask, mask_data
from test.abstract import AbstractTest


class TestMasking(AbstractTest):
    REF_1D = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    REF_2D = np.array([REF_1D] * AbstractTest.NUM_CLASSES).T
    MASK = np.array([True, True, False, True, False, False])

    def _test_get(self, ref: np.ndarray, mask: Optional[float], all_columns: bool):
        ref_mask = [True] * len(self.MASK) if mask is None else list(self.MASK)
        act_mask = list(get_mask(reference=ref, mask=mask, all_columns=all_columns))
        self.assertListEqual(act_mask, ref_mask)

    def test_get_none_1d(self):
        self._test_get(ref=self.REF_1D, mask=None, all_columns=True)
        self._test_get(ref=self.REF_1D, mask=None, all_columns=False)

    def test_get_nan_1d(self):
        ref = np.where(self.MASK, self.REF_1D, np.nan)
        self._test_get(ref=ref, mask=np.nan, all_columns=True)
        self._test_get(ref=ref, mask=np.nan, all_columns=False)

    def test_get_val_1d(self):
        ref = np.where(self.MASK, self.REF_1D, -1)
        self._test_get(ref=ref, mask=-1, all_columns=True)
        self._test_get(ref=ref, mask=-1, all_columns=True)

    def test_get_none_2d(self):
        self._test_get(ref=self.REF_2D, mask=None, all_columns=True)
        self._test_get(ref=self.REF_2D, mask=None, all_columns=False)

    def test_get_nan_all(self):
        ref = self.REF_2D
        for index in np.arange(len(self.MASK))[~self.MASK]:
            ref[index, :] = np.nan
        self._test_get(ref=ref, mask=np.nan, all_columns=True)
        self._test_get(ref=ref, mask=np.nan, all_columns=False)

    def test_get_val_all(self):
        ref = self.REF_2D
        for index in np.arange(len(self.MASK))[~self.MASK]:
            ref[index, :] = -1
        self._test_get(ref=ref, mask=-1, all_columns=True)
        self._test_get(ref=ref, mask=-1, all_columns=False)

    def test_get_nan_any(self):
        rng = np.random.default_rng(self.SEED)
        ref = self.REF_2D
        for index in np.arange(len(self.MASK))[~self.MASK]:
            ref[index, rng.integers(0, self.NUM_CLASSES, size=1)] = np.nan
        self._test_get(ref=ref, mask=np.nan, all_columns=False)
        with self.assertRaises(AssertionError, msg="Mask test does not fail when using all_columns=True"):
            self._test_get(ref=ref, mask=np.nan, all_columns=True)

    def test_get_val_any(self):
        rng = np.random.default_rng(self.SEED)
        ref = self.REF_2D
        for index in np.arange(len(self.MASK))[~self.MASK]:
            ref[index, rng.integers(0, self.NUM_CLASSES, size=1)] = -1
        self._test_get(ref=ref, mask=-1, all_columns=False)
        with self.assertRaises(AssertionError, msg="Mask test does not fail when using all_columns=True"):
            self._test_get(ref=ref, mask=-1, all_columns=True)

    def test_masking(self):
        masked = self.REF_1D[self.MASK]
        vectors = [
            self.REF_1D,
            pd.Series(self.REF_1D),
            pd.DataFrame(self.REF_1D),
            np.array([self.REF_1D] * self.NUM_FEATURES).T
        ]
        ref_vec = [masked, pd.Series(masked), pd.DataFrame(masked), np.array([masked] * self.NUM_FEATURES).T]
        act_vec = mask_data(*vectors, mask=self.MASK)
        for ref, act in zip(ref_vec, act_vec):
            self.assertIsInstance(ref, type(act))
            # deal with matrices by adapting all the other arrays to bi-dimensional shape
            ref, act = np.atleast_2d(ref), np.atleast_2d(act)
            for r, a in zip(ref, act):
                self.assertEqual(list(r), list(a))
