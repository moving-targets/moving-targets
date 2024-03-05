from typing import Optional

import numpy as np
import pandas as pd

from moving_targets.util.masking import get_mask, mask_data
from test.test_abstract import TestAbstract


class TestMasking(TestAbstract):
    REF_1D = [0., 1., 2., 3., 4., 5.]
    REF_2D = [[i] * TestAbstract.NUM_CLASSES for i in REF_1D]
    MASK = np.array([True, True, False, True, False, False])

    def _test_get(self, ref: np.ndarray, mask: Optional[float], all_columns: bool, should_fail: bool):
        ref_mask = [True] * len(self.MASK) if mask is None else list(self.MASK)
        act_mask = list(get_mask(reference=ref, mask=mask, all_columns=all_columns))
        is_equal = np.equal(ref_mask, act_mask).all()
        msg = f"\nMasks {ref_mask} and {act_mask} are "
        if should_fail:
            self.assertFalse(is_equal, msg=msg + "not expected to be equal, but they are.")
        else:
            self.assertTrue(is_equal, msg=msg + "expected to be equal, but they are not.")

    def test_get_none_1d(self):
        self._test_get(ref=np.array(self.REF_1D), mask=None, all_columns=True, should_fail=False)
        self._test_get(ref=np.array(self.REF_1D), mask=None, all_columns=False, should_fail=False)

    def test_get_nan_1d(self):
        ref = np.where(self.MASK, self.REF_1D, np.nan)
        self._test_get(ref=ref, mask=np.nan, all_columns=True, should_fail=False)
        self._test_get(ref=ref, mask=np.nan, all_columns=False, should_fail=False)

    def test_get_val_1d(self):
        ref = np.where(self.MASK, self.REF_1D, -1)
        self._test_get(ref=ref, mask=-1, all_columns=True, should_fail=False)
        self._test_get(ref=ref, mask=-1, all_columns=True, should_fail=False)

    def test_get_none_2d(self):
        self._test_get(ref=np.array(self.REF_2D), mask=None, all_columns=True, should_fail=False)
        self._test_get(ref=np.array(self.REF_2D), mask=None, all_columns=False, should_fail=False)

    def test_get_nan_all(self):
        ref = np.array(self.REF_2D)
        for index in np.arange(len(self.MASK))[~self.MASK]:
            ref[index, :] = np.nan
        self._test_get(ref=ref, mask=np.nan, all_columns=True, should_fail=False)
        self._test_get(ref=ref, mask=np.nan, all_columns=False, should_fail=False)

    def test_get_val_all(self):
        ref = np.array(self.REF_2D)
        for index in np.arange(len(self.MASK))[~self.MASK]:
            ref[index, :] = -1
        self._test_get(ref=ref, mask=-1, all_columns=True, should_fail=False)
        self._test_get(ref=ref, mask=-1, all_columns=False, should_fail=False)

    def test_get_nan_any(self):
        rng = np.random.default_rng(self.SEED)
        ref = np.array(self.REF_2D)
        for index in np.arange(len(self.MASK))[~self.MASK]:
            ref[index, rng.integers(0, self.NUM_CLASSES, size=1)] = np.nan
        self._test_get(ref=ref, mask=np.nan, all_columns=False, should_fail=False)
        self._test_get(ref=ref, mask=np.nan, all_columns=True, should_fail=True)

    def test_get_val_any(self):
        rng = np.random.default_rng(self.SEED)
        ref = np.array(self.REF_2D)
        for index in np.arange(len(self.MASK))[~self.MASK]:
            ref[index, rng.integers(0, self.NUM_CLASSES, size=1)] = -1
        self._test_get(ref=ref, mask=-1, all_columns=False, should_fail=False)
        self._test_get(ref=ref, mask=-1, all_columns=True, should_fail=True)

    def test_masking(self):
        masked = np.array(self.REF_1D)[self.MASK]
        vectors = [
            np.array(self.REF_1D),
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
