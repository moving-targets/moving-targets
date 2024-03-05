import inspect
from typing import Dict, Optional, Union, List, Set, Any

import moving_targets
from moving_targets.learners import Learner
from moving_targets.masters import Master
from moving_targets.masters.backends import Backend
from moving_targets.masters.losses import Loss
from moving_targets.masters.optimizers import Optimizer
from moving_targets.metrics import Metric
from moving_targets.util.scalers import Scaler
from test.test_abstract import TestAbstract


class TestSignatures(TestAbstract):
    MACS_SIGNATURE: Dict[str, Any] = {
        'master': Master,
        'learner': Learner,
        'init_step': (str, 'pretraining'),
        'metrics': (List[Metric], ()),
        'stats': (Union[bool, List[str]], False)
    }

    LEARNER_SIGNATURE: Dict[str, Any] = {
        'mask': (Optional[float], None),
        'x_scaler': (Union[None, Scaler, str], None),
        'y_scaler': (Union[None, Scaler, str], None),
        'stats': (Union[bool, List[str]], False)
    }

    MASTER_SIGNATURE: Dict[str, Any] = {
        'backend': Union[str, Backend],
        'loss': Union[str, Loss],
        'alpha': Union[str, float, Optimizer],
        'mask': (Optional[float], None),
        'x_scaler': (Union[None, Scaler, str], None),
        'y_scaler': (Union[None, Scaler, str], None),
        'stats': (Union[bool, List[str]], False)
    }

    LOSS_SIGNATURE: Dict[str, Any] = {
        'binary': bool,
        'name': str
    }

    def _test(self, module, signature: Dict[str, Any], excluded: Optional[Set[str]] = None):
        """
        :param module:
            The module to test.

        :param signature:
            A dictionary of annotations (and optional default values) indexed by parameters to match in the signature.

        :param excluded:
            A set of objects names to exclude from the check.
        """
        excluded = excluded or set()
        for name, obj in inspect.getmembers(module):
            if name not in excluded and inspect.isclass(obj):
                params = inspect.signature(obj.__init__).parameters
                for param, ann in signature.items():
                    self.assertIn(param, params, msg=f'\n"{name}" has no "{param}" parameter')
                    if isinstance(ann, tuple):
                        ann, dfl = ann
                        self.assertEqual(dfl, params[param].default, msg=f'"{name}" has wrong "{param}" default value')
                    self.assertEqual(ann, params[param].annotation, msg=f'"{name}" has wrong "{param}" annotation')

    def test_macs(self):
        self._test(moving_targets, self.MACS_SIGNATURE)

    def test_learners(self):
        # exclude the learner interface since it does not have default values
        self._test(moving_targets.learners, self.LEARNER_SIGNATURE, excluded={'Learner'})

    def test_masters(self):
        self._test(moving_targets.masters, self.MASTER_SIGNATURE)
