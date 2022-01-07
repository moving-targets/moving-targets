"""Basic Logger Interface."""
import logging
from typing import Dict, Set, Union, List, Any, Optional

from moving_targets.callbacks.callback import Callback
from moving_targets.util.errors import not_implemented_message


class Logger(Callback):
    """Basic interface for a Moving Targets Logger callback."""

    def __init__(self):
        """"""
        super(Logger, self).__init__()

        self._cache: Dict = {}
        """The internally stored cache."""

    def log(self, **cache):
        """Adds the given keyword argument to the inner cache.

        :param cache:
            Key-value pairs to be added to the cache (if the key is already present, the value is overwrote).
        """
        self._cache.update(cache)


class StatsLogger(Logger):

    @staticmethod
    def _parameters() -> Set[str]:
        """The set of all the possible parameters whose statistics may be logged."""
        raise NotImplementedError(not_implemented_message('_parameters', static=True))

    def __init__(self, stats: Union[bool, List[str]], logger: Optional[str]):
        """
        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param logger:
            Either a custom logger name or None (in that case, the root logger is used and no prefix is added).
        """
        super(StatsLogger, self).__init__()

        # handle stats logging ache optionally check for parameter correctness
        if isinstance(stats, bool):
            stats = self._parameters() if stats else set()
        else:
            parameters = self._parameters()
            for param in stats:
                assert param in parameters, f"Parameter '{param}' is not valid for stats logging."

        self.parameters: Set[str] = set(stats)
        """The set of parameters whose statistics must be logged."""

        self.logger: Optional[str] = logger
        """Either a custom logger name or None (in that case, the root logger is used and no prefix is added)."""

    def _log_stats(self, **params: Any):
        """Logs the statistics about the parameters if they are included in the inner 'parameters' field.

        :param params:
            A dictionary which associates to each parameter name its respective value.
        """
        to_cache = {}
        # handle logger and prefix
        if self.logger is None:
            prefix = ''
            logger = logging.root
        else:
            prefix = self.logger.lower() + '/'
            logger = logging.getLogger(self.logger)
        # log data:
        #   > if the parameter is in the stats set, we cache it and log it at info level
        #   > otherwise we don't cache it and log it at debug level
        for name, value in params.items():
            if name in self.parameters:
                to_cache[f'{prefix}{name}'] = value
                logger.info(f'{name} = {value}')
            else:
                logger.debug(f'{name} = {value}')
        self.log(**to_cache)
