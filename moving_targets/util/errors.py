"""Utility functions and classes for errors handling."""
import logging
from typing import Dict


def not_implemented_message(name: str, abstract: bool = True, static: bool = False) -> str:
    """Default error message for not implemented methods.

    :param name:
        The method name.

    :param abstract:
        Whether or not the method is abstract.

    :param static:
        Whether or not the method is static.

    :return:
        A string representing the error message.
    """

    abstract = 'abstract ' if abstract else ''
    static = 'static ' if static else ''
    return f"Please implement {static}{abstract}method '{name}'"


class MissingDependencyError(ModuleNotFoundError):
    """Error class for missing external dependencies which mat be lazily imported."""

    _versions: Dict[str, str] = {
        'cvxpy': '1.1.13',
        'docplex': '2.20.204',
        'gurobipy': '9.1.2',
        'tensorflow': '2.7.0',
        'wandb': '0.12.6'
    }
    """Expected versions (as in project requirements) of each external dependency which may be lazily imported."""

    def __init__(self, package: str):
        """
        :param package:
            The package name.
        """
        version = self._versions.get(package)
        if version is None:
            command = package
            logging.getLogger('MissingDependencyError').warning(f"unexpected lazy dependency '{package}'")
        else:
            command = f'{package}~={version}'
        message = f"This class requires '{package}' in order to be used, please install it via 'pip install {command}'"
        super(MissingDependencyError, self).__init__(message)
