# Run this script from the origin folder as:
#   > "python setup.py test" in order to execute all the unittests
#   > "python setup.py bdist_wheel" in order to build the library

import shutil

from setuptools import find_packages, setup

# clears previous builds
shutil.rmtree('build', ignore_errors=True)
shutil.rmtree('dist', ignore_errors=True)

# retrieves the description from the readme file
with open('README.md', 'r') as readme:
    long_description = readme.read()

# sets the library metadata up
setup(
    name="moving-targets",
    version="0.1.0",
    maintainer="Luca Giuliani",
    maintainer_email="luca.giuliani13@unibo.it",
    author="University of Bologna - DISI",
    description="Moving Targets: a framework for constrained machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=['moving_targets*']),
    python_requires='>=3.7',
    install_requires=['matplotlib~=3.4.3', 'numpy~=1.21.4', 'pandas~=1.3.4', 'scikit-learn~=1.0.1'],
    test_suite='test'
)
