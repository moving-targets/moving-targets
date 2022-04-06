# Run this script from the origin folder as:
#   > "python setup.py clean" in order to clean previous builds
#   > "python setup.py test" in order to execute all the unittests
#   > "python setup.py sdist" in order to build the library
#
# The package can then be published with:
#   > twin upload dist/*

from setuptools import find_packages, setup

# set up the library metadata and make the build
with open('README.md', 'r') as readme:
    setup(
        name='moving-targets',
        version='0.2.1',
        maintainer='Luca Giuliani',
        maintainer_email='luca.giuliani13@unibo.it',
        author='University of Bologna - DISI',
        description='Moving Targets: a framework for constrained machine learning',
        long_description=readme.read(),
        long_description_content_type='text/markdown',
        packages=find_packages(include=['moving_targets*']),
        python_requires='>=3.7',
        install_requires=['matplotlib~=3.4.3', 'numpy~=1.21.4', 'pandas~=1.3.4', 'scikit-learn~=1.0.1'],
        test_suite='test'
    )
