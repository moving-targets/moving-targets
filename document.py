# Run this script from the origin folder as:
#   > "python document.py" in order to clean previous builds and generate new documentation
#   > "python document.py clean" in order to clean previous builds without generating new documentation

import os
import shutil
import sys

# clear previous documentation
shutil.rmtree('docs/html', ignore_errors=True)
shutil.rmtree('docs/source', ignore_errors=True)

# if clean is not passed, generate new documentation .rst with sphinx-apidoc for main module, then make html
if 'clean' not in sys.argv:
    os.mkdir('docs/source')
    os.system(f'sphinx-apidoc -e -T -M -d 8 -o .\\docs\\source .\\moving_targets')
    os.system('sphinx-build .\\docs .\\docs\\html')
