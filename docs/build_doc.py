import os
import shutil

if __name__ == '__main__':
    # remove ./html and ./source folders
    shutil.rmtree('./html', ignore_errors=True)
    shutil.rmtree('./source', ignore_errors=True)

    # generate documentation .rst with sphinx-apidoc for main module
    os.mkdir('./source')
    os.system(f'sphinx-apidoc -e -T -M -d 8 -o .\\source ..\\moving_targets')

    # make html
    os.system('sphinx-build . .\\html')
