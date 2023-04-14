"""
 python setup.py install
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

filename = 'utils.nms'
full_filename = 'utils/nms.pyx'

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [Extension('utils.nms',
                         ['utils/nms.pyx'],
                         language='c++',
                         extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
                         include_dirs=[numpy_include]
                         )
               ]

setup(
    cmdclass={
        'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[np.get_include()])
