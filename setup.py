from distutils.core import setup
from Cython.Build import cythonize

setup(name='Pose recognition', ext_modules=cythonize("api_expression.pyx"))
