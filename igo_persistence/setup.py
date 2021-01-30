from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("persistence", sources=["persistence.pyx"], include_dirs=['.', get_include()])
setup(name="persistence", ext_modules=cythonize([ext]))