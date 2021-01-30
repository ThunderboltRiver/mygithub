from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("alpha", sources=["alpha.pyx"], include_dirs=['.', get_include()])
setup(name="alpha", ext_modules=cythonize([ext]))