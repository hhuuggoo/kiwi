from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("tree_func_c", ["tree_func_c.pyx"], include_dirs=[numpy.get_include()])]

setup(
      name = 'tree_func_c',
    cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules
    )
