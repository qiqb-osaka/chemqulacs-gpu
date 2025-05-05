from distutils.core import Extension, setup

from Cython.Build import cythonize

ext = Extension("chemqulacs_cpp", sources=["chemqulacs_cpp.pyx"], include_dirs=["."])
setup(name="chemqulacs_cpp", ext_modules=cythonize([ext]))
