from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='sparse_mm',
      ext_modules=[cpp_extension.CppExtension('sparse_mm', ['sparse_mm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})