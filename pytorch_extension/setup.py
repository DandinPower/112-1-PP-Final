from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

module = CppExtension(
      'sparse_mm',
      sources=[
            'sparse_mm.cpp'
      ],
      include_dirs=[
            'include', 'include/sparse_mm'
      ],
      extra_compile_args=['-O3', '-fopenmp']
)

setup(
      name='sparse_mm',
      ext_modules=[module],
      cmdclass={'build_ext': BuildExtension}
)