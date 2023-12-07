from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

module = CppExtension(
      'sparse_mm',
      sources=[
            'sparse_mm.cpp'
      ],
      include_dirs=[
            '/data/oscar310118/112-1-PP-Final/pytorch_extension/include/'
      ],
      extra_compile_args=['-O3', '-fopenmp']
)

setup(
      name='sparse_mm',
      ext_modules=[module],
      cmdclass={'build_ext': BuildExtension}
)