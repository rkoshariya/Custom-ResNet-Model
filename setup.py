#removed include path directory
#START
from distutils.command import build_ext
from distutils.extension import Extension
from distutils.core import setup
from torch.utils import cpp_extension



setup(name='op',
      ext_modules=[cpp_extension.CppExtension('op', ['op.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)})