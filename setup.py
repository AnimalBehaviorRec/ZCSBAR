from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        "utils/data_aug.py",
        "utils/FPsort.py",
        "utils/track.py",
        "utils/Track_Demo.py",
        "utils/trackconfig.py"
    ], compiler_directives={"language_level": "3"})
)
