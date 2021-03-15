# Liam Kilcommons - University of Colorado, Boulder - Colorado Center for Astrodynamics Research
# May, 2019
# (C) 2019 University of Colorado AES-CCAR-SEDA (Space Environment Data Analysis) Group

import os
import glob

os.environ['DISTUTILS_DEBUG'] = "1"

from setuptools import setup, Extension
from setuptools.command import install as _install

setup(name='esabin',
      version = "0.2.1",
      description = "Bin geophysical data in equal solid angle bins",
      author = "Liam Kilcommons",
      author_email = 'liam.kilcommons@colorado.edu',
      url = "https://github.com/lkilcommons/esabin",
      download_url = "https://github.com/lkilcommons/esabin",
      long_description = "Equal solid angle binning using HDF5 for storage",
      install_requires=['numpy','h5py'],
      packages=['esabin'],
      package_dir={'esabin' : 'esabin'},
      license='LICENSE.txt',
      zip_safe = False,
      classifiers = [
            "Development Status :: 4 - Beta",
            "Topic :: Scientific/Engineering",
            "Intended Audience :: Science/Research",
            "License :: MIT",
            "Natural Language :: English",
            "Programming Language :: Python"
            ]
      )
