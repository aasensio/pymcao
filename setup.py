#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymcao v0.1


For more information see: https://github.com/aasensio/pymcao
::
    Main Changes in 0.1
    ---------------------
    * Working version
:copyright:
    A. Asensio Ramos    
:license:
    The MIT License (MIT)
"""
from distutils.ccompiler import CCompiler
from distutils.errors import DistutilsExecError, CompileError
from distutils.unixccompiler import UnixCCompiler
from setuptools import find_packages, setup
from setuptools.extension import Extension

import os
import platform
from subprocess import Popen, PIPE
import sys
import numpy
import glob
import re

DOCSTRING = __doc__.strip().split("\n")

tmp = open('pymcao/__init__.py', 'r').read()
author = re.search('__author__ = "([^"]+)"', tmp).group(1)
version = re.search('__version__ = "([^"]+)"', tmp).group(1)

setup_config = dict(
    name='pymcao',
    version=version,
    description=DOCSTRING[0],
    long_description="\n".join(DOCSTRING[2:]),
    author=author,
    author_email='aasensio@iac.es',
    url='https://github.com/aasensio/pymcao',
    license='GNU General Public License, version 3 (GPLv3)',
    platforms='OS Independent',
    install_requires=['numpy','scipy','configobj','h5py','astropy','tqdm'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: Implementation :: CPython",
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords=['MCAO'],
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True
)

if __name__ == "__main__":
    setup(**setup_config)