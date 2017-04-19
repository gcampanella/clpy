import numpy as np

from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='clpy',
    version='0.1',
    description='Low-level Cython bindings to the Clp linear programming solver',
    long_description=readme(),
    url='https://github.com/gcampanella/clpy',
    author='Gianluca Campanella',
    author_email='gianluca@campanella.org',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    packages=['clpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    ext_modules=cythonize(
        Extension('clpy.clp', ['clpy/clp.pyx'], include_dirs=[np.get_include()], libraries=['Clp']),
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
            'language_level': 3
        }
    )
)
