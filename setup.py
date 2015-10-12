#!/usr/bin/env python

from setuptools import setup

setup(name='fairtest',
      version='0.1',
      description='fairtest',
      author='Florian Tramer, Vaggelis Atlidakis',
      author_email='florian.tramer@epfl.ch, vatlidak@cs.columbia.edu',
      package_dir={'': 'src'},
      packages=['fairtest', 'fairtest.bugreport',
                'fairtest.bugreport.clustering',
                'fairtest.bugreport.core',
                'fairtest.bugreport.statistics',
                'fairtest.bugreport.helpers',
                'fairtest.bugreport.trees'],
      test_suite="tests",
      requires=['numpy', 'pandas', 'prettytable', 'matplotlib', 'pydot', 'ete2',
                'sklearn', 'scipy', 'statsmodels', 'rpy2', 'ipython']
      )
