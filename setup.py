#!/usr/bin/env python

from setuptools import setup

setup(name='fairtest',
      version='0.1',
      description='fairtest',
      author='Florian Tramer, Vaggelis Atlidakis',
      author_email='florian.tramer@epfl.ch, vatlidak@cs.columbia.edu',
      package_dir={'': 'src_new_api'},
      packages=['fairtest', 'fairtest.bugreport',
                'fairtest.bugreport.clustering',
                'fairtest.bugreport.core',
                'fairtest.bugreport.statistics',
                'fairtest.bugreport.helpers',
                'fairtest.bugreport.trees'],
      test_suite="tests",
    )
