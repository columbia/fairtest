#!/usr/bin/env python

from setuptools import setup

setup(name='fairtest',
      version='0.1',
      description='fairtest',
      author='Florian Tramer, Vaggelis Atlidakis',
      author_email='tramer@stanford.edu, vatlidak@cs.columbia.edu',
      package_dir={'': 'src'},
      packages=['fairtest',
                'fairtest.modules',
                'fairtest.modules.bug_report',
                'fairtest.modules.context_discovery',
                'fairtest.modules.metrics',
                'fairtest.modules.statistics',
                'fairtest.service',
                'fairtest.service.helpers',
                'fairtest.utils'],
      test_suite="tests",
      )
