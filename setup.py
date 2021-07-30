# for when package is released to PyPi?

from setuptools import setup

setup(
    name="discoprocess",
    description="Python package to pre-process NMR data for machine learning applications.",
    packages=['src/discoprocess'],
    install_requires=[
        'numpy', 
        'pandas',
        'matplotlib',
        'pytest',
        'pytest-mock',
        'openpyxl'
    ], 
    license='MIT',
)