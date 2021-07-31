from setuptools import setup

setup(
    name="discoprocess",
    description="Python package to pre-process NMR data for machine learning applications.",
    version='0.0.1',
    packages=['discoprocess'],
    install_requires=[
        'numpy', 
        'pandas',
        'matplotlib',
        'pytest',
        'pytest-mock',
        'openpyxl'
    ], 
    license='MIT'
)