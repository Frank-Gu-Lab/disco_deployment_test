from setuptools import setup

setup(
    name="discoprocess",
    description="Python package to pre-process NMR data for machine learning applications.",
    author = ['Jeffrey Watchorn', 'Samantha Stuart', 'Jennifer Tram Su'],
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