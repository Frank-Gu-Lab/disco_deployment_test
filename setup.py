from setuptools import setup, find_packages

setup(
    name="discoprocess",
    description="Python package to pre-process NMR data for machine learning applications.",
    packages=find_packages(),
    install_requires=[
        'numpy', 
        'pandas',
        'matplotlib',
        're',
        'glob',
        'shutil',
        'pytest',
        'pytest-mock',
        'openpyxl'
    ]
)