from setuptools import setup, find_packages

setup(
    name="src", 
    version='0.0.1',
    description=DESCRIPTION,
    long_description='Python package for NMR data pre-processing.'
    packages=find_packages(),
    install_requires=[
        'pandas', 
        'numpy', 
        'glob',
        'os',
        're', 
        'shutil'
        
        ], # add any additional packages that 
    # needs to be installed along with your package. Eg: 'caer'
    

)